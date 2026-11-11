#include "../detail.hpp"
#include "../../common/accuracy.hpp"
#include "semcache/history_features.hpp"

namespace semcache::sampling {
const Json &NativeSampler::Impl::profile_statistic(const std::string &bank,
                                                   const std::string &model,
                                                   const std::string &key,
                                                   const Json *fallback) const {
  static const Json missing = nullptr;
  const auto &default_value = fallback ? *fallback : missing;
  auto entries = profile_.find(bank);
  if (entries == profile_.end())
    return default_value;
  auto models = entries->find(model);
  if (models == entries->end())
    return default_value;
  auto value = models->find(key);
  return value == models->end() ? default_value : value.value();
}

bool NativeSampler::Impl::harvest(NativeSQLBackend *source) {
  if (!source)
    source = backend.get();
  if (!source || (source->events().empty() && source->observations().empty()))
    return false;
  PhaseTimer timer(phase_metrics, "history_and_calibration", trace_timings);
  Json observations = source->drain_observations();
  const auto observation_begin = archived_observations.size();
  for (auto &entry : observations)
    archived_observations.push_back(std::move(entry));
  for (auto index = observation_begin; index < archived_observations.size();
       ++index) {
    const auto &observation = archived_observations[index];
    if (observation.value("purpose", std::string()) ==
        "operator_time_calibration")
      continue;
    Json model = observation.value("model_id", Json(nullptr));
    if (!model.is_null()) {
      std::vector<double> values;
      if (auto found = observation.find("events"); found != observation.end())
        for (const auto &e : *found)
          if (e.value("event", std::string()) == "semantic_row" &&
              e.value("parse_status", std::string()) == "validated" &&
              e.value("sampling_batch_size", std::size_t(1)) == 1 &&
              !(e.value("submit_ns", std::int64_t(-1)) == 0 &&
                e.value("response_ns", std::int64_t(-1)) == 0) &&
              finite_number(e.value("latency_ms", Json(nullptr))))
            values.push_back(e["latency_ms"].get<double>() / 1000);
      calibrate("request_profiles", model.get<std::string>(),
                observation.value("kind", std::string()), values,
                "seconds_per_request", "sampled warm request latency");
      if (observation.contains("task_key"))
        calibrate("task_request_profiles", model.get<std::string>(),
                  observation["task_key"].get<std::string>(), values,
                  "seconds_per_request", "sampled task request latency");
    } else {
      double count = 1;
      for (auto &n : observation.value("inputs", Json::array()))
        count *= n.get<double>();
      if (count && finite_number(observation.value("seconds", Json(nullptr))))
        calibrate("relational_profiles", "",
                  std::to_string(observation.at("operator_id").get<Id>()),
                  {observation["seconds"].get<double>() / count},
                  "seconds_per_input", "sampled materialized SQL stage");
    }
  }
  Json events = source->drain_events();
  const auto event_begin = archived_events.size();
  for (auto &entry : events)
    archived_events.push_back(std::move(entry));
  Json actions = Json::object(), operator_bank = Json::object();
  std::map<Id, std::vector<const Json *>> confirmed;
  std::map<Id, std::pair<std::pair<std::uint64_t, std::uint64_t>, double>>
      starts;
  std::map<Id, std::string> operator_model;
  // Publish all owned events before borrowing any pointers; archive growth
  // can move ordered_json array elements. Exceptions still retain evidence.
  for (auto index = event_begin; index < archived_events.size(); ++index) {
    const auto &e = archived_events[index];
    auto type = e.value("event", std::string());
    if (type == "planning_decision") {
      for (auto &a : e.at("plan_spec").at("actions"))
        actions[std::to_string(a.at("operator_id").get<Id>())] = a;
      for (auto &op : e.at("query_ir").at("semantic_operators"))
        operator_bank[std::to_string(op.at("operator_id").get<Id>())] = op;
      continue;
    }
    if (!e.contains("operator_id"))
      continue;
    Id oid = e["operator_id"];
    std::string key = std::to_string(oid);
    if (!actions.contains(key) ||
        actions[key].value("model_id", Json(nullptr)).is_null())
      continue;
    std::string model = actions[key]["model_id"];
    operator_model[oid] = model;
    double now = e.value("observed_unix_seconds", 0.);
    if (type == "operator_start")
      starts.try_emplace(oid, std::make_pair(loads[model], now));
    bool proved =
        type == "semantic_row" &&
        (e.value("parse_status", std::string()) == "validated" ||
         (e.contains("response_ns") && e["response_ns"].is_number_integer() &&
          e["response_ns"].get<std::int64_t>() >= 0 &&
          e.contains("raw_result") && e["raw_result"].is_string()));
    if (!proved || !e.contains("input_values") ||
        !e["input_values"].is_array() || !e.contains("system_text") ||
        !e.contains("user_text") || !e.contains("operator_specification"))
      continue;
    bool structured = true;
    for (auto &value : e["input_values"])
      structured &= value.is_string();
    if (!structured)
      continue;
    starts.try_emplace(oid, std::make_pair(loads[model], now));
    ++loads[model].first;
    loads[model].second +=
        character_length(e["system_text"].get<std::string>()) +
        character_length(e["user_text"].get<std::string>());
    confirmed[oid].push_back(&e);
  }
  for (auto &entry : confirmed) {
    Id oid = entry.first;
    auto &rows = entry.second;
    auto key = std::to_string(oid);
    auto op = operator_bank.contains(key) ? operator_bank[key] : ops.at(oid),
         action = actions[key];
    auto layout = action.at("layout");
    std::string first = layout.contains("field_order")
                            ? layout["field_order"][0].get<std::string>()
                            : layout["left_field_order"][0].get<std::string>();
    Json names = op.value("input_fields", Json::array());
    if (names.empty()) {
      names = op.value("left_input_fields", Json::array());
      for (auto &n : op.value("right_input_fields", Json::array()))
        names.push_back(n);
    }
    auto position = std::find(names.begin(), names.end(), Json(first));
    if (position == names.end())
      throw std::invalid_argument(
          "History first field does not match request fields");
    // A stage may contain multi-sample prompts and ordinary singleton
    // fallbacks. Keep their distinct physical prefixes separate, in
    // first-observed order.
    std::map<std::string, std::size_t> indices;
    std::vector<std::pair<std::string, Json>> groups;
    for (const auto *entry : rows) {
      const auto &row = *entry;
      auto pre = row.contains("input_prefix")
                     ? canonical(Json::array({"prompt-v4", row["system_text"],
                                              row["input_prefix"]}))
                     : prefix_namespace(op, action, row["system_text"],
                                        row["operator_specification"]);
      auto found = indices.find(pre);
      if (found == indices.end()) {
        auto index = groups.size();
        indices.emplace(pre, index);
        groups.emplace_back(std::move(pre), Json{{first, Json::array()}});
        found = indices.find(groups.back().first);
      }
      groups[found->second].second[first].push_back(
          row["input_values"][position - names.begin()]);
    }
    auto before = starts.at(oid);
    std::string model = operator_model.at(oid);
    for (auto &group : groups) {
      auto built = build_summaries(group.second, sketch_config);
      Json history = {
          {"id", digest(Json::array({"native-history", ++history_serial, model,
                                     oid, before.second}))},
          {"model_id", model},
          {"pre", group.first},
          {"operator",
           digest(Json::array({op["kind"], op["task"],
                               op.value("output_schema", Json::array())}))},
          {"first_field", first},
          {"age_seconds", 0.},
          {"started_at", before.second},
          {"requests_before", before.first.first},
          {"characters_before", before.first.second},
          {"requests_since", loads[model].first - before.first.first},
          {"characters_since", loads[model].second - before.first.second},
          {"summary", built["summaries"][first]},
          {"capacity", built["summaries"][first]["metadata"]["capacity"]}};
      history_summaries[history.at("id").get<std::string>()] =
          std::make_shared<const Json>(history.at("summary"));
      history_.push_back(std::move(history));
      ++history_version;
    }
    evict_history(model, before.second);
  }
  update_history_distances();
  rebuild_history_index();
  return observation_begin != archived_observations.size() ||
         event_begin != archived_events.size();
}
void NativeSampler::Impl::calibrate(const std::string &bank,
                                    const std::string &owner,
                                    const std::string &key,
                                    const std::vector<double> &values,
                                    const std::string &metric,
                                    const std::string &basis) {
  if (values.empty() || !settings.value("collect_calibration", true))
    return;
  Json *entries = &profile_[bank];
  if (entries->is_null())
    *entries = Json::object();
  if (!owner.empty()) {
    entries = &(*entries)[owner];
    if (entries->is_null())
      *entries = Json::object();
  }
  const bool counted = bank == "request_profiles";
  if (entries->contains(key) && !counted &&
      (*entries)[key].value("basis", std::string()) != basis)
    return;
  const auto identity = std::make_tuple(bank, owner, key);
  auto position = moments.find(identity);
  Moment seed;
  if (position == moments.end() && counted && entries->contains(key)) {
    const auto &old = entries->at(key);
    const auto &n = old.at("samples");
    if (!n.is_number_integer() || n.get<double>() < 1 ||
        !finite_number(old.at(metric)) || old.at(metric).get<double>() < 0 ||
        !finite_number(old.at("latency_m2")) ||
        old.at("latency_m2").get<double>() < 0)
      throw std::invalid_argument("Invalid counted latency prior");
    seed.n = n.get<std::size_t>();
    seed.mean = old.at(metric).get<double>();
    seed.m2 = old.at("latency_m2").get<double>();
  }
  auto &m = position == moments.end()
                ? moments.emplace(identity, seed).first->second
                : position->second;
  for (double value : values) {
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("Invalid sampled latency");
    ++m.n;
    double delta = value - m.mean;
    m.mean += delta / m.n;
    m.m2 += delta * (value - m.mean);
  }
  Json result = {{metric, m.mean},
                 {"basis", basis},
                 {"samples", m.n},
                 {"latency_m2", m.m2}};
  if (counted)
    result["basis"] = "counted initial and online atomic request latency";
  if (metric == "seconds_per_request")
    result["latency_cv"] =
        m.n > 1 && m.mean > 0
            ? Json(std::sqrt(std::max(0., m.m2) / (m.n - 1)) / m.mean)
            : Json(nullptr);
  (*entries)[key] = result;
}

void NativeSampler::Impl::update_history_distances() {
  double now = imported.value("history_now", Json(nullptr)).is_null()
                   ? std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()
                   : imported["history_now"].get<double>();
  for (auto &e : history_) {
    e["age_seconds"] = std::max(0., now - e.value("started_at", now));
    std::string model = e.at("model_id");
    auto load = loads[model];
    if (e.contains("requests_before") && !e["requests_before"].is_null())
      e["requests_since"] =
          load.first - e["requests_before"].get<std::uint64_t>();
    if (e.contains("characters_before") && !e["characters_before"].is_null())
      e["characters_since"] =
          load.second - e["characters_before"].get<std::uint64_t>();
  }
}

void NativeSampler::Impl::rebuild_history_index() {
  history_by_model.clear();
  history_by_prefix.clear();
  history_incomplete_models.clear();
  std::set<std::string> retained;
  for (const auto &entry : history_) {
    const auto model = entry.value("model_id", std::string());
    history_by_model[model].push_back(&entry);
    retained.insert(entry.at("id").get<std::string>());
    const auto pre = entry.find("pre");
    if (pre == entry.end() || pre->is_null())
      history_incomplete_models.insert(model);
    else
      history_by_prefix[model][*pre].push_back(&entry);
  }
  for (auto entry = history_summaries.begin();
       entry != history_summaries.end();)
    if (!retained.count(entry->first))
      entry = history_summaries.erase(entry);
    else
      ++entry;
}

void NativeSampler::Impl::evict_history(const std::string &model, double now) {
  std::vector<std::size_t> indices;
  std::array<double, 3> scales{};
  for (std::size_t i = 0; i < history_.size(); ++i)
    if (history_[i]["model_id"] == model) {
      indices.push_back(i);
      now = std::max(now, history_[i].value("started_at", 0.));
    }
  auto capacity = imported.value("history_capacity", std::size_t(256));
  if (indices.size() <= capacity)
    return;
  std::map<std::size_t, std::array<double, 3>> distance;
  for (auto i : indices) {
    auto &e = history_[i];
    distance[i] = {std::max(0., now - e.value("started_at", now)),
                   e.value("requests_before", Json(nullptr)).is_null()
                       ? 0.
                       : double(loads[model].first -
                                e["requests_before"].get<std::uint64_t>()),
                   e.value("characters_before", Json(nullptr)).is_null()
                       ? 0.
                       : double(loads[model].second -
                                e["characters_before"].get<std::uint64_t>())};
    for (int j = 0; j < 3; ++j)
      scales[j] = std::max(scales[j], distance[i][j]);
  }
  auto score = [&](std::size_t i) {
    double value = 0;
    for (int j = 0; j < 3; ++j)
      if (scales[j])
        value += distance[i][j] / scales[j];
    return std::make_tuple(value, -history_[i].value("started_at", 0.),
                           history_[i]["id"].get<std::string>());
  };
  std::sort(indices.begin(), indices.end(),
            [&](auto a, auto b) { return score(a) > score(b); });
  std::set<std::size_t> remove(indices.begin(),
                               indices.begin() + indices.size() - capacity);
  Json kept = Json::array();
  for (std::size_t i = 0; i < history_.size(); ++i)
    if (!remove.count(i))
      kept.push_back(history_[i]);
  history_ = kept;
}

Json NativeSampler::Impl::cache_inputs(const Json &op, const Json &action,
                                       const SampleInputs &inputs,
                                       bool complete, Observation &observation,
                                       const OwnedInputs *input_boundaries,
                                       const Json *previous_survivals) const {
  PhaseTimer timer(phase_metrics, "cache_inputs", trace_timings);
  // Prior refreshes can carry a view's previously frozen alpha. Do not let
  // those overlays populate the cache for other views of the physical core.
  if (previous_survivals)
    return build_cache_inputs(op, action, inputs, complete, observation,
                              input_boundaries, previous_survivals);
  const auto &layout = action.at("layout");
  const auto &first = layout.contains("field_order")
                          ? layout.at("field_order")
                          : layout.value("left_field_order", Json::array());
  if (first.empty())
    return Json::object();
  std::vector<std::string> key{first[0].get<std::string>()};
  if (layout.contains("right_field_order"))
    key.push_back(layout.at("right_field_order")[0].get<std::string>());
  auto &views = observation.core->cache_input_views;
  auto found = views.find(key);
  if (found == views.end()) {
    auto result = build_cache_inputs(op, action, inputs, complete, observation,
                                     input_boundaries, nullptr);
    auto physical = result;
    physical.erase("pre");
    physical.erase("pre_length");
    views.emplace(std::move(key),
                  resident_detail::CacheInputView{
                      std::move(physical), observation.summary,
                      observation.right_summary, observation.conditional_plan,
                      observation.cache_relation, observation.cache_field});
    return result;
  }
  ++cache_input_view_hits;
  const auto &view = found->second;
  observation.summary = view.summary;
  observation.right_summary = view.right_summary;
  observation.conditional_plan = view.conditional_plan;
  observation.cache_relation = view.relation;
  observation.cache_field = view.field;
  auto result = view.statistics;
  if (result.contains("alpha")) {
    auto spec = op.value("kind", std::string());
    const auto specs = profile_.find("operator_specifications");
    if (specs != profile_.end() && specs->contains(spec))
      spec = specs->at(spec);
    result["pre"] = prefix_namespace(
        op, action,
        profile_.value("system_prefix", std::string("semantic-global-v1")),
        spec);
    result["pre_length"] = prefix_length(result.at("pre").get<std::string>());
  }
  return result;
}

Json NativeSampler::Impl::build_cache_inputs(
    const Json &op, const Json &action, const SampleInputs &inputs,
    bool complete, Observation &observation,
    const OwnedInputs *input_boundaries, const Json *previous_survivals) const {
  const auto &layout = action.at("layout");
  auto first = layout.value("field_order",
                            layout.value("left_field_order", Json::array()));
  if (first.empty())
    return Json::object();
  std::vector<std::string> names{first[0].get<std::string>()};
  if (layout.contains("right_field_order"))
    names.push_back(layout["right_field_order"][0].get<std::string>());
  const std::set<std::string> prefix_fields(names.begin(), names.end());
  Json descriptors = Json::object();
  for (const auto &f : op.value("input_field_schema", Json::array())) {
    const auto name = f.at("name").get<std::string>();
    descriptors[name] = f;
    // Coverage describes every consumed field, independently of which one
    // the current layout places first for prefix reuse.
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(name);
  }
  std::map<std::string, std::shared_ptr<const Json>> summaries;
  // Keep a small catalogue of every field actually consumed by this
  // semantic request.  Base relation fields are normally present in
  // estimated_output_survival; fields produced by an earlier semantic
  // operator are not.  Registering both here prevents generated fields from
  // silently disappearing from coverage reports when they have no summary.
  std::map<std::string, std::string> catalog_keys, catalog_origins;
  Json survivals = Json::object();
  bool estimated_survival = false;
  Json estimated_fields = Json::array();
  std::vector<NativeFieldValues> conditional;
  std::vector<std::optional<std::set<std::string>>> input_columns(
      inputs.size());
  std::vector<std::optional<Origins>> input_origins(inputs.size());
  for (auto &name : names) {
    std::string key = descriptors.contains(name)
                          ? descriptors[name].value("value_key", name)
                          : name;
    const Json *source = nullptr;
    std::size_t source_index = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!input_columns[i])
        input_columns[i] =
            row_count(inputs[i]) ? columns(inputs[i]) : std::set<std::string>{};
      if (input_columns[i]->count(key)) {
        source = &inputs[i];
        source_index = i;
        break;
      }
    }
    std::set<std::string> origin;
    if (source && !input_origins[source_index])
      input_origins[source_index] = origins(*source);
    const Origins empty_origins;
    const auto &lineages =
        source ? *input_origins[source_index] : empty_origins;
    for (auto &entry : lineages)
      if (relations.at(entry.first).summaries.count(key))
        origin.insert(entry.first);
    if (origin.empty())
      for (auto &entry : relations)
        if (entry.second.summaries.count(key)) {
          auto scope = op.value("scope", Json::array());
          if (scope.empty() || std::find(scope.begin(), scope.end(),
                                         entry.second.id) != scope.end())
            origin.insert(entry.first);
        }
    if (origin.size() == 1) {
      auto relation = *origin.begin();
      catalog_keys[name] = relation + ":" + key;
      catalog_origins[name] = "base";
      if (name == names.front()) {
        observation.cache_relation = relation;
        observation.cache_field = key;
      }
      const bool incomplete_matches = observation.core->input_match_incomplete;
      if (previous_survivals) {
        const auto slot = name == names.front() ? "alpha" : "right_alpha";
        survivals[name] = previous_survivals->value(slot, Json(nullptr));
      } else {
        Json evidence;
        const auto &selected = observation.core->cohort->selected.at(relation);
        if (observation.core->join_evidence) {
          const auto &included = observation.core->join_evidence->arrived;
          evidence =
              sampled_field_rates(relation, key, included, included, false,
                                  complete, incomplete_matches, &selected);
        } else if (incomplete_matches ||
                   (source && sampled_population(*source))) {
          const resident_detail::SourceEvidence included(
              source ? SampleInputs{source} : SampleInputs{});
          evidence =
              sampled_field_rates(relation, key, included, included, false,
                                  complete, incomplete_matches, &selected);
        } else {
          const auto reached = lineages.find(relation);
          const std::set<std::string> empty;
          evidence =
              field_rates(relation, key,
                          reached == lineages.end() ? empty : reached->second,
                          std::nullopt, complete, true, &selected);
        }
        Json depth = Json::object();
        for (auto it = evidence.begin(); it != evidence.end(); ++it)
          if (it.key() != "0")
            depth[it.key()] = it.value()["rate"];
        survivals[name] = {{"total", evidence["0"]["rate"]}, {"depths", depth}};
      }
      const auto estimated =
          observation.core->estimated_survivals.find(relation + ":" + key);
      if (estimated != observation.core->estimated_survivals.end()) {
        Json depth = Json::object();
        for (auto d = estimated->begin(); d != estimated->end(); ++d)
          if (d.key() != "0")
            depth[d.key()] = d.value();
        survivals[name] = {{"total", estimated->value("0", Json(nullptr))},
                           {"depths", std::move(depth)}};
        if (prefix_fields.count(name)) {
          estimated_survival = true;
          estimated_fields.push_back(name);
        }
      }
      summaries[name] = relations.at(relation).summaries.at(key);
    } else if (source) {
      catalog_keys[name] = "generated:" + key;
      catalog_origins[name] = "generated";
      auto *p = source;
      conditional.push_back({name, row_count(*p), [p, key](std::size_t i) {
                               const auto *value = find_value(*p, i, key);
                               if (!value)
                                 throw std::invalid_argument(
                                     "Missing conditional summary field: " +
                                     key);
                               return serialize_value(*value);
                             }});
      const auto inherited =
          observation.core->estimated_survivals.find("generated:" + key);
      if (inherited != observation.core->estimated_survivals.end()) {
        Json depths = Json::object();
        for (auto d = inherited->begin(); d != inherited->end(); ++d)
          if (d.key() != "0")
            depths[d.key()] = d.value();
        survivals[name] = {{"total", inherited->value("0", Json(nullptr))},
                           {"depths", std::move(depths)}};
      } else
        survivals[name] = 1.;
    } else {
      // Keep an explicit pending coordinate for a field that is required by
      // the layout but is not available in the current input boundary.  The
      // sampler cannot invent its depth sketch, but diagnostics must expose
      // the missing evidence instead of reporting a complete denominator.
      catalog_keys[name] = "missing:" + key;
      catalog_origins[name] = "unknown";
    }
  }
  if (!conditional.empty()) {
    std::sort(conditional.begin(), conditional.end(),
              [](const auto &a, const auto &b) { return a.name < b.name; });
    Json field_names = Json::array();
    for (auto &field : conditional)
      field_names.push_back(field.name);
    auto key =
        std::to_string(observation.core->serial) + canonical(field_names);
    auto found = conditional_builds.find(key);
    if (found == conditional_builds.end()) {
      auto built = std::make_shared<const Json>(
          build_summaries(conditional, sketch_config));
      if (conditional_builds.size() >= 128)
        conditional_builds.erase(conditional_builds.begin());
      found = conditional_builds.emplace(key, built).first;
    }
    auto built = found->second;
    for (auto it = built->at("summaries").begin();
         it != built->at("summaries").end(); ++it)
      summaries[it.key()] = std::shared_ptr<const Json>(built, &it.value());
    observation.conditional_plan =
        std::shared_ptr<const Json>(built, &built->at("plan"));
  }
  for (const auto &[name, origin] : catalog_origins)
    if (origin == "generated" && summaries.count(name) &&
        survivals[name].is_object()) {
      const auto &depths = survivals[name]["depths"];
      Json aligned = Json::object();
      for (auto depth = summaries.at(name)->at("sketches").begin();
           depth != summaries.at(name)->at("sketches").end(); ++depth)
        aligned[depth.key()] =
            depth.value().at("entries").empty()
                ? Json(0.)
                : depths.value(depth.key(),
                               survivals[name].value("total", Json(nullptr)));
      survivals[name]["depths"] = std::move(aligned);
    }
  Json field_depth_catalog = Json::object();
  for (const auto &[name, catalog_key] : catalog_keys) {
    const auto &descriptor = descriptors.value(name, Json::object());
    Json entry = {{"origin", catalog_origins[name]},
                  {"field", descriptor.value("value_key", name)},
                  {"depths", Json::array()}};
    const auto summary = summaries.find(name);
    std::map<std::string, bool> depth_support;
    if (summary != summaries.end() && summary->second &&
        summary->second->contains("sketches")) {
      for (auto depth = summary->second->at("sketches").begin();
           depth != summary->second->at("sketches").end(); ++depth) {
        // A sketch with no entries has no reachable prefix coordinate.  Keep
        // the empty-field anchor, which is always meaningful.
        depth_support[depth.key()] =
            depth.key() == "0" ||
            !depth->value("entries", Json::array()).empty();
      }
    }
    const auto &core = *observation.core;
    const auto field = entry.at("field").get<std::string>();
    for (const auto &[depth, supported] : depth_support)
      if (supported)
        entry["depths"].push_back(depth);
    entry["pending"] = depth_support.empty();
    if (entry.at("origin") == "generated" && !entry.at("pending").get<bool>()) {
      Json evidence = Json::object();
      if (core.expression_statistics &&
          core.expression_statistics->contains("generated:" + field))
        evidence = core.expression_statistics->at("generated:" + field);
      static const Json empty = Json::object();
      const auto &info = core.statistics ? *core.statistics : empty;
      const auto rates =
          blend_depths(core, info, "generated", field, depth_support, evidence);
      Json output_rates = rates;
      const auto incoming = core.estimated_survivals.find("generated:" + field);
      if (incoming != core.estimated_survivals.end())
        for (auto d = output_rates.begin(); d != output_rates.end(); ++d) {
          const auto factor =
              incoming->value(d.key(), incoming->value("0", Json(nullptr)));
          d.value() = d.value() == 0 || factor == 0 ? Json(0.)
                      : finite_number(d.value()) && finite_number(factor)
                          ? Json(d.value().get<double>() * factor.get<double>())
                          : Json(nullptr);
        }
      entry["depth_output_survival"] = std::move(output_rates);
      entry["depth_sources"] =
          depth_selectivity_sources(core, "generated", field);
      entry["depth_fusion"] =
          depth_selectivity_fusion(core, "generated", field);
      entry["survival_domain"] = "independent generated-input population; "
                                 "predecessor depth-rate product";
    }
    const auto output_key =
        entry.contains("catalog_evidence") ? "generated:" + field : catalog_key;
    field_depth_catalog[output_key] = std::move(entry);
  }
  std::string left = first[0];
  if (!summaries.count(left))
    return Json{{"field_depth_catalog", std::move(field_depth_catalog)}};
  observation.summary = summaries[left];
  std::string system = profile_.value("system_prefix",
                                      std::string("semantic-global-v1")),
              spec = op.value("kind", std::string());
  const auto specs = profile_.find("operator_specifications");
  if (specs != profile_.end() && specs->contains(spec))
    spec = specs->at(spec);
  Json result = {

      {"alpha", survivals[left]},
      {"pre", prefix_namespace(op, action, system, spec)},
      {"basis",
       "independent field summary with predecessor depth-rate product"},
      {"conditional_sketch_plan", nullptr}};
  result["pre_length"] = prefix_length(result.at("pre").get<std::string>());
  result["input_survival_estimated"] = estimated_survival;
  result["estimated_fields"] = std::move(estimated_fields);
  result["field_depth_catalog"] = std::move(field_depth_catalog);
  if (estimated_survival) {
    result["basis"] =
        "base-field summary with approximate selectivity-recursed survival";
    for (const auto *key :
         {"alpha", "right_alpha", "l_self", "l_existing", "l_previous"})
      result["reliability"][key] = .1;
  }
  if (!observation.cache_relation.empty())
    result["cache_field_origin"] =
        Json::array({observation.cache_relation, observation.cache_field});
  if (layout.contains("right_field_order")) {
    std::string right = layout["right_field_order"][0];
    if (!summaries.count(right))
      return result;
    observation.right_summary = summaries[right];
    result["right_alpha"] = survivals[right];
    double average = 0;
    if (!inputs.empty() && row_count(inputs[0])) {
      Json measured;
      if (!input_boundaries)
        measured = boundary(inputs[0]);
      const auto &fields =
          input_boundaries ? input_boundaries->at(0)->at(1) : measured[1];
      for (auto &f : op.value("left_input_fields", Json::array())) {
        std::string n = f, key = descriptors.contains(n)
                                     ? descriptors[n].value("value_key", n)
                                     : n;
        average +=
            fields.value(key, Json::object()).value("average_length", 0.);
      }
    }
    result["left_total_length"] =
        average * summaries[left]->at("count").get<double>();
  }
  return result;
}

void NativeSampler::Impl::cache_edges(StateData &data) {
  PhaseTimer timer(phase_metrics, "cache_edges", trace_timings);
  std::map<Id, std::set<Id>> ancestors;
  auto nodes = data.state->nodes();
  for (auto &node : nodes) {
    for (auto child : node->children) {
      ancestors[node->node_id].insert(child);
      ancestors[node->node_id].insert(ancestors[child].begin(),
                                      ancestors[child].end());
    }
    if (!node->operator_id)
      continue;
    const auto &op = ops.at(*node->operator_id);
    if (!semantic_operator(op))
      continue;
    const auto &action = action_for(data.state, *node->operator_id);
    auto &overlay = [&]() -> Json & {
      PhaseTimer detach(phase_metrics, "cache_edges_detach", trace_timings);
      return data.stats.at(node->node_id).write();
    }();
    std::vector<Json> sizes;
    for (auto child : node->children)
      sizes.push_back(
          data.stats.at(child).read().value("output_rows", Json(nullptr)));
    propagate_inplace(op, overlay, sizes);
    const auto &observation = *data.observations.at(node->node_id);
    overlay.erase("result_cache");
    const auto &info = observation.statistics.read();
    if (info.value("survival_estimated", false)) {
      overlay["output_rows"] = info.value("output_rows", Json(nullptr));
      overlay["selectivity"] = info.value("selectivity", Json(nullptr));
    }
    const auto &quality = accuracy_evidence(
        profile_statistic("accuracy", action.at("model_id"), op.at("kind")),
        action.at("layout"));
    const auto &agreement = quality.is_object() && quality.contains("value")
                                ? quality.at("value")
                                : quality;
    if (finite_number(agreement))
      overlay["accuracy"] = agreement;
    else
      overlay.erase("accuracy");
    if (info.value("pair_input", false) &&
        !info.value("pair_request_summary", false)) {
      overlay["l_self"] = nullptr;
      overlay["l_existing"] = Json::object();
      overlay["l_previous"] = Json::object();
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          Json::array({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
      continue;
    }
    if (!observation.summary) {
      // Precision/budget annotations may survive a replay. Cache facts cannot
      // survive loss of the current input evidence, even if once known.
      for (auto key : {"alpha", "right_alpha", "pre", "pre_length",
                       "left_total_length", "conditional_sketch_plan", "basis"})
        overlay.erase(key);
      overlay["l_self"] = nullptr;
      overlay["l_existing"] = Json::object();
      overlay["l_previous"] = Json::object();
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          Json::array({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
      continue;
    }
    auto alpha = info.value("alpha", Json(1.));
    if (observation.core->input_match_incomplete)
      for (const auto *key : {"l_self", "l_existing", "l_previous"})
        overlay["reliability"][key] = 0.;
    bool right = observation.right_summary && !info.value("pair_input", false);
    const auto requests = overlay.value("input_rows", Json(nullptr));
    const auto sides = overlay.value("input_side_rows", Json::array());
    const Json left_requests =
                   right && sides.size() == 2 ? sides[0] : Json(nullptr),
               right_requests =
                   right && sides.size() == 2 ? sides[1] : Json(nullptr);
    const auto model = action.at("model_id").get<std::string>();
    {
      PhaseTimer self_timer(phase_metrics, "cache_edges_self", trace_timings);
      overlay["l_self"] =
          right
              ? reuse_cache.join(observation.summary, observation.right_summary,
                                 info.value("left_total_length", Json(nullptr)),
                                 alpha, info.value("right_alpha", Json(1.)),
                                 left_requests, right_requests)
              : reuse_cache.self(observation.summary, alpha, requests);
    }
    overlay["l_existing"] = Json::object();
    overlay["l_previous"] = Json::object();
    overlay["cache_sources_complete"] =
        !info.value("pre", Json(nullptr)).is_null();
    overlay["cache_dependence"] = Json::object();
    Json right_rate = right ? survival_rate(info.value("right_alpha", Json(1.)))
                            : Json(nullptr);
    if (right && right_rate.is_null()) {
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          semcache::statistics::existing_history_features(history_, overlay,
                                                          model);
      continue;
    }
    const auto pre = info.value("pre", Json(nullptr));
    semcache::statistics::ExistingHistoryFeatures packed_history(pre, model);
    const auto indexed_history = history_by_model.find(model);
    if (indexed_history != history_by_model.end())
      for (const auto *source : indexed_history->second)
        packed_history.observe(*source);
    auto edge = [&](const std::shared_ptr<const Json> &source,
                    const Json &source_alpha,
                    const CrossDependence &dependence = CrossDependence{}) {
      auto value = reuse_cache.cross(source, observation.summary, source_alpha,
                                     alpha, dependence, requests);
      return value;
    };
    {
      PhaseTimer history_timer(phase_metrics, "cache_edges_existing",
                               trace_timings);
      if (history_incomplete_models.count(model))
        overlay["cache_sources_complete"] = false;
      const auto model_history = history_by_prefix.find(model);
      if (model_history != history_by_prefix.end()) {
        const auto matching = model_history->second.find(pre);
        if (matching != model_history->second.end())
          for (const auto *source : matching->second) {
            const auto id = source->at("id").get<std::string>();
            overlay["l_existing"][id] =
                source->contains("summary") && !source->at("summary").is_null()
                    ? edge(history_summaries.at(id), 1.)
                    : Json(nullptr);
          }
      }
    }
    {
      PhaseTimer previous_timer(phase_metrics, "cache_edges_previous",
                                trace_timings);
      for (auto &old : nodes) {
        if (!old->operator_id || !ancestors[node->node_id].count(old->node_id))
          continue;
        const auto &old_action = action_for(data.state, *old->operator_id);
        if (old_action.value("model_id", Json(nullptr)) != action["model_id"])
          continue;
        const auto &old_observation = *data.observations.at(old->node_id);
        const auto &source = old_observation.statistics.read();
        if (source.value("pre", Json(nullptr)).is_null())
          overlay["cache_sources_complete"] = false;
        else if (source.value("pre", Json(nullptr)) ==
                 info.value("pre", Json(nullptr))) {
          auto value =
              old_observation.summary
                  ? edge(old_observation.summary,
                         source.value("alpha", Json(1.)),
                         cache_dependence(old_observation, observation))
                  : Json(nullptr);
          overlay["cache_dependence"][std::to_string(*old->operator_id)] =
              old_observation.cache_relation.empty() ||
                      observation.cache_relation.empty()
                  ? "unknown-field independent approximation"
              : old_observation.cache_relation != observation.cache_relation
                  ? "cross-relation independent approximation"
              : old_observation.cache_field == observation.cache_field
                  ? "same-field lineage containment"
              : relations.at(observation.cache_relation).offline_cooccurrence &&
                      relations.at(observation.cache_relation)
                          .offline_columns.count(old_observation.cache_field) &&
                      relations.at(observation.cache_relation)
                          .offline_columns.count(observation.cache_field)
                  ? "same-relation offline conditional row samples"
                  : "same-relation offline statistics unavailable; independent "
                    "approximation";
          overlay["l_previous"][std::to_string(*old->operator_id)] = value;
        }
      }
    }
    overlay["l_existing_features"] =
        packed_history.finish(overlay["l_existing"]);
  }
}
}
