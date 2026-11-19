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
  if (!confirmed.empty()) {
    update_history_distances();
    rebuild_history_index();
  }
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

}
