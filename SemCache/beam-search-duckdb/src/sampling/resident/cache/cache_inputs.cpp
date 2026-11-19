#include "../detail.hpp"
#include "../../common/accuracy.hpp"
#include "semcache/history_features.hpp"

namespace semcache::sampling {
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
  std::vector<std::string> key;
  for (const auto &f : first)
    key.push_back(f.get<std::string>());
  if (layout.contains("right_field_order")) {
    key.push_back("|");
    for (const auto &f : layout.at("right_field_order"))
      key.push_back(f.get<std::string>());
  }
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
  std::vector<std::string> names;
  auto append_unique = [&](const std::string &name) {
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(name);
  };
  append_unique(first[0].get<std::string>());
  if (layout.contains("right_field_order"))
    append_unique(layout["right_field_order"][0].get<std::string>());
  const std::set<std::string> prefix_fields(names.begin(), names.end());
  for (const auto &f : first)
    append_unique(f.get<std::string>());
  if (layout.contains("right_field_order"))
    for (const auto &f : layout["right_field_order"])
      append_unique(f.get<std::string>());
  Json descriptors = Json::object();
  for (const auto &f : op.value("input_field_schema", Json::array())) {
    const auto name = f.at("name").get<std::string>();
    descriptors[name] = f;
    // Coverage describes every consumed field, independently of which one
    // the current layout places first for prefix reuse.
    append_unique(name);
  }
  std::map<std::string, std::shared_ptr<const Json>> summaries;
  // Keep a small catalogue of every field actually consumed by this
  // semantic request.  Base relation fields are normally present in
  // estimated_output_survival; fields produced by an earlier semantic
  // operator are not.  Registering both here prevents generated fields from
  // silently disappearing from coverage reports when they have no summary.
  std::map<std::string, std::string> catalog_keys, catalog_origins,
      catalog_relations, catalog_vkeys;
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
      catalog_relations[name] = relation;
      catalog_vkeys[name] = key;
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
  auto resolve_joint = [&](const Json &order,
                           const std::shared_ptr<const Json> &fallback) {
    if (order.size() < 2)
      return fallback;
    std::string relation;
    std::vector<std::string> vks;
    vks.reserve(order.size());
    for (const auto &raw : order) {
      const auto col = raw.get<std::string>();
      auto rel_it = catalog_relations.find(col);
      auto vk_it = catalog_vkeys.find(col);
      if (rel_it == catalog_relations.end() || vk_it == catalog_vkeys.end())
        return fallback;
      if (relation.empty())
        relation = rel_it->second;
      else if (relation != rel_it->second)
        return fallback;
      vks.push_back(vk_it->second);
    }
    const auto &joints = relations.at(relation).joint_summaries;
    const auto found = joints.find(canonical(Json(vks)));
    return found != joints.end() ? found->second : fallback;
  };
  auto align_joint_alpha = [](Json &alpha, const Json &single_summary,
                              const Json &selected_summary) {
    if (&single_summary == &selected_summary || !alpha.is_object() ||
        !alpha.contains("depths") || !alpha["depths"].is_object())
      return;
    const auto &single_sketches = single_summary.at("sketches");
    auto &alpha_depths = alpha["depths"];
    for (auto it = selected_summary.at("sketches").begin();
         it != selected_summary.at("sketches").end(); ++it)
      if (it.key() != "0" && !it.value().at("entries").empty() &&
          (!single_sketches.contains(it.key()) ||
           single_sketches.at(it.key()).at("entries").empty()))
        alpha_depths.erase(it.key());
  };
  std::string left = first[0];
  if (!summaries.count(left))
    return Json{{"field_depth_catalog", std::move(field_depth_catalog)}};
  observation.summary = resolve_joint(first, summaries[left]);
  std::string system = profile_.value("system_prefix",
                                      std::string("semantic-global-v1")),
              spec = op.value("kind", std::string());
  const auto specs = profile_.find("operator_specifications");
  if (specs != profile_.end() && specs->contains(spec))
    spec = specs->at(spec);
  Json left_alpha = survivals[left];
  align_joint_alpha(left_alpha, *summaries[left], *observation.summary);
  Json result = {

      {"alpha", std::move(left_alpha)},
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
    observation.right_summary =
        resolve_joint(layout["right_field_order"], summaries[right]);
    Json right_alpha = survivals[right];
    align_joint_alpha(right_alpha, *summaries[right],
                      *observation.right_summary);
    result["right_alpha"] = std::move(right_alpha);
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
}
