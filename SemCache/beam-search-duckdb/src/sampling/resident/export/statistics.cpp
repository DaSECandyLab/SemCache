#include "../detail.hpp"
#include "../../common/prefix_selectivity.hpp"
#include "semcache/prefix_cooccurrence.hpp"

namespace semcache::sampling {
std::map<Id, std::map<std::string, SharedSamples>>
NativeSampler::statistic_fields(const StatePtr &state) const {
  auto &data = impl->states.at(state->handle_id);
  std::map<Id, std::map<std::string, SharedSamples>> result;
  for (const auto &entry : data.observations) {
    auto &fields = result[entry.first];
    const auto &observation = entry.second;
    for (auto it = observation->core->statistics->begin();
         it != observation->core->statistics->end(); ++it)
      fields[it.key()] =
          SharedSamples(observation->core->statistics, &it.value());
    const auto statistics = observation->statistics.handle();
    for (auto it = statistics->begin(); it != statistics->end(); ++it)
      fields[it.key()] = SharedSamples(statistics, &it.value());
    if (observation->summary)
      fields["summary"] = observation->summary;
    if (observation->right_summary)
      fields["right_summary"] = observation->right_summary;
    if (observation->conditional_plan)
      fields["conditional_sketch_plan"] = observation->conditional_plan;
    const auto overlay = data.stats.at(entry.first).handle();
    for (auto it = overlay->begin(); it != overlay->end(); ++it)
      fields[it.key()] = SharedSamples(overlay, &it.value());
  }
  return result;
}
Json NativeSampler::full_statistics(const StatePtr &state) const {
  auto &data = impl->states.at(state->handle_id);
  Json result = Json::object();
  for (auto &entry : data.observations) {
    auto nid = std::to_string(entry.first);
    result[nid] = *entry.second->core->statistics;
    result[nid].update(entry.second->statistics.read());
    if (entry.second->summary)
      result[nid]["summary"] = *entry.second->summary;
    if (entry.second->right_summary)
      result[nid]["right_summary"] = *entry.second->right_summary;
    if (entry.second->conditional_plan)
      result[nid]["conditional_sketch_plan"] = *entry.second->conditional_plan;
    result[nid].update(data.stats.at(entry.first).read());
  }
  return result;
}
Json NativeSampler::statistics(const StatePtr &state) const {
  Json result = Json::object();
  for (const auto &entry : impl->states.at(state->handle_id).stats)
    result[std::to_string(entry.first)] = entry.second.read();
  return result;
}
std::map<Id, SharedSamples>
NativeSampler::statistic_blocks(const StatePtr &state) const {
  std::map<Id, SharedSamples> result;
  for (const auto &entry : impl->states.at(state->handle_id).stats)
    result.emplace(entry.first, entry.second.handle());
  return result;
}
Json NativeSampler::cache(const StatePtr &) const {
  Json history = Json::array(), profile = Json::object();
  for (auto &e : impl->history_) {
    Json slim = Json::object();
    for (auto it = e.begin(); it != e.end(); ++it)
      if (it.key() != "summary")
        slim[it.key()] = it.value();
    history.push_back(slim);
  }
  for (auto *key :
       {"models", "accuracy", "task_accuracy", "request_profiles",
        "task_request_profiles", "relational_profiles", "constraints",
        "system_prefix", "operator_specifications", "cost_seconds_per_unit"})
    if (impl->profile_.contains(key))
      profile[key] = impl->profile_[key];
  return {{"history", history},
          {"profile", profile},
          {"version", impl->history_version}};
}
Json NativeSampler::samples(const StatePtr &state) const {
  Json result = Json::object();
  for (auto &e : impl->states.at(state->handle_id).observations)
    result[std::to_string(e.first)] = *e.second->rows;
  return result;
}
Json NativeSampler::profile() const {
  Json result = Json::object();
  static const std::set<std::string> immutable{
      "prefix_statistics", "conditional_inputs", "statistics",
      "node_statistics", "history"};
  for (auto it = impl->profile_.begin(); it != impl->profile_.end(); ++it)
    if (!immutable.count(it.key()))
      result[it.key()] = it.value();
  result["sampling_strategy"] = "independent_base";
  // Freeze only the required field-pair estimates, not another copy of the
  // offline row bank. This also covers alternate first fields/layouts that
  // were not visited by the preparation traversal.
  result["prefix_cooccurrences"] = Json::object();
  std::map<std::string, std::set<std::string>> used_fields;
  for (const auto &[_, op] : impl->ops)
    if (semantic_operator(op))
      for (const auto &field : input_keys(op))
        for (const auto &[relation, source] : impl->relations)
          if (source.summaries.count(field) &&
              source.offline_columns.count(field))
            used_fields[relation].insert(field);
  for (const auto &[relation, fields] : used_fields) {
    const auto &source = impl->relations.at(relation);
    if (!source.offline_cooccurrence)
      continue;
    for (auto f = fields.begin(); f != fields.end(); ++f)
      for (auto g = std::next(f); g != fields.end(); ++g) {
        const auto key = std::make_tuple(relation, *f, *g);
        auto found = impl->prefix_cooccurrences.find(key);
        if (found == impl->prefix_cooccurrences.end())
          found =
              impl->prefix_cooccurrences
                  .emplace(key, std::make_shared<const Json>(
                                    semcache::statistics::offline_prefix_common(
                                        *source.offline_cooccurrence,
                                        *source.summaries.at(*f),
                                        *source.summaries.at(*g),
                                        source.offline_columns.at(*f),
                                        source.offline_columns.at(*g))))
                  .first;
        result["prefix_cooccurrences"]
              [canonical(Json::array({relation, *f, *g}))] = *found->second;
      }
  }
  result["operator_statistics"] = Json::object();
  result["source_statistics"] = Json::object();
  std::map<Id, double> strengths;
  for (const auto &state_entry : impl->states)
    for (const auto &observation_entry : state_entry.second.observations) {
      const auto &observation = observation_entry.second;
      const auto &core = *observation->core;
      if (!core.node->operator_id) {
        Json record = *core.statistics;
        if (impl->sample_depth_selectivity || impl->ensure_prefix_depth_rates) {
          const auto relation = node_relkey(*core.node->relations.begin());
          const auto &source = impl->relations.at(relation);
          for (const auto &[field, summary] : source.summaries) {
            auto &rates =
                record["estimated_output_survival"][relation + ":" + field];
            rates["0"] = source.population ? 1. : 0.;
            for (auto depth = summary->at("sketches").begin();
                 depth != summary->at("sketches").end(); ++depth)
              rates[depth.key()] =
                  depth.value().at("entries").empty() ? 0. : 1.;
          }
        }
        result["source_statistics"][std::to_string(core.node->node_id)] =
            std::move(record);
        continue;
      }
      const auto oid = *core.node->operator_id;
      const auto &op = impl->ops.at(oid);
      if (!semantic_operator(op))
        continue;
      const auto &facts = *core.statistics;
      const auto rate = facts.value("selectivity", Json(nullptr));
      const auto strength = facts.value("effective_samples", 0.);
      if ((!finite_number(rate) && !impl->ensure_prefix_depth_rates) ||
          (strengths.count(oid) && strengths.at(oid) >= strength))
        continue;
      strengths[oid] = strength;
      Json record = facts;
      record["selectivity_scope"] = "operator";
      record["sample_model_id"] = facts.value("sample_model", Json(nullptr));
      if (impl->settings.value("selectivity_parallel_models", true) &&
          impl->models(op).size() > 1) {
        record["sample_model_id"] = nullptr;
        record["sample_model_ids"] = impl->models(op);
        record["selection_model_assignment"] =
            "completion_driven_idle_model_lanes";
      }
      record["sample_layout"] = op.at("default_layout");
      const auto &action = impl->action_for(state_entry.second.state, oid);
      if (action.value("layout", Json(nullptr)) == op.at("default_layout")) {
        if (observation->summary)
          record["summary"] = *observation->summary;
        if (observation->right_summary)
          record["right_summary"] = *observation->right_summary;
      }
      record["input_field_summaries"] = Json::object();
      if (impl->sample_depth_selectivity || impl->ensure_prefix_depth_rates)
        for (const auto &field : input_keys(op)) {
          bool base = false;
          for (const auto &raw : core.node->relations) {
            const auto relation = node_relkey(raw);
            const auto &summaries = impl->relations.at(relation).summaries;
            const auto summary = summaries.find(field);
            if (summary == summaries.end())
              continue;
            record["input_field_summaries"][field] = {
                {"summary", *summary->second},
                {"survival_field", relation + ":" + field},
                {"origin", Json::array({relation, field})}};
            base = true;
            break;
          }
          if (base)
            continue;
          std::string alias = field;
          std::set<std::string> generated_aliases;
          for (const auto &descriptor :
               op.value("input_field_schema", Json::array())) {
            const auto key = descriptor.value(
                "value_key", descriptor.at("name").get<std::string>());
            if (key == field)
              alias = descriptor.at("name").get<std::string>();
            bool base_field = false;
            for (const auto &raw : core.node->relations)
              base_field |=
                  impl->relations.at(node_relkey(raw)).summaries.count(key) !=
                  0;
            if (!base_field)
              generated_aliases.insert(
                  descriptor.at("name").get<std::string>());
          }
          const Json aliases = generated_aliases;
          const auto built = impl->conditional_builds.find(
              std::to_string(core.serial) + canonical(aliases));
          if (built != impl->conditional_builds.end() &&
              built->second->at("summaries").contains(alias)) {
            record["input_field_summaries"][field] = {
                {"summary", built->second->at("summaries").at(alias)},
                {"survival_field", "generated:" + field}};
            continue;
          }
          for (const auto &input : core.inputs) {
            const auto leaf = leaf_for(*input, {field});
            if (!leaf)
              continue;
            const auto &rows = leaf_at(*input, *leaf);
            Json values = Json::array();
            for (std::size_t i = 0; i < row_count(rows); ++i) {
              const auto *value = find_value(rows, i, field);
              if (value)
                values.push_back(serialize_value(*value));
            }
            if (!values.empty()) {
              const auto built =
                  build_summaries(Json{{alias, values}}, impl->sketch_config);
              record["input_field_summaries"][field] = {
                  {"summary", built.at("summaries").at(alias)},
                  {"survival_field", "generated:" + field}};
            }
            break;
          }
          // An independently materialized empty producer domain has no values
          // from which to discover a leaf, but still needs an explicit summary.
          if (!record["input_field_summaries"].contains(field) &&
              core.input_complete && !core.inputs.empty() &&
              std::any_of(
                  core.inputs.begin(), core.inputs.end(),
                  [](const auto &input) { return row_count(*input) == 0; })) {
            const auto built = build_summaries(Json{{alias, Json::array()}},
                                               impl->sketch_config);
            record["input_field_summaries"][field] = {
                {"summary", built.at("summaries").at(alias)},
                {"survival_field", "generated:" + field}};
          }
        }
      if (impl->sample_depth_selectivity || impl->ensure_prefix_depth_rates) {
        record["prefix_selectivity_statistics"] = Json::object();
        const auto kind = op.value("kind", op.value("type", std::string()));
        const bool preserving = prefix_preserving_operator(kind);
        const auto raw =
            record.value("depth_selectivity_statistics", Json::object());
        for (const auto &field : input_keys(op)) {
          const auto summary = record["input_field_summaries"].find(field);
          if (summary == record["input_field_summaries"].end()) {
            if (impl->ensure_prefix_depth_rates)
              throw std::runtime_error(
                  "Missing prefix input summary for operator " +
                  std::to_string(oid) + ", field " + field);
            continue;
          }
          const auto key = summary->at("survival_field").get<std::string>();
          Json scalar = preserving ? Json(1.) : rate;
          bool empty =
              summary->at("summary").value("count", Json(nullptr)) == 0;
          if (summary->contains("origin")) {
            const auto relation = summary->at("origin")[0].get<std::string>();
            empty |= impl->relations.at(relation).population == 0;
            if (join_operator(op, core.node->children.size()) &&
                !record.value("selectivity_floor_applied", false))
              for (std::size_t side = 0; side < core.inputs.size(); ++side)
                if (origins(*core.inputs[side]).count(relation))
                  scalar = record.value(
                      side ? "right_selectivity" : "left_selectivity", scalar);
          }
          auto dense = complete_prefix_selectivity(
              prefix_depth_support(summary->at("summary")),
              raw.value(key, Json::object()), scalar, preserving, empty,
              record.value("selectivity_floor_applied", false));
          if (impl->ensure_prefix_depth_rates)
            for (auto depth = dense.begin(); depth != dense.end(); ++depth)
              if (!finite_number(depth.value().at("rate")))
                throw std::runtime_error(
                    "No finite prefix selectivity anchor for operator " +
                    std::to_string(oid) + ", field " + field + ", depth " +
                    depth.key() + "; sampling budget or deadline exhausted");
          record["prefix_selectivity_statistics"][key] = std::move(dense);
        }
      }
      record["generated_field_identities"] = Json::object();
      for (const auto &descriptor :
           op.value("input_field_schema", Json::array())) {
        const auto field = descriptor.value(
            "value_key", descriptor.value("name", std::string()));
        const auto found = record["input_field_summaries"].find(field);
        if (found == record["input_field_summaries"].end() ||
            found->value("survival_field", std::string())
                    .rfind("generated:", 0) != 0)
          continue;
        const auto identity =
            descriptor.value("sample_expression", Json(nullptr));
        record["generated_field_identities"]["generated:" + field] = canonical(
            Json::array({identity.is_null() || identity == ""
                             ? descriptor.value("binding", Json(field))
                             : identity,
                         descriptor.value("type", Json(nullptr))}));
      }
      result["operator_statistics"][std::to_string(oid)] = std::move(record);
    }
  return result;
}
Json NativeSampler::history() const { return impl->history_; }
}
