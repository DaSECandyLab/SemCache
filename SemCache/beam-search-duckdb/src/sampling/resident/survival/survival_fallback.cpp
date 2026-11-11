#include "../detail.hpp"
#include "../../common/expression_bindings.hpp"
#include "../../common/prefix_selectivity.hpp"
#include "estimate_fields.hpp"
#include "../evidence/depth_fusion.hpp"

namespace semcache::sampling {
std::shared_ptr<const CoreObservation> NativeSampler::Impl::refresh_core(
    const NodePtr &node, const Json &op, const StateData &data,
    const CoreObservation &core, std::uint64_t estimate_serial) {
  PhaseTimer timer(phase_metrics, "survival_estimate_refresh", trace_timings,
                   true);
  auto refreshed = std::make_shared<CoreObservation>(core);
  refreshed->cache_input_views.clear();
  refreshed->estimate_serial = estimate_serial;
  auto info = *core.statistics;
  std::vector<Json> sizes;
  for (auto child : node->children)
    sizes.push_back(data.observations.at(child)->statistics.read().value(
        "output_rows", Json(nullptr)));
  const auto observed_selectivity = info.value("selectivity", Json(nullptr));
  propagate_inplace(op, info, sizes);
  if (info.value("survival_estimated", false))
    info["selectivity"] = observed_selectivity;
  info["input_side_rows"] = sizes;
  estimate_survivals(node, op, data, info, *refreshed);
  refreshed->scalar_statistics = SharedStatistics(slim(info));
  refreshed->statistics = std::make_shared<const Json>(std::move(info));
  return refreshed;
}

bool NativeSampler::Impl::refresh_estimates(StateData &data) {
  bool changed = false;
  for (const auto &node : data.state->nodes()) {
    const auto old = data.observations.at(node->node_id);
    if (!node->operator_id || estimates_current(node, *old->core, data))
      continue;
    changed = true;
    auto observation = std::make_shared<Observation>(*old);
    observation->serial = ++observation_serial;
    const auto &op = execution_op(data.state, node);
    const auto key = core_token(node, data);
    const auto found = cores.find(key);
    const auto canonical =
        found == cores.end() ? nullptr : found->second.lock();
    if (canonical && canonical->serial == old->core->serial &&
        estimates_current(node, *canonical, data)) {
      observation->core = canonical;
      counters["shared_estimate_core_hits"] =
          counters.value("shared_estimate_core_hits", std::size_t(0)) + 1;
    } else {
      observation->core =
          refresh_core(node, op, data, *old->core, observation->serial);
      // Updating a historical estimate must not replace a newer physical
      // Join/probe snapshot that happens to have the same structural key.
      if (!canonical || canonical->serial == old->core->serial)
        cores[key] = observation->core;
    }
    resident_detail::replace_estimate_fields(
        observation->statistics.write(),
        observation->core->scalar_statistics.read());
    if (semantic_operator(op) && sample_depth_selectivity)
      observation->statistics.write().update(cache_inputs(
          op, action_for(data.state, *node->operator_id), old->core->inputs,
          old->core->input_complete, *observation, &old->core->input_boundaries,
          &old->statistics.read()));
    resident_detail::replace_estimate_fields(
        data.stats.at(node->node_id).write(), observation->statistics.read());
    data.observations[node->node_id] = std::move(observation);
  }
  return changed;
}

bool NativeSampler::Impl::estimates_current(const NodePtr &node,
                                            const CoreObservation &core,
                                            const StateData &data) const {
  if (core.input_estimate_versions.size() != node->children.size())
    return false;
  for (std::size_t i = 0; i < node->children.size(); ++i)
    if (core.input_estimate_versions[i] !=
        data.observations.at(node->children[i])->core->estimate_serial)
      return false;
  if (core.shared_prior_subscribed &&
      core.shared_prior_version != core.evidence->prior->version())
    return false;
  if (core.depth_prior_subscribed &&
      core.depth_prior_version !=
          core.evidence->depth_priors->version_for(core.depth_exposure))
    return false;
  const auto prior = selection_priors.find(core.prior_task);
  if (core.prior_samples !=
      (prior == selection_priors.end() ? 0 : prior->second.size()))
    return false;
  return core.condition_pool.empty() ||
         core.condition_version ==
             filter_pools.at(core.condition_pool)->version();
}

void NativeSampler::Impl::estimate_survivals(const NodePtr &node,
                                             const Json &op,
                                             const StateData &data, Json &info,
                                             CoreObservation &core) {
  PhaseTimer timer(phase_metrics, "survival_estimation", trace_timings, true);
  core.shared_prior_subscribed = false;
  core.depth_prior_subscribed = false;
  core.prior_task.clear();
  core.prior_samples = 0;
  core.condition_pool.clear();
  core.condition_version = 0;
  core.input_estimate_versions.clear();
  for (const auto child : node->children)
    core.input_estimate_versions.push_back(
        data.observations.at(child)->core->estimate_serial);
  info["sampling_strategy"] = "independent_base";
  info["selectivity_basis"] = info.value("selectivity_floor_applied", false)
                                  ? "zero-output sample cap floor; not an observed success"
                              : semantic_operator(op) ? "independent base samples"
                                                      : "DuckDB relational estimate";
  if (!sample_depth_selectivity || !semantic_operator(op)) {
    core.estimated_survivals = Json::object();
    for (const auto *key :
         {"depth_survival_statistics", "estimated_output_survival",
          "input_survival_precision"})
      info.erase(key);
    if (!ensure_prefix_depth_rates)
      info.erase("depth_selectivity_statistics");
    return;
  }
  Json incoming = Json::object(), outgoing = Json::object();
  for (const auto &raw : node->relations) {
    const auto relation = node_relkey(raw);
    for (const auto &field : relations.at(relation).summaries) {
      const auto key = relation + ":" + field.first;
      auto rates = input_survival(core, data, relation, field.first);
      incoming[key] = rates;
      const auto local = depth_selectivity(core, info, relation, field.first);
      for (auto depth = rates.begin(); depth != rates.end(); ++depth) {
        const auto factor = local.value(depth.key(), Json(nullptr));
        depth.value() =
            depth.value() == 0 || factor == 0 ? Json(0.)
            : finite_number(depth.value()) && finite_number(factor)
                ? Json(depth.value().get<double>() * factor.get<double>())
                : Json(nullptr);
      }
      outgoing[key] = std::move(rates);
    }
  }
  // Generated values use the same stable value_key as their request schema.
  // Their population starts at the producing projection; subsequent predicate
  // rates are multiplied along the actual input branch.
  auto binds = [&](const CoreObservation &ancestor, const std::string &field) {
    std::set<std::string> dependencies;
    for (const auto &descriptor : op.value("input_field_schema", Json::array()))
      if (descriptor.value("value_key", std::string()) == field)
        dependencies = expression_bindings(
            descriptor.value("sample_expression", std::string()));
    for (const auto id : ancestor.node->fields) {
      const auto name = sample_column(imported.at("fields").at(id));
      if (name == field || dependencies.count(name))
        return true;
    }
    return false;
  };
  auto multiply = [](const Json &a, const Json &b) -> Json {
    return a == 0 || b == 0 ? Json(0.)
           : finite_number(a) && finite_number(b)
               ? Json(a.get<double>() * b.get<double>())
               : Json(nullptr);
  };
  std::function<Json(const CoreObservation &, const std::string &,
                     const std::string &)>
      generated_output;
  generated_output = [&](const CoreObservation &source,
                         const std::string &field,
                         const std::string &depth) -> Json {
    if (!source.node->operator_id)
      return 1.;
    const auto &producer = ops.at(*source.node->operator_id);
    const auto kind = producer.value("kind", std::string());
    if (prefix_population_boundary(kind))
      return source.statistics->value("output_rows", Json(nullptr)) == 0 ? 0.
                                                                        : 1.;
    std::string expression, expression_type;
    for (const auto &descriptor : op.value("input_field_schema", Json::array()))
      if (descriptor.value("value_key", std::string()) == field) {
        expression = descriptor.value("sample_expression", std::string());
        expression_type = descriptor.value("type", std::string());
        break;
      }
    std::string source_field = field;
    if (!expression.empty())
      for (const auto &descriptor :
           producer.value("input_field_schema", Json::array()))
        if (descriptor.value("sample_expression", std::string()) ==
                expression &&
            descriptor.value("type", std::string()) == expression_type) {
          source_field = descriptor.value("value_key", source_field);
          break;
        }
    if (kind == "SEMANTIC_PROJECTION") {
      const auto generated = output_column(producer);
      std::string quoted = "\"";
      for (const auto c : generated) {
        quoted += c;
        if (c == '"')
          quoted += c;
      }
      quoted += '"';
      if (generated == field ||
          (!expression.empty() && expression.find(quoted) != std::string::npos))
        return 1.;
    }
    Json inherited = 1.;
    for (const auto child : source.node->children) {
      const auto &ancestor = *data.observations.at(child)->core;
      if (source.node->children.size() == 1 || binds(ancestor, field))
        inherited =
            multiply(inherited, generated_output(ancestor, field, depth));
    }
    const auto &facts = *source.statistics;
    Json factor = prefix_preserving_operator(kind)
                      ? Json(1.)
                      : facts.value("selectivity", Json(nullptr));
    const auto bank = facts.find("depth_selectivity_statistics");
    if (!facts.value("selectivity_floor_applied", false) &&
        bank != facts.end() && bank->contains("generated:" + source_field)) {
      const auto &rates = bank->at("generated:" + source_field);
      factor = complete_prefix_selectivity(
                   {{depth, true}}, rates, factor,
                   prefix_preserving_operator(kind))
                   .at(depth)
                   .at("rate");
    }
    return inherited == 0 || factor == 0 ? Json(0.)
           : finite_number(inherited) && finite_number(factor)
               ? Json(inherited.get<double>() * factor.get<double>())
               : Json(nullptr);
  };
  for (const auto &field : input_keys(op)) {
    bool base = false;
    for (const auto &raw : node->relations)
      base |= relations.at(node_relkey(raw)).summaries.count(field) != 0;
    if (base)
      continue;
    const auto key = "generated:" + field;
    std::set<std::string> depths{"0"};
    if (core.expression_statistics && core.expression_statistics->contains(key))
      for (auto depth = core.expression_statistics->at(key).begin();
           depth != core.expression_statistics->at(key).end(); ++depth)
        depths.insert(depth.key());
    for (const auto &depth : depths) {
      Json inherited = 1.;
      for (const auto child : node->children) {
        const auto &ancestor = *data.observations.at(child)->core;
        if (node->children.size() == 1 || binds(ancestor, field))
          inherited =
              multiply(inherited, generated_output(ancestor, field, depth));
      }
      incoming[key][depth] = inherited;
      Json factor = op.value("kind", std::string()) == "SEMANTIC_PROJECTION"
                        ? Json(1.)
                        : info.value("selectivity", Json(nullptr));
      if (core.expression_statistics &&
          core.expression_statistics->contains(key))
        factor =
            complete_prefix_selectivity(
                {{depth, true}}, core.expression_statistics->at(key), factor,
                op.value("kind", std::string()) == "SEMANTIC_PROJECTION")
                .at(depth)
                .at("rate");
      outgoing[key][depth] =
          inherited == 0 || factor == 0 ? Json(0.)
          : finite_number(inherited) && finite_number(factor)
              ? Json(inherited.get<double>() * factor.get<double>())
              : Json(nullptr);
    }
  }
  core.estimated_survivals = std::move(incoming);
  info["estimated_output_survival"] = std::move(outgoing);
  info["survival_estimation_basis"] =
      "product of independent predecessor field-depth selectivities";
}
}
