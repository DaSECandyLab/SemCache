#include "../detail.hpp"

namespace semcache::sampling {
namespace {
template <class T> void unique(std::vector<T> &values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}
}

std::shared_ptr<const resident_detail::EvidenceNode>
NativeSampler::Impl::evidence_node(const NodePtr &node, const StateData &data) {
  if (!share_state_statistics)
    return {};
  const auto known = evidence_nodes.find(node->handle_id);
  if (known != evidence_nodes.end())
    return known->second;
  PhaseTimer timer(phase_metrics, "shared_prior_context", trace_timings, true);
  auto context = std::make_shared<resident_detail::EvidenceNode>();
  for (const auto child : node->children) {
    const auto &source = data.observations.at(child)->core->evidence;
    context->conditions.insert(context->conditions.end(),
                               source->conditions.begin(),
                               source->conditions.end());
    context->generators.insert(context->generators.end(),
                               source->generators.begin(),
                               source->generators.end());
    context->barriers.insert(context->barriers.end(), source->barriers.begin(),
                             source->barriers.end());
  }
  unique(context->conditions);
  unique(context->generators);
  unique(context->barriers);
  if (node->operator_id) {
    const auto oid = *node->operator_id;
    const auto &op = ops.at(oid);
    auto kind = op.value("kind", std::string());
    std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
    const bool semantic = semantic_operator(op);
    const bool target =
        semantic && (kind == "SEMANTIC_FILTER" || kind == "SEMANTIC_JOIN");
    auto [task, inserted] = evidence_tasks.try_emplace(oid, 0);
    if (inserted) {
      const auto contract =
          semantic ? digest(Json::array(
                         {task_key(op), models(op)[0],
                          op.value("input_field_schema", Json::array()),
                          op.value("required_columns", Json::array()),
                          op.value("scope", Json::array()),
                          op.value("left_input_fields", Json::array()),
                          op.value("right_input_fields", Json::array()),
                          op.value("operator_specification", Json(nullptr)),
                          op.at("default_layout")}))
          : fixed_relation_filter(op)
              ? digest(Json::array(
                    {"fixed-relation-filter", op.value("scope", Json::array()),
                     op.value("sampling_required_fields", Json::array()),
                     op.value("sampling_row_local_sql", op.value("sql", ""))}))
              : "opaque-relation:" + std::to_string(oid);
      auto [entry, _] = evidence_contracts.try_emplace(
          contract, evidence_contracts.size() + 1);
      task->second = entry->second;
    }
    if (target) {
      resident_detail::EvidenceFamily family;
      family.task = task->second;
      for (const auto child : node->children) {
        const auto &relations =
            data.observations.at(child)->core->node->relations;
        family.sides.emplace_back(relations.begin(), relations.end());
      }
      family.generators = context->generators;
      family.barriers = context->barriers;
      context->prior = &shared_priors[family];
      context->depth_priors = &shared_depth_priors[std::move(family)];
      context->input_conditions = context->conditions;
    }
    const bool join = kind == "JOIN" || kind == "INNER_JOIN" ||
                      kind == "CROSS_PRODUCT" || kind == "SEMANTIC_JOIN";
    if (kind == "FILTER" || kind == "SEMANTIC_FILTER" || join) {
      context->conditions.push_back(task->second);
      unique(context->conditions);
    } else if (kind == "SEMANTIC_PROJECTION") {
      // Binding IDs identify the generated value definition. Sampling always
      // uses this operator's same minimum model/default layout across states.
      context->generators.push_back(static_cast<std::uint64_t>(oid));
      unique(context->generators);
    } else if (kind != "PROJECTION" || node->children.size() != 1 ||
               !std::all_of(node->fields.begin(), node->fields.end(),
                            [&](const auto field) {
                              const auto &input =
                                  data.observations.at(node->children[0])
                                      ->core->node->fields;
                              return std::find(input.begin(), input.end(),
                                               field) != input.end();
                            })) {
      // LIMIT, aggregates and value-changing expressions are not comparable
      // merely because their predicate sets look similar.
      context->barriers.push_back(node->handle_id);
      unique(context->barriers);
    }
    if (kind == "FILTER" && !fixed_relation_filter(op)) {
      context->barriers.push_back(node->handle_id);
      unique(context->barriers);
    }
  }
  evidence_nodes.emplace(node->handle_id, context);
  return context;
}

std::optional<resident_detail::ContextPrior::Estimate>
NativeSampler::Impl::shared_selectivity(const Json &info,
                                        CoreObservation &core) {
  core.shared_prior_subscribed = false;
  if (!core.evidence || !core.evidence->prior)
    return std::nullopt;
  PhaseTimer timer(phase_metrics, "shared_prior", trace_timings, true);
  auto &prior = *core.evidence->prior;
  const auto &conditions = core.evidence->input_conditions;
  const double effective = info.value("effective_samples", 0.);
  const auto observed = info.value("selectivity", Json(nullptr));
  if (info.value("sampling_complete", false) &&
      info.value("sample_input_rows", std::size_t(0)) &&
      finite_number(observed)) {
    const auto before = prior.size();
    if (prior.publish(conditions, observed.get<double>(), effective)) {
      counters["shared_prior_updates"] =
          counters.value("shared_prior_updates", std::size_t(0)) + 1;
      counters["shared_prior_entries"] =
          counters.value("shared_prior_entries", std::size_t(0)) +
          prior.size() - before;
    }
  }
  if (info.value("input_rows", Json(nullptr)) == 0 ||
      info.value("population_census", false) || effective >= input_samples)
    return std::nullopt;
  core.shared_prior_subscribed = true;
  core.shared_prior_version = prior.version();
  counters["shared_prior_queries"] =
      counters.value("shared_prior_queries", std::size_t(0)) + 1;
  counters["shared_prior_coordinate_reads"] =
      counters.value("shared_prior_coordinate_reads", std::size_t(0)) +
      conditions.size();
  return prior.estimate(conditions, effective);
}

void NativeSampler::Impl::publish_depth_selectivity(const Json &info,
                                                    CoreObservation &core) {
  core.depth_prior_subscribed = false;
  if (!core.evidence || !core.evidence->depth_priors)
    return;
  PhaseTimer timer(phase_metrics, "shared_depth_publish", trace_timings, true);
  auto &priors = *core.evidence->depth_priors;
  // Join source survival describes an at-least-one match in a particular
  // opposite-side/probe experiment, not pair acceptance. Do not transfer it
  // to a different exposure merely because task and source field agree.
  const auto &op = ops.at(*core.node->operator_id);
  if (join_operator(op, core.node->children.size())) {
    if (core.join_sample_key.empty()) {
      // No comparable probe identity: a unique donor cannot help another
      // observation, so neither retain its summaries nor subscribe to them.
      core.depth_exposure = "observation:" + std::to_string(core.serial);
      return;
    }
    core.depth_exposure = core.join_sample_key + ":" +
                          std::to_string(core.join_probe_limit) + ":" +
                          std::to_string(core.cohort->serial);
  }
  const auto bank = info.find("depth_selectivity_statistics");
  if (info.value("sampling_complete", false) &&
      info.value("sample_input_rows", std::size_t(0)) && bank != info.end())
    for (auto field = bank->begin(); field != bank->end(); ++field)
      for (auto depth = field->begin(); depth != field->end(); ++depth) {
        const auto &evidence = depth.value();
        const auto rate = evidence.value("rate", Json(nullptr));
        const double effective = evidence.value("effective_samples", 0.);
        if (!finite_number(rate) || effective <= 0 ||
            !evidence.value("observed", std::size_t(0)))
          continue;
        auto &prior =
            priors.fields[{field.key(), depth.key(), core.depth_exposure}];
        if (prior.publish(core.evidence->input_conditions, rate.get<double>(),
                          effective)) {
          if (core.depth_exposure.empty())
            ++priors.version;
          else
            ++priors.exposure_versions[core.depth_exposure];
        }
      }
  // Subscribe even when the scalar has enough evidence: a long depth can be
  // absent or sparse independently. No donor or estimate becomes a sample.
  core.depth_prior_subscribed = !info.value("population_census", false);
  core.depth_prior_version = priors.version_for(core.depth_exposure);
}
}
