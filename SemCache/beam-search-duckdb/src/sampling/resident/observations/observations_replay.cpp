#include "../detail.hpp"
#include "../survival/estimate_fields.hpp"

namespace semcache::sampling {
StateData NativeSampler::Impl::replay(const StatePtr &state) {
  PhaseTimer timer(phase_metrics, "replay", trace_timings);
  row_weights.cohort(cohort_serial);
  population_moments.cohort(cohort_serial);
  if (!survival_cohort || survival_cohort->serial != cohort_serial) {
    PhaseTimer timer(phase_metrics, "survival_cohort", trace_timings, true);
    auto cohort = std::make_shared<resident_detail::AdmissionCohort>();
    cohort->serial = cohort_serial;
    for (const auto &entry : relations) {
      auto &selected = cohort->selected[entry.first];
      for (const auto &stratum : entry.second.strata)
        selected.push_back(stratum.selected);
    }
    survival_cohort = std::move(cohort);
  }
  StateData data;
  data.state = state;
  auto nodes = state->nodes();
  const StateData *previous = nullptr;
  if (auto found = states.find(state->handle_id); found != states.end())
    previous = &found->second;
  auto current = [&](const std::shared_ptr<const CoreObservation> &core) {
    if (!core->complete &&
        (!core->retry || core->retry->requests != ledger.used() ||
         core->retry->expired != deadline_reached()))
      return false;
    return (core->join_sample_key.empty() ||
            join_probe_limits.at(core->join_sample_key) ==
                core->join_probe_limit) &&
           (core->join_stream_key.empty() ||
            join_output_limits.at(core->join_stream_key) ==
                core->join_output_limit);
  };
  if (!previous && state->parent)
    if (auto found = states.find(state->parent->handle_id);
        found != states.end())
      previous = &found->second;
  auto publish = [&](const NodePtr &node,
                     const std::shared_ptr<const Observation> &observation) {
    PhaseTimer timer(phase_metrics, "publish_statistics", trace_timings, true);
    // Cohort evidence replaces intrinsic facts; branch precision/budget
    // annotations survive exactly as in the reference publisher.
    const auto nid = node->node_id;
    auto prior = previous;
    if (prior && prior->stats.count(nid)) {
      data.stats.emplace(nid, prior->stats.at(nid));
      if (prior->observations.at(nid) != observation)
        resident_detail::replace_estimate_fields(
            data.stats.at(nid).write(), observation->statistics.read());
    } else
      data.stats.emplace(nid, observation->statistics);
    data.observations[node->node_id] = observation;
  };
  for (auto &node : nodes) {
    std::string nid = std::to_string(node->node_id);
    auto token = state_token(node, data);
    const auto core_key = core_token(node, data);
    if (auto it = observations.find(token); it != observations.end())
      if (auto existing = it->second.lock()) {
        if (existing->complete && current(existing->core) &&
            estimates_current(node, *existing->core, data)) {
          publish(node, existing);
          continue;
        }
      }
    auto observation = std::make_shared<Observation>();
    observation->serial = ++observation_serial;
    observation->token = token;
    if (auto it = cores.find(core_key); it != cores.end())
      if (auto core = it->second.lock(); core && current(core)) {
        if (!estimates_current(node, *core, data)) {
          core = refresh_core(node, execution_op(state, node), data, *core,
                              observation->serial);
          cores[core_key] = core;
        }
        if (!core->complete) {
          counters["incomplete_core_hits"] =
              counters.value("incomplete_core_hits", std::size_t(0)) + 1;
        }
        observation->core = core;
        observation->rows = core->rows;
        observation->boundary = core->boundary;
        observation->complete = core->complete;
        observation->statistics = core->scalar_statistics;
        if (node->operator_id) {
          const auto &op = execution_op(state, node);
          const auto &action = action_for(state, *node->operator_id);
          if (semantic_operator(op) && sample_depth_selectivity)
            observation->statistics.write().update(
                cache_inputs(op, action, core->inputs, core->input_complete,
                             *observation, &core->input_boundaries));
        }
        publish(node, observation);
        observations[token] = observation;
        continue;
      }
    auto core = std::make_shared<CoreObservation>();
    core->node = node;
    core->cohort = survival_cohort;
    core->serial = observation->serial;
    core->estimate_serial = observation->serial;
    core->evidence = evidence_node(node, data);
    observation->core = core;
    if (!node->operator_id) {
      auto key = node_relkey(*node->relations.begin());
      auto &rel = relations.at(key);
      observation->rows = std::make_shared<const Json>(rel.admitted);
      row_weights.register_population(observation->rows);
      population_moments.register_population(observation->rows);
      observation->boundary =
          std::make_shared<const Json>(boundary(*observation->rows));
      double population = rel.population;
      observation->statistics = SharedStatistics(
          Json{{"output_rows", population},
               {"input_rows", population},
               {"selectivity", 1.},
               {"output_width", (*observation->boundary)[0]},
               {"input_width", (*observation->boundary)[0]},
               {"output_field_statistics", (*observation->boundary)[1]},
               {"effective_samples", effective(*observation->rows)},
               {"cardinality_basis",
                profile_.value("cardinality_only_sources", Json::object())
                        .contains(key)
                    ? "correlated input domain upper bound; no sampled values"
                    : "DuckDB base cardinality; sampled field distribution"}});
    } else {
      replay_semantic_node(state, node, data, nodes, nid, observation, core);
    }
    core->rows = observation->rows;
    core->boundary = observation->boundary;
    core->complete = observation->complete;
    Json intrinsic = std::move(observation->statistics.write());
    observation->statistics = SharedStatistics(slim(intrinsic));
    core->scalar_statistics = observation->statistics;
    intrinsic.erase("conditional_sketch_plan");
    for (auto *key : {"alpha", "right_alpha", "pre", "basis"}) {
      intrinsic.erase(key);
      if (core->scalar_statistics.read().contains(key))
        core->scalar_statistics.write().erase(key);
    }
    core->statistics = std::make_shared<const Json>(std::move(intrinsic));
    observation->core = core;
    if (!core->complete && !selection_remaining())
      core->retry =
          CoreObservation::RetryContext{ledger.used(), deadline_reached()};
    if (core->complete || core->retry)
      cores[core_key] = core;
    publish(node, observation);
    if (observation->complete)
      observations[token] = observation;
  }
  return data;
}

void NativeSampler::Impl::observe(const StatePtr &state) {
  auto updated = replay(state);
  if (state->parent) {
    auto &latest = updated.stats.at(state->added->node_id).write();
    latest["sampling_strategy"] = "independent_base";
  }
  cache_edges(updated);
  states[state->handle_id] = std::move(updated);
}
}
