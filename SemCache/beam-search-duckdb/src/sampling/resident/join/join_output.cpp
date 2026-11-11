#include "../detail.hpp"

namespace semcache::sampling {
Selection NativeSampler::Impl::join_batch(const Json &op, Selection selected,
                                          std::size_t input_count,
                                          bool full_join_input) {
  if (!selected || (!full_join_input && !join_operator(op, input_count)))
    return selected;
  PhaseTimer timer(phase_metrics, "join_output_publication", trace_timings,
                   true);
  population_moments.register_population(selected.rows);
  const auto stream_key = op.value(
      "sampling_instance", std::to_string(op.at("operator_id").get<Id>()));
  const auto count = selected.diagnostics.value(
      "bounded_join_population", selected.join_evidence
                                     ? selected.join_evidence->matches.size()
                                     : row_count(*selected.rows));
  // Both semantic and relational Join publications have bounded payloads.
  const bool semantic = semantic_operator(op);
  // A Join publication retains its operator's own allowance, even when the
  // shared base pool is larger for another kind of consumer. Probe decision
  // evidence remains complete while published sample rows stay bounded.
  const auto target =
      semantic
          ? std::min(count, initial_size)
          : join_output_limits.try_emplace(stream_key, relational_sample_rows)
                .first->second;
  if (semantic)
    join_output_limits[stream_key] = target;
  auto &stream = join_streams[stream_key];
  // Statistical mass belongs to this probe population. Old success rows must
  // not occupy its quota and hide newly confirmed matches after resampling.
  // Prior decisions/cursors and old immutable State publications stay alive.
  if (semantic)
    stream = resident_detail::JoinOutputStream();
  const bool bounded_relation =
      selected.diagnostics.contains("bounded_join_population");
  if (!bounded_relation)
    selected.statistical_rows = selected.rows;
  selected.diagnostics["observed_join_matches"] = count;
  selected.diagnostics["observed_join_output_mass"] =
      selected.join_evidence ? selected.join_evidence->output_mass
                             : mass(*selected.rows);
  selected.diagnostics["join_output_limit"] = target;
  selected.diagnostics["join_output_exhausted"] = count < target;
  selected.diagnostics["output_sampled"] = count > target;
  // A census retains its exact representation, including lazy products.
  if (selected.join_evidence) {
    const auto &evidence = *selected.join_evidence;
    std::vector<std::string> ids;
    std::shared_ptr<
        const resident_detail::JoinOutputStream::GeneratedPopulation>
        population;
    {
      PhaseTimer identities(phase_metrics, "join_output_identities",
                            trace_timings, true);
      population = evidence.owner->output_population(selected.join_evidence);
      if (!population) {
        ids.reserve(count);
        for (const auto i : evidence.matches)
          ids.push_back(evidence.owner->identity(i));
      }
    }
    selected.diagnostics["join_output_identity_computations"] =
        evidence.owner->identity_computations();
    auto restore = [&](std::size_t i) {
      return evidence.owner->output(evidence.matches.at(i), evidence);
    };
    const auto seed =
        std::stoull(digest_text(stream_key).substr(0, 16), nullptr, 16);
    selected.rows =
        population
            ? stream.generated_publication(std::move(population), restore,
                                           target, seed, selected.join_evidence)
            : stream.generated_publication(ids, restore, target, seed,
                                           selected.join_evidence);
    selected.statistical_rows.reset();
    selected.diagnostics["output_sampled"] =
        count > target || stream.approximate();
    auto diversity = join_diversity(op, Json::array());
    for (auto field = diversity.begin(); field != diversity.end(); ++field)
      field.value()["distinct"] =
          evidence.owner->distinct(field.key(), evidence.prefix);
    selected.diagnostics["sampling_diversity"] = std::move(diversity);
  } else if (count <= target && stream.empty()) {
    // Remember even a small first census without changing its order or
    // expanding a complete Cartesian representation.
    stream.remember_census(selected.rows);
  } else {
    selected.rows = stream.publication(
        selected.rows, target,
        std::stoull(digest_text(stream_key).substr(0, 16), nullptr, 16));
    selected.diagnostics["output_sampled"] =
        count > target || stream.approximate();
  }
  selected.diagnostics["join_output_exhausted"] =
      bounded_relation ? count <= target : row_count(*selected.rows) < target;
  if (bounded_relation) {
    selected.diagnostics["output_sampled"] =
        count > target || stream.approximate();
    selected.statistical_rows.reset();
  }
  selected.join_stream_key = stream_key;
  selected.join_output_limit = target;
  return selected;
}

}
