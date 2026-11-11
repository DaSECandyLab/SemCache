#include "../detail.hpp"
#include "join_candidates.hpp"
#include "origin_multiplicities.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::join_diversity(const Json &op, const Json &rows) {
  Json result = Json::object();
  for (const auto &field :
       op.value("sampling_dependency_fields", Json::array())) {
    const auto name = field.get<std::string>();
    auto [it, inserted] = diversity_limits.emplace(name, batch_size);
    if (inserted) {
      std::set<std::string> available;
      bool base_field = false;
      for (const auto &entry : relations)
        if (entry.second.summaries.count(name)) {
          base_field = true;
          for (const auto &stratum : entry.second.strata)
            for (const auto &row : stratum.rows)
              available.insert(canonical(row.at("values").at(name)));
        }
      if (base_field)
        it->second = std::min(batch_size, available.size());
    }
    std::set<std::string> distinct;
    for (const auto &row : rows) {
      auto value = row.at("values").find(name);
      if (value != row.at("values").end())
        distinct.insert(canonical(*value));
    }
    result[name] = {{"distinct", distinct.size()}, {"target", it->second}};
  }
  return result;
}

Selection NativeSampler::Impl::sample_join(const Json &op,
                                           const SampleInputs &inputs,
                                           const std::string &model,
                                           const std::string &key) {
  PhaseTimer timer(phase_metrics, "join_candidate_sampling", trace_timings);
  const auto oid = op.at("operator_id").get<Id>();
  const auto limit = direct_sampling && operator_sample_sizes.count(oid)
                         ? operator_sample_sizes.at(oid)
                         : join_probe_limits.at(key);
  if (direct_sampling)
    join_probe_limits[key] = limit;
  auto fields = input_keys(op);
  std::set<std::string> dependencies;
  for (const auto &name : op.value("sampling_dependency_fields", Json::array()))
    dependencies.insert(name.get<std::string>());
  auto seed = join_probe_seeds.find(key);
  if (seed == join_probe_seeds.end()) {
    const auto content_key =
        digest(Json::array({op, populations->fingerprints(inputs), model,
                            op.value("default_layout", Json(nullptr))}));
    seed =
        join_probe_seeds
            .emplace(key, std::stoull(content_key.substr(0, 16), nullptr, 16))
            .first;
  }
  if (!direct_sampling &&
      !(settings.value("selectivity_parallel_models", true) && models(op).size() > 1))
    if (auto result = incremental_join(op, inputs, model, key, seed->second))
      return *result;
  auto selected =
      direct_sampling
          ? resident_detail::direct_join_candidates(inputs, limit)
          : resident_detail::join_candidates(
                inputs, fields, limit,
                budget > std::numeric_limits<std::size_t>::max() / max_strata
                    ? budget
                    : budget * max_strata,
                seed->second, dependencies);
  if (deadline_reached())
    return {};
  bool census = selected.probes.size() == selected.population;
  for (const auto &rows : inputs)
    census &= !sampled_population(rows);

  Json units = Json::array();
  OwnedInputs prepared;
  units.get_ref<Json::array_t &>().reserve(selected.probes.size());
  {
    PhaseTimer timer(phase_metrics, "join_probe_metadata", trace_timings, true);
    std::vector<std::map<std::string, double>> multiplicities(inputs.size());
    std::vector<bool> origins_known(inputs.size());
    const auto source_limit = std::max(budget, limit);
    const auto origin_limit =
        source_limit <= std::numeric_limits<std::size_t>::max() / max_strata
            ? source_limit * max_strata
            : std::numeric_limits<std::size_t>::max();
    for (std::size_t side = 0; !census && side < inputs.size(); ++side)
      origins_known[side] = resident_detail::origin_multiplicities(
          inputs[side], 1., origin_limit, multiplicities[side]);
    Sum probe_mass;
    std::vector<double> weights(selected.probes.size());
    const Json empty_probabilities = Json::object();
    cpu_workers->run(
        selected.probes.size(),
        [&](std::size_t begin, std::size_t end) {
          for (std::size_t probe = begin; probe < end; ++probe) {
            auto &candidate = selected.probes[probe];
            for (std::size_t side = 0; !census && side < candidate.unit.size();
                 ++side) {
              auto &row = candidate.unit[side];
              const auto &previous =
                  row.contains("_sample_origin_probabilities")
                      ? row.at("_sample_origin_probabilities")
                      : empty_probabilities;
              Json probabilities = Json::object();
              bool approximate = row.value("_sample_origin_approximate", false);
              for (const auto &origin : row.at("lineage")) {
                const auto origin_key = canonical(origin);
                if (previous.contains(origin_key) &&
                    previous.at(origin_key).is_null())
                  probabilities[origin_key] = nullptr;
                else if (row.contains("_sample_weight_factor") &&
                         !previous.contains(origin_key))
                  probabilities[origin_key] = nullptr;
                else if (!origins_known[side])
                  probabilities[origin_key] = nullptr;
                else {
                  const auto copies = multiplicities[side].at(origin_key);
                  auto probability = candidate.row_inclusions[side];
                  if (copies > 1) {
                    // Multiple candidate rows can carry one ancestor. A union
                    // approximation prevents counting each copy as a new
                    // source; dependencies between their inclusion events
                    // remain uncertain.
                    probability =
                        probability == 1.
                            ? 1.
                            : -std::expm1(copies * std::log1p(-probability));
                    approximate = true;
                  }
                  probabilities[origin_key] =
                      previous.value(origin_key, 1.) * probability;
                }
              }
              // ordered_json inserts can relocate ALL member values. Publish
              // after visiting lineage instead of invalidating a held
              // lineage/map reference.
              row["_sample_origin_probabilities"] = std::move(probabilities);
              if (approximate)
                row["_sample_origin_approximate"] = true;
            }
            if (!census) {
              sample_factor(candidate.unit[0],
                            sample_factor(candidate.unit[0]) *
                                candidate.expansion);
              double w = raw_weight(candidate.unit[0], nullptr);
              if (candidate.unit.size() == 2)
                w *= raw_weight(candidate.unit[1], nullptr);
              weights[probe] = w;
            }
          }
        },
        RangeWork::JoinMetadata);
    for (std::size_t i = 0; i < selected.probes.size(); ++i) {
      if (!census)
        probe_mass.add(weights[i]);
      units.push_back(std::move(selected.probes[i].unit));
    }
    OwnedInputs representatives;
    for (std::size_t side = 0; side < inputs.size(); ++side) {
      Json rows = Json::array();
      std::set<std::string> seen_ids;
      for (const auto &unit : units)
        if (seen_ids.insert(unit[side].at("id").get<std::string>()).second)
          rows.push_back(unit[side]);
      representatives.push_back(std::make_shared<const Json>(std::move(rows)));
    }
    prepared = prepare_inputs(op, representatives);
    if (deadline_reached())
      return {};
    for (std::size_t side = 0; side < prepared.size(); ++side) {
      // Unchanged expressions return the same owned population. No value
      // assignment is needed for any repeated pair that borrows those rows.
      if (prepared[side] == representatives[side])
        continue;
      std::map<std::string, const Json *> by_id;
      for (const auto &row : *prepared[side])
        by_id.emplace(row.at("id").get<std::string>(), &row);
      for (auto &unit : units)
        unit[side]["values"] =
            by_id.at(unit[side].at("id").get<std::string>())->at("values");
    }
    if (!census) {
      double input_mass = 1;
      for (const auto &rows : inputs)
        input_mass *= mass(rows);
      // Self-normalized inverse-inclusion weights keep the finite-sample ratio
      // in [0,1], including unequal base strata. They represent the entire
      // input mass; only the actually evaluated probes contribute positives or
      // negatives.
      const auto normalization = input_mass / probe_mass.get();
      for (auto &unit : units)
        sample_factor(unit[0], sample_factor(unit[0]) * normalization);
    }
  }

  // Reorder a FIXED sample; label-independent inclusion probabilities stay
  // unchanged. New downstream field values reach early prompt batches first.
  Json ordered = Json::array(), deferred = Json::array();
  std::vector<std::size_t> coordinate_order, deferred_coordinates;
  coordinate_order.reserve(units.size());
  std::map<std::string, std::set<std::string>> seen;
  for (std::size_t position = 0; position < units.size(); ++position) {
    auto &unit = units[position];
    bool novel = seen.empty();
    std::vector<std::pair<std::string, std::string>> values;
    for (const auto &field :
         op.value("sampling_dependency_fields", Json::array())) {
      const auto name = field.get<std::string>();
      for (const auto &row : unit) {
        auto value = row.at("values").find(name);
        if (value != row.at("values").end()) {
          auto signature = canonical(*value);
          novel |= !seen[name].count(signature);
          values.emplace_back(name, std::move(signature));
          break;
        }
      }
    }
    if (novel) {
      for (auto &value : values)
        seen[value.first].insert(std::move(value.second));
      ordered.push_back(std::move(unit));
      coordinate_order.push_back(position);
    } else {
      deferred.push_back(std::move(unit));
      deferred_coordinates.push_back(position);
    }
  }
  for (auto &unit : deferred)
    ordered.push_back(std::move(unit));
  coordinate_order.insert(coordinate_order.end(), deferred_coordinates.begin(),
                          deferred_coordinates.end());
  auto owned_units = std::make_shared<const Json>(std::move(ordered));
  std::optional<std::vector<int>> leaf;
  auto observed = groups(op, inputs, model, nullptr, leaf, owned_units.get());
  std::size_t reused = 0, missing = 0;
  for (const auto &group : observed)
    (ledger.find(group.key) ? reused : missing) += 1;
  std::vector<Json> probe_rows(owned_units->size());
  bool probes_prepared = false;
  auto prepare_probes = [&] {
    PhaseTimer timer(phase_metrics, "join_probe_preparation", trace_timings,
                     true);
    cpu_workers->run(
        owned_units->size(),
        [&](std::size_t begin, std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            const auto &unit = owned_units->at(i);
            if (unit.size() == 2)
              probe_rows[i] = combine(unit[0], unit[1], false);
            else {
              Json probe = {{"id", unit[0].at("id")},
                            {"values", Json::object()},
                            {"lineage", unit[0].at("lineage")}};
              copy_sample_metadata(probe, unit[0]);
              probe_rows[i] = std::move(probe);
            }
          }
        },
        RangeWork::JoinProbes);
    probes_prepared = true;
  };
  const bool overlap = settings.value("parallelism", std::size_t(1)) > 1 ||
                       owned_units->size() >= 64;
  if (!resolve_groups(op, inputs, model, observed,
                      overlap ? std::function<void()>(prepare_probes)
                              : std::function<void()>{}))
    return {};
  if (!probes_prepared)
    prepare_probes();

  Json probes = Json::array(), output = Json::array();
  probes.get_ref<Json::array_t &>().reserve(owned_units->size());
  output.get_ref<Json::array_t &>().reserve(owned_units->size());
  std::vector<const RequestResult *> results(owned_units->size());
  for (const auto &group : observed)
    for (const auto &member : group.members)
      results[member[0]] = ledger.find(group.key);
  {
    PhaseTimer timer(phase_metrics, "join_restore_rows", trace_timings, true);
    std::vector<Json> output_rows(owned_units->size());
    cpu_workers->run(
        owned_units->size(),
        [&](std::size_t begin, std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            const auto &unit = owned_units->at(i);
            auto probe = std::move(probe_rows[i]);
            if (results.at(i)->success) {
              auto row = probe;
              if (selected.compact_values) {
                const auto &indices =
                    selected.probes[coordinate_order[i]].indices;
                auto restored = resident_detail::row_at(inputs[0], indices[0]);
                row["values"] = std::move(restored["values"]);
                if (unit.size() == 2) {
                  auto right = resident_detail::row_at(inputs[1], indices[1]);
                  row["values"].update(right.at("values"));
                }
              } else {
                row["values"] = unit[0].at("values");
                if (unit.size() == 2)
                  row["values"].update(unit[1].at("values"));
              }
              row = resident_detail::restore_row(op, std::move(row),
                                                 *results.at(i));
              if (!census)
                row["_sample_match_incomplete"] = true;
              output_rows[i] = std::move(row);
            }
            probe_rows[i] = std::move(probe);
          }
        },
        RangeWork::JoinRestore);
    for (std::size_t i = 0; i < owned_units->size(); ++i) {
      probes.push_back(std::move(probe_rows[i]));
      if (!output_rows[i].is_null())
        output.push_back(std::move(output_rows[i]));
    }
  }
  Json diagnostics = {{"sample_pair_population_rows", selected.population},
                      {"probe_census", census},
                      {"join_pair_budget", join_pair_budget},
                      {"join_probe_limit", limit},
                      {"join_group_pair_count", selected.group_population},
                      {"join_sampling_basis",
                       "value-stratified balanced group-pair sample; "
                       "self-normalized inverse-inclusion weights"},
                      {"sampling_diversity", join_diversity(op, output)}};
  Json input_rows = Json::array(), admitted = Json::object();
  for (const auto &rows : inputs)
    input_rows.push_back(row_count(rows));
  for (const auto &entry : relations)
    admitted[entry.first] = entry.second.admitted.size();
  diagnostics["join_sampling_progress"] = {
      {"input_rows", std::move(input_rows)},
      {"base_admitted_rows", std::move(admitted)},
      {"probe_count", owned_units->size()},
      {"new_parameter_groups", missing},
      {"reused_parameter_groups", reused}};
  Selection result{std::make_shared<const Json>(std::move(output)),
                   std::make_shared<const Json>(std::move(probes)),
                   std::move(owned_units),
                   std::move(diagnostics),
                   std::move(prepared),
                   key,
                   limit};
  auto decisions =
      std::make_shared<resident_detail::DecisionMask>(results.size());
  for (std::size_t i = 0; i < results.size(); ++i)
    if (results[i])
      decisions->publish(i, results[i]->success);
  result.decisions = std::move(decisions);
  const auto cache_key =
      key + ':' + std::to_string(cohort_serial) + ':' + std::to_string(limit);
  selection_cache[cache_key] = result;
  auto &recent = join_selection_keys[key];
  recent.push_back(cache_key);
  // Keep the latest result plus the previous successful prefix for rollback.
  // Published cores own their rows/probes independently; ledger evidence is
  // never evicted, so an older cache miss cannot repeat a model request.
  while (recent.size() > 2) {
    selection_cache.erase(recent.front());
    recent.pop_front();
  }
  return result;
}
}
