#include "../detail.hpp"
#include "../join/join_candidates.hpp"

namespace semcache::sampling {
std::vector<Group>
NativeSampler::Impl::quality_groups(const Json &op, const CoreObservation &core,
                                    const std::string &model,
                                    const Counts &counts) {
  std::optional<std::vector<int>> leaf;
  if (core.probe_units || core.complete)
    return groups(op, SampleInputs(core.inputs), model, &counts, leaf,
                  core.probe_units.get());

  const auto seed = std::stoull(
      digest(
          Json::array({op, populations->fingerprints(SampleInputs(core.inputs)),
                       models(op)[0], op.at("default_layout")}))
          .substr(0, 16),
      nullptr, 16);
  const bool join =
      op.at("kind").get<std::string>().find("JOIN") != std::string::npos;
  if (!join) {
    auto prepared = prepare_inputs(op, core.inputs);
    return groups(op, SampleInputs(prepared), model, &counts, leaf);
  }

  // Use the same label-independent pair sampler as selection, on the SAME
  // admitted inputs. Never enumerate a large Cartesian product or select only
  // positive pairs. This consumes no selection HTTP allowance.
  std::set<std::string> dependencies;
  for (const auto &field :
       op.value("sampling_dependency_fields", Json::array()))
    dependencies.insert(field.get<std::string>());
  auto candidates =
      direct_sampling
          ? resident_detail::direct_join_candidates(SampleInputs(core.inputs),
                                                    join_pair_budget)
          : resident_detail::join_candidates(
                SampleInputs(core.inputs), input_keys(op), join_pair_budget,
                budget > std::numeric_limits<std::size_t>::max() / max_strata
                    ? budget
                    : budget * max_strata,
                seed, dependencies);
  OwnedInputs representatives;
  for (std::size_t side = 0; side < core.inputs.size(); ++side) {
    Json rows = Json::array();
    std::set<std::string> seen;
    for (const auto &candidate : candidates.probes) {
      const auto &row = candidate.unit.at(side);
      if (seen.insert(row.at("id").get<std::string>()).second)
        rows.push_back(row);
    }
    representatives.push_back(std::make_shared<const Json>(std::move(rows)));
  }
  auto prepared = prepare_inputs(op, representatives);
  std::vector<std::map<std::string, const Json *>> values(prepared.size());
  for (std::size_t side = 0; side < prepared.size(); ++side)
    for (const auto &row : *prepared[side])
      values[side].emplace(row.at("id").get<std::string>(), &row.at("values"));
  Json units = Json::array();
  for (auto &candidate : candidates.probes) {
    for (std::size_t side = 0; side < prepared.size(); ++side) {
      auto &row = candidate.unit.at(side);
      row["values"] = *values[side].at(row.at("id").get<std::string>());
    }
    sample_factor(candidate.unit[0],
                  sample_factor(candidate.unit[0]) * candidate.expansion);
    units.push_back(std::move(candidate.unit));
  }
  return groups(op, SampleInputs(core.inputs), model, &counts, leaf, &units);
}
}
