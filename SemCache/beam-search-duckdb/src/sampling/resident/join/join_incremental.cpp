#include "../detail.hpp"

namespace semcache::sampling {
namespace {
void source_relations(const Json &rows, std::set<std::string> &out) {
  if (product(rows)) {
    for (const auto &part : rows.at("product"))
      source_relations(part, out);
  } else {
    for (const auto &row : rows)
      for (const auto &origin : row.at("lineage"))
        out.insert(relkey(origin[0]));
  }
}
}
std::optional<Selection> NativeSampler::Impl::incremental_join(
    const Json &op, const SampleInputs &inputs, const std::string &model,
    const std::string &key, std::uint64_t seed) {
  const auto cursor_key = key + ':' + std::to_string(cohort_serial);
  auto found = join_probe_cursors.find(cursor_key);
  auto limit = join_probe_limits.at(key);
  const auto requested_limit = limit;
  if (found == join_probe_cursors.end()) {
    if (join_probe_fallbacks.count(cursor_key))
      return std::nullopt;
    auto fields = input_keys(op);
    const auto origin_limit =
        budget > std::numeric_limits<std::size_t>::max() / max_strata
            ? budget
            : budget * max_strata;
    bool supported = !inputs.empty() && inputs.size() <= 2 && !fields.empty();
    std::set<std::string> ancestry;
    for (std::size_t side = 0; supported && side < inputs.size(); ++side) {
      supported &= bool(inputs.owner(side));
      std::set<std::string> source_names;
      source_relations(inputs[side], source_names);
      for (const auto &name : source_names)
        supported &= ancestry.insert(name).second;
    }
    for (const auto &field : fields) {
      supported &= field.rfind("__semantic_input:", 0) != 0;
      bool base = false;
      for (const auto &origin : ancestry)
        base |= relations.at(origin).summaries.count(field) != 0;
      supported &= base;
    }
    if (!supported) {
      join_probe_fallbacks.insert(cursor_key);
      return std::nullopt;
    }
    std::set<std::string> dependencies;
    for (const auto &field :
         op.value("sampling_dependency_fields", Json::array()))
      dependencies.insert(field.get<std::string>());
    resident_detail::QuotaBank quotas;
    for (const auto &entry : relations)
      for (std::size_t i = 0; i < entry.second.strata.size(); ++i)
        quotas[{entry.first, i}] = {entry.second.strata[i].population,
                                    entry.second.strata[i].selected};
    OwnedInputs owned;
    for (std::size_t side = 0; side < inputs.size(); ++side)
      owned.push_back(inputs.owner(side));
    auto cursor = std::make_shared<resident_detail::JoinProbeEvidence>(
        std::move(owned), fields, dependencies, origin_limit, origin_limit,
        seed, std::move(quotas),
        [this](const Json &origin) {
          const auto relation = relkey(origin[0]);
          return resident_detail::QuotaKey{relation,
                                           relations.at(relation).row_strata.at(
                                               origin[1].get<std::string>())};
        },
        max_strata);
    cursor->configure_request(
        RequestKeyPlan(op, model, op.at("default_layout"), inputs.size() == 2));
    found = join_probe_cursors.emplace(cursor_key, std::move(cursor)).first;
  }
  auto &cursor = *found->second;
  const auto before = cursor.size();
  std::vector<std::size_t> added;
  {
    PhaseTimer timer(phase_metrics, "join_coordinate_extension", trace_timings,
                     true);
    added = cursor.prepare(limit);
  }
  if (deadline_reached())
    return Selection{};
  std::vector<Group> missing;
  std::set<std::string> seen;
  std::size_t reused = 0;
  for (const auto i : added) {
    const auto &request = cursor.key(i);
    if (!seen.insert(request).second)
      continue;
    if (ledger.find(request)) {
      ++reused;
      continue;
    }
    missing.push_back({request, cursor.unit(i), {}, 0.});
  }
  double input_mass = 1.;
  bool prepared = false;
  auto prepare = [&] {
    PhaseTimer timer(phase_metrics, "join_evidence_preparation", trace_timings,
                     true);
    input_mass = 1.;
    for (const auto &input : inputs)
      input_mass *= mass(input);
    cursor.prepare_snapshot(std::min(limit, cursor.group_population()),
                            input_mass);
    prepared = true;
  };
  // Only immutable source/weight evidence is prepared by the owner while the
  // backend worker owns its connection. Label publication keeps its original
  // order after the transport is drained and actual requests are charged.
  const bool overlap =
      settings.value("parallelism", std::size_t(1)) > 1 || cursor.size() >= 64;
  if (!resolve_groups(op, inputs, model, missing,
                      overlap ? std::function<void()>(prepare)
                              : std::function<void()>{})) {
    // Retry an interrupted experiment while allowance remains. At the query
    // HTTP limit, retain its resolved prefix as direct evidence.
    if (ledger.remaining() != 0)
      return Selection{};
    // The HTTP allowance can end before all prepared pairs finish.
    // Publish only a genuinely resolved prefix, never
    // turn its unlabelled suffix into negative matches.
    std::size_t paid = 0;
    while (paid < std::min(limit, cursor.group_population()) &&
           ledger.find(cursor.key(paid)))
      ++paid;
    if (!paid)
      return Selection{};
    limit = paid;
    cursor.prepare(limit);
    prepared = false;
  }
  if (!prepared)
    prepare();
  auto &prior = selection_priors[task_key(op)];
  cursor.complete([&](const std::string &request) {
    const auto *result = ledger.find(request);
    if (!result)
      throw std::logic_error("Join probe has no published result");
    const auto success = result->success;
    prior.emplace(request, success);
    return success;
  });
  std::shared_ptr<const resident_detail::JoinProbeSnapshot> evidence;
  {
    PhaseTimer timer(phase_metrics, "join_evidence_publication", trace_timings,
                     true);
    PhaseTimer sources(phase_metrics, "source_inclusions", trace_timings);
    evidence =
        cursor.snapshot(std::min(limit, cursor.group_population()), input_mass);
  }
  Json input_rows = Json::array(), admitted = Json::object();
  for (const auto &input : inputs)
    input_rows.push_back(row_count(input));
  for (const auto &entry : relations)
    admitted[entry.first] = entry.second.admitted.size();
  Json diagnostics = {
      {"sample_pair_population_rows", cursor.population()},
      {"probe_census", evidence->census},
      {"join_pair_budget", join_pair_budget},
      {"join_probe_limit", limit},
      {"join_probe_requested_limit", requested_limit},
      {"join_probe_truncated", limit < requested_limit},
      {"join_group_pair_count", cursor.group_population()},
      {"join_sampling_basis", "value-stratified balanced group-pair sample; "
                              "self-normalized inverse-inclusion weights"},
      {"join_probe_progress",
       {{"new_coordinates", cursor.size() - before},
        {"retained_coordinates", std::min(before, evidence->prefix)}}},
      {"join_sampling_progress",
       {{"input_rows", std::move(input_rows)},
        {"base_admitted_rows", std::move(admitted)},
        {"probe_count", evidence->prefix},
        {"new_parameter_groups", missing.size()},
        {"reused_parameter_groups", reused}}}};
  Selection result;
  result.rows = std::make_shared<const Json>(Json::array());
  result.join_evidence = std::move(evidence);
  result.diagnostics = std::move(diagnostics);
  result.join_sample_key = key;
  result.join_probe_limit = limit;
  const auto cache_key = cursor_key + ':' + std::to_string(limit);
  selection_cache[cache_key] = result;
  auto &recent = join_selection_keys[key];
  recent.push_back(cache_key);
  while (recent.size() > 2) {
    selection_cache.erase(recent.front());
    recent.pop_front();
  }
  return result;
}
}
