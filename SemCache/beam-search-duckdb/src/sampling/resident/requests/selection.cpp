#include "../detail.hpp"
#include "../join/join_candidates.hpp"

namespace semcache::sampling {
Selection NativeSampler::Impl::selection(const Json &op,
                                         const SampleInputs &inputs,
                                         bool semantic, bool full_join_input) {
  std::string model = semantic ? models(op)[0] : std::string();
  std::string key;
  {
    PhaseTimer timer(phase_metrics, "selection_key_serialization_hash",
                     trace_timings, true);
    auto found = execution_signatures.find(&op);
    if (found == execution_signatures.end())
      found = execution_signatures.emplace(&op, digest(op)).first;
    key = found->second + '/' + population_versions.key(inputs) + '/' + model;
    if (full_join_input) {
      const auto stream = op.value(
          "sampling_instance", std::to_string(op.at("operator_id").get<Id>()));
      const auto limit =
          join_output_limits.try_emplace(stream, budget).first->second;
      key += ":output:" + std::to_string(limit);
    }
  }
  if (semantic &&
      op.value("kind", std::string()).find("JOIN") != std::string::npos) {
    std::size_t pairs = row_count(inputs[0]);
    if (inputs.size() == 2) {
      const auto right = row_count(inputs[1]);
      if (right && pairs > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error("Join sample population overflow");
      pairs *= right;
    }
    if (pairs > join_pair_budget ||
        (direct_sampling && operator_sample_sizes.count(op.at("operator_id").get<Id>()))) {
      const auto limit =
          join_probe_limits.try_emplace(key, join_pair_budget).first->second;
      auto it = selection_cache.find(key + ':' + std::to_string(cohort_serial) +
                                     ':' + std::to_string(limit));
      if (it != selection_cache.end())
        return join_batch(op, it->second, inputs.size(), full_join_input);
      if (deadline_reached())
        return {};
      return join_batch(op, sample_join(op, inputs, model, key), inputs.size(),
                        full_join_input);
    }
  }
  if (auto it = selection_cache.find(key); it != selection_cache.end())
    return join_batch(op, it->second, inputs.size(), full_join_input);
  if (semantic)
    if (auto filtered = filter_selection(op, inputs)) {
      selection_cache[key] = *filtered;
      return *filtered;
    }
  // Deadline expiry stops new SQL/model work, not lookup-only recovery of
  // answers already paid for by another candidate with this input boundary.
  if (deadline_reached())
    return {};
  if (!semantic) {
    std::string kind = op.value("kind", op.value("type", std::string()));
    std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
    Json result;
    if (op.contains("sampling_passthrough_input")) {
      const auto side = op.at("sampling_passthrough_input").get<std::size_t>();
      if (side >= inputs.size())
        throw std::invalid_argument("Invalid materialized CTE output boundary");
      selection_cache[key] = Selection{
          inputs.owner(side) ? inputs.owner(side)
                             : std::make_shared<const Json>(inputs[side]),
          {},
          {},
          Json::object(),
          {},
          {},
          0};
      return selection_cache.at(key);
    } else if (direct_sampling && inputs.size() == 2) {
      auto candidates = resident_detail::direct_join_candidates(
          inputs, relational_sample_rows);
      Json pairs = Json::array(), probes = Json::array();
      for (auto &candidate : candidates.probes) {
        sample_factor(candidate.unit[0],
                      sample_factor(candidate.unit[0]) * candidate.expansion);
        probes.push_back(combine(candidate.unit[0], candidate.unit[1]));
        pairs.push_back(std::move(candidate.unit));
      }
      result = execute(op, inputs, nullptr, &pairs);
      auto &selected = selection_cache[key];
      selected.rows = std::make_shared<const Json>(std::move(result));
      selected.probes = std::make_shared<const Json>(std::move(probes));
      selected.diagnostics = {{"direct_pair_sampling", true},
                              {"probe_census", candidates.population <=
                                                   relational_sample_rows}};
      return join_batch(op, selected, inputs.size(), full_join_input);
    } else if (kind == "CROSS_PRODUCT" && inputs.size() == 2) {
      auto a = columns(inputs[0]), b = columns(inputs[1]);
      std::vector<std::string> common;
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                            std::back_inserter(common));
      auto oa = origins(inputs[0]), ob = origins(inputs[1]);
      bool independent = true;
      for (auto &e : oa)
        independent &= !ob.count(e.first);
      if (common.empty() && independent &&
          independent_leaf_identities(inputs[0]) &&
          independent_leaf_identities(inputs[1])) {
        result = {{"product", Json::array({inputs[0], inputs[1]})}};
        counters["selectivity"] =
            counters["selectivity"].get<std::size_t>() + 1;
        Id oid = op.at("operator_id");
        if (row_count(result) && !product_calibrations.count(oid)) {
          auto left_count =
              std::min(row_count(inputs[0]),
                       std::max(std::size_t(1),
                                static_cast<std::size_t>(std::sqrt(budget))));
          while ((left_count + 1) <= budget / (left_count + 1) &&
                 left_count < row_count(inputs[0]))
            ++left_count;
          Json left = Json::array(), right = Json::array();
          for (std::size_t i = 0; i < left_count; ++i)
            left.push_back(row_at(inputs[0], i));
          for (std::size_t i = 0;
               i < std::min(row_count(inputs[1]),
                            std::max(std::size_t(1), budget / left_count));
               ++i)
            right.push_back(row_at(inputs[1], i));
          execute(op, {&left, &right}, nullptr, nullptr, "product_calibration");
          product_calibrations.insert(oid);
        }
      } else
        result = execute(op, inputs, nullptr);
    } else if (full_join_input) {
      auto bounded = op;
      const auto stream = op.value(
          "sampling_instance", std::to_string(op.at("operator_id").get<Id>()));
      bounded["sampling_output_limit"] = join_output_limits.at(stream);
      bounded["sampling_output_seed"] =
          std::stoull(digest_text(stream).substr(0, 16), nullptr, 16);
      result = execute(bounded, inputs, nullptr);
    } else
      result = execute(op, inputs, nullptr);
    auto owned = std::make_shared<const Json>(std::move(result));
    selection_cache[key] = Selection{owned, {}, {}, Json::object(), {}, {}, 0};
    if (full_join_input)
      selection_cache[key].diagnostics["bounded_join_population"] =
          owned->empty()
              ? std::size_t(0)
              : owned->at(0).at("_sample_population_rows").get<std::size_t>();
    return join_batch(op, selection_cache.at(key), inputs.size(),
                      full_join_input);
  }
  std::optional<std::vector<int>> leaf;
  auto observed =
      groups(op, inputs, model, nullptr, leaf);
  if (!resolve_groups(op, inputs, model, observed))
    return {};
  auto &prior = selection_priors[task_key(op)];
  auto result_for = [&](const std::string &key) -> const RequestResult * {
    return ledger.find(key);
  };
  for (const auto &group : observed) {
    const auto *result = result_for(group.key);
    if (!result)
      throw std::logic_error("Selection group has no published result");
    prior.emplace(group.key, result->success);
  }
  // Factorized unary tasks restore their physical leaf; other tasks restore
  // original rows/pairs. Both use the same decision scatter in source order.
  const auto &first = leaf ? leaf_at(inputs[0], *leaf) : inputs[0];
  const auto n = leaf ? first.size() : row_count(first);
  const auto m = inputs.size() == 2 ? row_count(inputs[1]) : 1;
  auto decisions = std::make_shared<resident_detail::DecisionMask>(n * m);
  std::vector<const RequestResult *> results(n * m);
  for (const auto &group : observed) {
    const auto *result = result_for(group.key);
    for (const auto &member : group.members) {
      const auto position = member[0] * m + member[1];
      results[position] = result;
      decisions->publish(position, result->success);
    }
  }
  Json output = Json::array();
  decisions->each_accepted([&](std::size_t position) {
    std::vector<Json> unit{leaf ? first.at(position)
                                : row_at(inputs[0], position / m)};
    if (inputs.size() == 2)
      unit.push_back(row_at(inputs[1], position % m));
    auto value = restore(op, unit, *results[position]);
    if (!value.is_null())
      output.push_back(std::move(value));
  });
  if (leaf)
    output = replace_leaf(inputs[0], *leaf, output);
  auto owned = std::make_shared<const Json>(std::move(output));
  selection_cache[key] = Selection{owned, {}, {}, Json::object(), {}, {}, 0};
  selection_cache[key].decisions = std::move(decisions);
  return join_batch(op, selection_cache.at(key), inputs.size(),
                    full_join_input);
}

}
