#include "../detail.hpp"
#include "request_keys.hpp"
#include "request_memo.hpp"

namespace semcache::sampling {
std::vector<std::string> NativeSampler::Impl::models(const Json &op) const {
  std::vector<std::string> result;
  const Json empty_catalog = Json::object();
  const auto &catalog =
      profile_.contains("models") ? profile_.at("models") : empty_catalog;
  const auto fixed_model = settings.value(
      "fixed_model_id", profile_.value("fixed_model_id", std::string()));
  for (auto &raw : op.at("models")) {
    std::string model = raw;
    if (!fixed_model.empty() && model != fixed_model)
      continue;
    // An external teacher is registered for quality comparison only.  It is
    // not a candidate sampling model and therefore has no parameter count in
    // the candidate catalog.
    if (!catalog.contains(model) && profile_.contains("teacher_models") &&
        profile_.at("teacher_models").is_object() &&
        profile_.at("teacher_models").contains(model))
      continue;
    auto info = catalog.value(model, Json::object());
    auto supported =
        info.value("supported_kinds", Json::array({op.at("kind")}));
    if (std::find(supported.begin(), supported.end(), op.at("kind")) ==
        supported.end())
      continue;
    result.push_back(std::move(model));
  }
  if (result.empty())
    throw std::invalid_argument("No model supports sampling task");
  // A fixed teacher needs no model-size comparison, and its parameter count
  // may be unpublished. Require sizes only when ordering multiple candidates.
  if (result.size() == 1)
    return result;
  std::vector<std::pair<double, std::string>> sorted;
  for (const auto &model : result) {
    const auto info = catalog.value(model, Json::object());
    const auto size = info.value("parameters", Json(nullptr));
    if (!finite_number(size) || size.get<double>() <= 0)
      throw std::invalid_argument(
          "Sampling requires model parameter counts for " + model +
          "; catalog entry=" + info.dump());
    sorted.emplace_back(size.get<double>(), model);
  }
  std::sort(sorted.begin(), sorted.end());
  result.clear();
  for (auto &e : sorted)
    result.push_back(e.second);
  return result;
}

std::vector<Group>
NativeSampler::Impl::groups(const Json &op, const SampleInputs &inputs,
                            const std::string &model, const Counts *counts,
                            std::optional<std::vector<int>> &leaf,
                            const Json *units, bool defer_units) const {
  PhaseTimer timer(phase_metrics, "actual_tuple_grouping", trace_timings);
  const bool pooled_selection = !counts &&
      settings.value("selectivity_parallel_models", true) && models(op).size() > 1;
  std::vector<Group> out;
  for (auto &rows : inputs)
    if (!row_count(rows))
      return out;
  if (units) {
    std::map<std::string, std::size_t> index;
    RequestKeyPlan key_plan(op, model, op.at("default_layout"),
                            inputs.size() == 2);
    resident_detail::RequestMemo keys(key_plan);
    for (std::size_t i = 0; i < units->size(); ++i) {
      const auto &unit = units->at(i);
      const Json *right = unit.size() == 2 ? &unit[1] : nullptr;
      auto key =
          keys.key([&](const std::string &name, bool is_right) -> const Json & {
            return (is_right ? *right : unit[0]).at("values").at(name);
          });
      if (pooled_selection)
        key = "selection_pool:" + key;
      auto [it, inserted] = index.emplace(key, out.size());
      if (inserted) {
        out.push_back({key,
                       !counts && (defer_units || ledger.find(key))
                           ? std::vector<Json>{}
                           : unit.get<std::vector<Json>>(),
                       {},
                       0});
      }
      auto &group = out[it->second];
      group.members.push_back({i, 0});
      // Only accuracy consumes group weights, using explicit state quotas.
      // Selection restores labels to every member and computes population
      // ratios from probe/output evidence, never from this accumulator.
      if (counts)
        group.weight +=
            weight(unit[0], counts) * (right ? weight(*right, counts) : 1.);
    }
    return out;
  }
  bool factored = inputs.size() == 1 && product(inputs[0]);
  if (factored)
    leaf = leaf_for(inputs[0], input_keys(op));
  const Json &first = leaf ? leaf_at(inputs[0], *leaf) : inputs[0];
  double multiplier =
      counts && leaf ? mass(inputs[0], counts) / mass(first, counts) : 1;
  std::map<std::string, std::size_t> index;
  std::size_t n = row_count(first),
              right = inputs.size() == 2 ? row_count(inputs[1]) : 1;
  RequestKeyPlan key_plan(op, model, op.at("default_layout"),
                          inputs.size() == 2);
  if (n > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("Semantic candidate population overflow");
  resident_detail::RequestKeys keys(*cpu_workers, phase_metrics, trace_timings,
                                    n * right);
  for (std::size_t i = 0; i < n; ++i) {
    auto left_owned = product(first) ? row_at(first, i) : Json(nullptr);
    const auto &left = product(first) ? left_owned : first.at(i);
    auto representative_owned =
        leaf ? representative(inputs[0], *leaf, left) : Json(nullptr);
    const auto &actual_left = leaf ? representative_owned : left;
    for (std::size_t j = 0; j < right; ++j) {
      auto right_owned = inputs.size() == 2 && product(inputs[1])
                             ? row_at(inputs[1], j)
                             : Json(nullptr);
      const Json *right_row =
          inputs.size() == 2
              ? (product(inputs[1]) ? &right_owned : &inputs[1].at(j))
              : nullptr;
      auto key = keys.at(
          i * right + j,
          [&](std::size_t position) {
            return key_plan.from_values([&](const std::string &name,
                                            bool is_right) -> const Json & {
              const auto &rows = is_right ? inputs[1] : first;
              auto value = find_value(
                  rows, is_right ? position % right : position / right, name);
              if (!value)
                throw std::out_of_range("Missing semantic input field");
              return *value;
            });
          },
          [&](std::size_t) { return key_plan(actual_left, right_row); });
      if (pooled_selection)
        key = "selection_pool:" + key;
      const auto [entry, inserted] = index.try_emplace(key, out.size());
      if (inserted) {
        std::vector<Json> unit;
        if (counts || (!defer_units && !ledger.find(key))) {
          unit.push_back(actual_left);
          if (right_row)
            unit.push_back(*right_row);
        }
        out.push_back({key, std::move(unit), {}, 0});
      }
      auto &group = out[entry->second];
      group.members.push_back({i, j});
      if (counts) {
        double w = multiplier * weight(left, counts);
        if (right_row)
          w *= weight(*right_row, counts);
        group.weight += w;
      }
    }
  }
  return out;
}

}
