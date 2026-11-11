#include "../detail.hpp"
#include "../../common/accuracy.hpp"
#include "request_memo.hpp"

namespace semcache::sampling {
void NativeSampler::Impl::initialize_accuracy_budget(const Json &operators) {
  if (settings.contains("max_accuracy_model_calls") ||
      settings.contains("max_accuracy_llm_requests"))
    throw std::invalid_argument("Replace the global accuracy HTTP limit with "
                                "max_accuracy_batches_per_operator");
  accuracy_batch_limit = positive("max_accuracy_batches_per_operator", 1);
  for (const auto &op : operators) {
    const auto id = op.at("operator_id").get<Id>();
    if (semantic_operator(op)) {
      accuracy_operator_ids.push_back(id);
      accuracy_batches_used[id] = accuracy_request_used[id] = 0;
      accuracy_sampled_inputs[id] = {};
    }
  }
}

void NativeSampler::Impl::update_quality(Json &estimate, bool match,
                                         double weight,
                                         const std::string &teacher) {
  if (!estimate.is_object())
    estimate = Json::object();
  if (!estimate.contains("total")) {
    estimate["correct"] = 0;
    estimate["total"] = 0;
    estimate["value"] = nullptr;
  }
  auto correct = estimate["correct"].get<std::size_t>() + std::size_t(match),
       total = estimate["total"].get<std::size_t>() + 1;
  double weighted_correct =
             estimate.value("weighted_correct",
                            double(correct - std::size_t(match))) +
             weight * match,
         old_weight = estimate.value("weighted_total", double(total - 1));
  Json squares = estimate.value(
      "weighted_sum_squares",
      old_weight == double(total - 1) ? Json(old_weight) : Json(nullptr));
  estimate["correct"] = correct;
  estimate["total"] = total;
  estimate["weighted_correct"] = weighted_correct;
  estimate["weighted_total"] = old_weight + weight;
  estimate["weighted_sum_squares"] =
      squares.is_null() ? Json(nullptr)
                        : Json(squares.get<double>() + weight * weight);
  estimate["effective_samples"] =
      estimate["weighted_sum_squares"].is_null() ||
              estimate["weighted_sum_squares"] == 0
          ? Json(nullptr)
          : Json((old_weight + weight) * (old_weight + weight) /
                 estimate["weighted_sum_squares"].get<double>());
  estimate["value"] = weighted_correct / (old_weight + weight);
  estimate["basis"] = "stratified weighted LLM judge agreement";
  estimate["judge_model"] = teacher;
}

void NativeSampler::Impl::prepare_accuracy(const StatePtr &state,
                                           const QualitySink &compare) {
  if (deadline_reached())
    return;
  if (!states.count(state->handle_id))
    observe(state);
  auto &data = states.at(state->handle_id);
  Counts counts;
  for (auto &node : state->nodes())
    if (!node->operator_id) {
      auto key = node_relkey(*node->relations.begin());
      for (auto &row : *data.observations.at(node->node_id)->rows)
        ++counts[{key, relations.at(key).row_strata.at(
                           row["id"].get<std::string>())}];
    }
  for (auto &node : state->nodes()) {
    if (!node->operator_id)
      continue;
    const auto &op = execution_op(state, node);
    if (!semantic_operator(op))
      continue;
    const auto oid = *node->operator_id;
    if (accuracy_batches_used.at(oid) >= accuracy_batch_limit)
      continue;
    const auto &core = *data.observations.at(node->node_id)->core;
    // Accuracy compares labels on existing inputs, including negative Join
    // pairs. It does not require selection to have finished producing output.
    if (core.inputs.empty() ||
        std::any_of(core.inputs.begin(), core.inputs.end(),
                    [](const auto &rows) { return !row_count(*rows); })) {
      counters["accuracy_empty_inputs"] =
          counters.value("accuracy_empty_inputs", std::size_t(0)) + 1;
      continue;
    }
    auto supported = models(op);
    std::string teacher = profile_.value("judge_model", supported.back());
    const bool candidate_teacher = std::find(supported.begin(), supported.end(),
                                             teacher) != supported.end();
    bool external_teacher = false;
    if (!candidate_teacher && profile_.contains("teacher_models") &&
        profile_.at("teacher_models").is_object() &&
        profile_.at("teacher_models").contains(teacher)) {
      const auto &kinds =
          profile_.at("teacher_models").at(teacher).at("supported_kinds");
      external_teacher =
          kinds.is_array() &&
          std::find(kinds.begin(), kinds.end(), op.at("kind")) != kinds.end();
    }
    if (!candidate_teacher && !external_teacher)
      throw std::invalid_argument("Judge model must support semantic task");
    // The state's physical core already owns its exact prepared input.
    // Group weights use this state's admitted inputs; no SQL expression replay.
    SampleInputs inputs(data.observations.at(node->node_id)->core->inputs);
    const auto evidence =
        data.observations.at(node->node_id)->core->join_evidence;
    std::vector<Group> observed;
    {
      PhaseTimer preparation(phase_metrics, "quality_group_prepare",
                             trace_timings);
      if (evidence) {
        RequestKeyPlan plan(op, supported.front(), op.at("default_layout"),
                            inputs.size() == 2);
        resident_detail::RequestMemo keys(plan);
        for (const auto &group : evidence->owner->quality(*evidence, counts)) {
          auto key = keys.key(
              [&](const std::string &field, bool right) -> const Json & {
                return evidence->owner->value(group.first, field, right);
              });
          observed.push_back(
              {std::move(key), {}, {{group.first, 0}}, group.second});
        }
      } else
        observed = quality_groups(op, core, supported.front(), counts);
      // Sampling identities use a fixed model/layout so changing a candidate's
      // physical implementation cannot spend another batch on the same input.
      const auto &sampled = accuracy_sampled_inputs.at(oid);
      observed.erase(std::remove_if(observed.begin(), observed.end(),
                                    [&](const auto &group) {
                                      return sampled.count(group.key);
                                    }),
                     observed.end());
      if (!core.complete && !observed.empty())
        counters["accuracy_incomplete_input_observations"] =
            counters.value("accuracy_incomplete_input_observations",
                           std::size_t(0)) +
            1;
      Random rng(std::stoull(
          digest(Json::array({op["kind"], op["task"]})).substr(0, 16), nullptr,
          16));
      rng.shuffle(observed);
    }
    const auto size = std::min(accuracy_size, observed.size());
    if (!size || deadline_reached())
      continue;
    // One batch belongs to the logical operator for this query, across every
    // candidate, model and layout. HTTP packing/splitting does not reset it.
    ++accuracy_batches_used.at(oid);
    for (std::size_t i = 0; i < size; ++i) {
      auto &group = observed[i];
      if (evidence)
        group.unit =
            evidence->owner->unit(group.members[0][0], evidence.get(), true);
      accuracy_sampled_inputs.at(oid).insert(group.key);
      for (const auto &layout : accuracy_layouts(op))
        compare(op, group, supported, teacher, layout);
    }
  }
}

void NativeSampler::Impl::finish_accuracy(const StatePtr &state) {
  auto &data = states.at(state->handle_id);
  cache_edges(data);
  for (auto &node : state->nodes())
    if (node->operator_id) {
      const auto &op = ops.at(*node->operator_id);
      if (!semantic_operator(op))
        continue;
      const auto &action = action_for(state, *node->operator_id);
      auto model = action["model_id"].get<std::string>();
      const auto &global =
          accuracy_evidence(profile_statistic("accuracy", model, op["kind"]),
                            action.at("layout"));
      if (!global.is_object())
        continue;
      const auto *evidence = &accuracy_evidence(
          profile_statistic("task_accuracy", model, task_key(op), &global),
          action.at("layout"));
      auto eff = evidence->value("effective_samples", Json(nullptr));
      if (finite_number(eff) && eff.get<double>())
        data.stats.at(node->node_id).write()["reliability"]["accuracy"] =
            eff.get<double>() / (eff.get<double>() + 1.);
    }
}
}
