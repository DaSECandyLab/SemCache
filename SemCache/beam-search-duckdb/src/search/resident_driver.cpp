#include "resident_driver.hpp"
#include "grouped_pruning.hpp"
#include "join_filter_cost.hpp"
#include "../sampling/profile/environment.hpp"
#include "../sampling/resident/sampler.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace semcache {
namespace {
ResidentPrediction predict_survivors(const std::vector<StatePtr> &states,
                                     const std::vector<std::size_t> &excluded,
                                     const ResidentPredictor &predict) {
  if (excluded.empty())
    return predict(states);
  std::vector<bool> removed(states.size(), false);
  for (auto i : excluded)
    removed.at(i) = true;
  std::vector<StatePtr> survivors;
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < states.size(); ++i)
    if (!removed[i]) {
      survivors.push_back(states[i]);
      indices.push_back(i);
    }
  const auto values = predict(survivors);
  if (values.scores.size() != indices.size() || values.times.size() != indices.size() ||
      values.accuracies.size() != indices.size() || values.time_reasons.size() != indices.size())
    throw std::invalid_argument("Prediction count differs from cost-pruning survivors");
  ResidentPrediction result;
  // Excluded positions have no prediction and are never passed to Pareto.
  result.scores.resize(states.size(), {0., 0.});
  result.times.resize(states.size());
  result.accuracies.resize(states.size());
  result.time_reasons.resize(states.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto index = indices[i];
    result.scores[index] = values.scores[i];
    result.times[index] = values.times[i];
    result.accuracies[index] = values.accuracies[i];
    result.time_reasons[index] = values.time_reasons[i];
  }
  if (values.guard_index)
    result.guard_index = indices.at(*values.guard_index);
  return result;
}
}
LayerRanking rank_layer(const ResidentPrediction &prediction,
                        const std::string &preference,
                        std::optional<double> minimum,
                        const std::string &layer_pruning) {
  if (preference != "fastest" && preference != "accurate")
    throw std::invalid_argument("preference must be fastest or accurate");
  if (layer_pruning != "pareto" && layer_pruning != "accuracy_time")
    throw std::invalid_argument(
        "layer_pruning must be pareto or accuracy_time");
  if (minimum && (!std::isfinite(*minimum) || *minimum < 0 || *minimum > 1))
    throw std::invalid_argument("a_min must be in [0,1]");
  validate_scores(prediction.scores);
  const auto size = prediction.scores.size();
  if (prediction.times.size() != size || prediction.accuracies.size() != size)
    throw std::invalid_argument(
        "Prediction count differs from candidate states");
  LayerRanking ranking{prediction.scores, {}, {}, {}, 0};
  for (std::size_t i = 0; i < size; ++i) {
    const auto &time = prediction.times[i],
               &accuracy = prediction.accuracies[i];
    if (time && (!std::isfinite(*time) || *time < 0))
      throw std::invalid_argument("Invalid candidate time prediction");
    if (accuracy &&
        (!std::isfinite(*accuracy) || *accuracy < 0 || *accuracy > 1))
      throw std::invalid_argument("Candidate accuracy must be in [0,1]");
  }
  if (layer_pruning == "pareto")
    return ranking;

  const auto known_time = [&](std::size_t index) {
    return prediction.times[index].value_or(
        std::numeric_limits<double>::infinity());
  };
  const auto known_accuracy = [&](std::size_t index) {
    return prediction.accuracies[index].value_or(-1.);
  };
  std::vector<std::size_t> all(prediction.scores.size());
  std::iota(all.begin(), all.end(), 0);
  // Accuracy-time mode only orders candidates for intermediate Beam
  // truncation. The final A_min check remains in select_completions(); it is
  // deliberately not applied to partial states.
  ranking.order = std::move(all);
  std::stable_sort(ranking.order.begin(), ranking.order.end(),
                   [&](auto left, auto right) {
                     if (known_accuracy(left) != known_accuracy(right))
                       return known_accuracy(left) > known_accuracy(right);
                     if (known_time(left) != known_time(right))
                       return known_time(left) < known_time(right);
                     return left < right;
                   });
  (void)minimum;
  return ranking;
}
CompletionSelection
select_completions(const Scores &scores,
                   const std::vector<std::optional<double>> &times,
                   const std::vector<std::optional<double>> &accuracies,
                   std::optional<double> minimum) {
  validate_scores(scores);
  if (times.size() != scores.size() || accuracies.size() != scores.size())
    throw std::invalid_argument(
        "Prediction count differs from completed states");
  if (minimum && (!std::isfinite(*minimum) || *minimum < 0 || *minimum > 1))
    throw std::invalid_argument("a_min must be in [0,1]");
  CompletionSelection result;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (times[i] && !std::isfinite(*times[i]))
      throw std::invalid_argument("Nonfinite time prediction");
    if (accuracies[i] && (!std::isfinite(*accuracies[i]) ||
                          *accuracies[i] < 0 || *accuracies[i] > 1))
      throw std::invalid_argument("Accuracy prediction must be in [0,1]");
    const auto status = !minimum                     ? "unconstrained"
                        : !accuracies[i]             ? "unknown"
                        : *accuracies[i] >= *minimum ? "feasible"
                                                     : "infeasible";
    result.statuses.push_back(status);
    if (status == std::string("unconstrained") ||
        status == std::string("feasible"))
      result.selected.push_back(i);
  }
  if (minimum) {
    result.constraint_satisfied = !result.selected.empty();
    if (result.selected.empty() && !scores.empty()) {
      std::size_t best = 0;
      for (std::size_t i = 1; i < scores.size(); ++i) {
        const auto quality = accuracies[i].value_or(-1.);
        const auto best_quality = accuracies[best].value_or(-1.);
        if (quality > best_quality ||
            (quality == best_quality && times[i] &&
             (!times[best] || *times[i] < *times[best])))
          best = i;
      }
      // Preserve constraint_satisfied=false: this is an executable fallback,
      // not evidence that the original target was met.
      result.selected.push_back(best);
    }
  } else {
    // Unknown estimates cannot prove dominance in either direction. Keep
    // these candidates visible instead of interpreting ranking placeholders
    // as measured zero accuracy or time.
    Scores known;
    std::vector<std::size_t> indices;
    std::vector<bool> frontier(scores.size(), true);
    for (std::size_t i = 0; i < scores.size(); ++i)
      if (times[i] && accuracies[i]) {
        indices.push_back(i);
        known.push_back({-*times[i], *accuracies[i]});
        frontier[i] = false;
      }
    const auto fronts = pareto_fronts(known);
    if (!fronts.empty())
      for (auto i : fronts.front())
        frontier[indices[i]] = true;
    result.selected.erase(std::remove_if(result.selected.begin(),
                                         result.selected.end(),
                                         [&](auto i) { return !frontier[i]; }),
                          result.selected.end());
  }
  std::stable_sort(result.selected.begin(), result.selected.end(),
                   [&](auto a, auto b) {
                     if (bool(times[a]) != bool(times[b]))
                       return bool(times[a]);
                     if (times[a].value_or(0) != times[b].value_or(0))
                       return times[a].value_or(0) < times[b].value_or(0);
                     if (scores[a][0] != scores[b][0])
                       return scores[a][0] > scores[b][0];
                     return scores[a][1] > scores[b][1];
                   });
  for (auto index : result.selected)
    if (!result.accurate_index ||
        (accuracies[index] &&
         (!accuracies[*result.accurate_index] ||
          *accuracies[index] > *accuracies[*result.accurate_index])))
      result.accurate_index = index;
  return result;
}
ResidentResult NativeSearchEngine::search(const ResidentOptions &options,
                                          ResidentRandom &random,
                                          const std::vector<StatePtr> &original,
                                          const ResidentPredictor &predict) {
  if (!sampler_)
    throw std::invalid_argument("Resident search requires a native sampler");
  if (!options.beam_width || !options.max_expansions ||
      !std::isfinite(options.max_seconds) || options.max_seconds <= 0 ||
      !std::isfinite(options.exploration_fraction) ||
      options.exploration_fraction < 0 || options.exploration_fraction >= 1)
    throw std::invalid_argument("Invalid native search budgets");
  if (options.preference != "fastest" && options.preference != "accurate")
    throw std::invalid_argument("preference must be fastest or accurate");
  if (options.layer_pruning != "pareto" &&
      options.layer_pruning != "accuracy_time" &&
      options.layer_pruning != "grouped")
    throw std::invalid_argument(
        "layer_pruning must be pareto, accuracy_time or grouped");
  if (options.layer_pruning == "grouped" &&
      (options.training || options.mode != "heuristic_value" ||
       !options.max_structure_groups || !options.max_implementations_per_group))
    throw std::invalid_argument(
        "Grouped pruning requires value-guided inference and positive quotas");
  for (const auto &state : original)
    if (!problem_.complete(state))
      throw std::invalid_argument("Original candidate must be complete");
  const bool cost_pruning_enabled = !options.training &&
      options.mode == "heuristic_value" && options.layer_pruning == "pareto";
  ResidentResult result;
  if (auto online =
          std::dynamic_pointer_cast<sampling::NativeSampler>(sampler_)) {
    const auto before = std::chrono::steady_clock::now();
    sampler_ = online->prepare_statistics(
        problem_, options.action_priorities,
        options.mode == "heuristic_requests" ? options.fixed_model
                                             : std::nullopt,
        options.mode != "heuristic_requests", options.max_seconds);
    result.timings["prepare_statistics"] = {
        std::chrono::duration<double>(std::chrono::steady_clock::now() - before)
            .count(),
        1.};
  }
  const auto start = std::chrono::steady_clock::now();
  auto timed = [&](const std::string &phase, auto &&operation) {
    const auto before = std::chrono::steady_clock::now();
    operation();
    const auto after = std::chrono::steady_clock::now();
    auto &counter = result.timings[phase];
    counter[0] += std::chrono::duration<double>(after - before).count();
    counter[1] += 1;
    std::array<std::int64_t, 2> span{};
    if (options.trace_timings) {
      span = {std::chrono::duration_cast<std::chrono::nanoseconds>(
                  before.time_since_epoch())
                  .count(),
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  after.time_since_epoch())
                  .count()};
      result.timing_spans[phase].push_back(span);
    }
    return span;
  };
  auto record = [&](const StatePtr &state, const std::string &kind,
                    std::size_t layer,
                    const std::array<std::int64_t, 2> &enumerate_span,
                    const std::array<std::int64_t, 2> &observe_span,
                    bool observed = true) {
    if (!options.trace_timings)
      return;
    timed("state_diagnostics", [&] {
      ResidentResult::StateDiagnostic row{
          kind,
          layer,
          state,
          enumerate_span,
          observe_span,
          observed ? sampler_->statistic_blocks(state)
                   : std::map<Id, std::shared_ptr<const sampling::Json>>{},
          {}};
      if (row.observe_spans.empty() && observe_span[1] > observe_span[0])
        row.observe_spans.push_back(observe_span);
      result.state_diagnostics.push_back(std::move(row));
    });
  };
  std::vector<StatePtr> completed;
  const auto observed_initial = problem_.initial();
  const auto initial_span = timed("initial_observation",
                                  [&] { sampler_->observe(observed_initial); });
  record(observed_initial, "initial", 0, {}, initial_span);
  struct PendingObservation {
    StatePtr state;
    std::string kind;
    std::size_t layer;
    std::array<std::int64_t, 2> enumeration;
  };
  std::vector<PendingObservation> pending;
  struct CompletionValue {
    std::array<double, 2> score;
    std::optional<double> time, accuracy;
    std::optional<std::string> reason;
  };
  std::unordered_map<std::uint64_t, CompletionValue> completion_values;
  std::unordered_set<std::string> completion_keys;
  std::vector<StatePtr> pending_completions;
  auto queue_completion = [&](const StatePtr &state) {
    if (completion_keys.insert(problem_.state_key(state)).second)
      pending_completions.push_back(state);
  };
  auto score_completions = [&] {
    if (pending_completions.empty())
      return;
    if (options.mode == "heuristic_requests") {
      // Request search selects from one final evidence snapshot below. Its
      // admission values are never consumed, so do not refresh/export/score
      // the same complete states here only to discard those predictions.
      pending_completions.clear();
      return;
    }
    timed("complete_refresh", [&] { sampler_->refresh(pending_completions); });
    ResidentPrediction prediction;
    timed("complete_predict",
          [&] { prediction = predict(pending_completions); });
    const auto size = pending_completions.size();
    validate_scores(prediction.scores);
    if (prediction.scores.size() != size || prediction.times.size() != size ||
        prediction.accuracies.size() != size ||
        prediction.time_reasons.size() != size)
      throw std::invalid_argument(
          "Prediction count differs from completed states");
    for (std::size_t i = 0; i < size; ++i)
      completion_values.emplace(pending_completions[i]->handle_id,
                                CompletionValue{prediction.scores[i],
                                                prediction.times[i],
                                                prediction.accuracies[i],
                                                prediction.time_reasons[i]});
    pending_completions.clear();
  };
  std::unordered_set<std::uint64_t> queued_original;
  auto flush_observations = [&] {
    if (pending.empty()) {
      score_completions();
      return;
    }
    auto frozen =
        std::dynamic_pointer_cast<sampling::ProfileEnvironment>(sampler_);
    if (frozen && options.planner_workers > 1 && !options.trace_timings) {
      std::vector<StatePtr> states;
      states.reserve(pending.size());
      for (const auto &entry : pending)
        states.push_back(entry.state);
      timed("observe", [&] {
        frozen->observe_batch(states, options.planner_workers);
      });
      result.timings["observe"][1] += pending.size() - 1;
    } else
      for (const auto &entry : pending) {
        const auto observed =
            timed("observe", [&] { sampler_->observe(entry.state); });
        record(entry.state, entry.kind, entry.layer, entry.enumeration, observed);
      }
    pending.clear();
    score_completions();
  };
  for (const auto &state : original) {
    std::vector<StatePtr> path;
    for (auto current = state; current->parent; current = current->parent)
      path.push_back(current);
    std::reverse(path.begin(), path.end());
    for (const auto &prefix : path) {
      if (queued_original.insert(prefix->handle_id).second) {
        pending.push_back({prefix, "original", 0, {}});
      }
    }
    completed.push_back(state);
    queue_completion(state);
  }
  auto initial = problem_.initial();
  if (problem_.complete(initial)) {
    completed.push_back(initial);
    queue_completion(initial);
  }
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  SearchSession session(
      problem_, initial, completed, options.beam_width, options.max_expansions,
      options.max_seconds, options.training, options.exploration_fraction,
      elapsed, options.action_priorities,
      [&](const auto &population, auto count) {
        return random.sample(population, count);
      },
      options.mode, options.fixed_model);
  while (true) {
    SearchRequest request;
    const auto enumeration =
        timed("enumerate_transition", [&] { request = session.advance(); });
    if (request.kind == "observe") {
      if (session.commit_observation(request.child_id)) {
        pending.push_back({request.child, "expanded",
                           session.trace().size() + 1, enumeration});
        if (problem_.complete(request.child))
          queue_completion(request.child);
      } else
        record(request.child, "duplicate", session.trace().size() + 1,
               enumeration, {}, false);
    } else if (request.kind == "score") {
      flush_observations();
      std::vector<StatePtr> states;
      for (auto id : request.ids)
        states.push_back(session.state(id));
      timed("refresh", [&] { sampler_->refresh(states); });
      JoinFilterPruning cost_pruning;
      if (cost_pruning_enabled)
        timed("join_filter_cost", [&] {
          cost_pruning = prune_join_filter_cost(
              problem_, states, join_filter_statistics(*sampler_, states),
              sampler_->time_calibration());
        });
      ResidentPrediction prediction;
      timed("predict", [&] {
        prediction = predict_survivors(states, cost_pruning.excluded, predict);
      });
      timed("rank_select", [&] {
        const auto grouped = options.layer_pruning == "grouped";
        auto ranking =
            !grouped && !options.training && options.mode != "heuristic_requests"
                ? rank_layer(prediction, options.preference,
                             options.minimum_accuracy, options.layer_pruning)
                : LayerRanking{prediction.scores, {}, {}, {}, 0};
        GroupedPruning groups;
        if (cost_pruning_enabled) {
          const auto &pruned = cost_pruning;
          ranking.excluded = pruned.excluded;
          session.set_cost_pruned(pruned.excluded.size());
          auto decisions = join_filter_decisions(pruned);
          std::map<Id, std::size_t> counts;
          for (const auto &decision : pruned.decisions)
            ++counts[decision.filter_operator];
          sampling::Json by_operator = sampling::Json::object();
          for (const auto &[oid, count] : counts)
            by_operator[std::to_string(oid)] = count;
          if (decisions.size() > 8)
            decisions.erase(decisions.begin() + 8, decisions.end());
          result.join_filter_trace.push_back({
              {"depth", session.trace().size() + 1},
              {"cost_pruned", pruned.excluded.size()},
              {"by_filter_operator", by_operator}, {"witnesses", decisions}});
        }
        std::optional<std::size_t> retention_limit;
        if (grouped) {
          groups = prune_structure_groups(
              problem_, states, prediction, options.max_structure_groups,
              options.max_implementations_per_group);
          ranking.order = groups.order;
          retention_limit = groups.order.size();
        }
        const auto keep = session.rank(ranking.scores, 0, ranking.excluded,
                                       prediction.guard_index, ranking.order,
                                       retention_limit);
        if (grouped) {
          std::map<std::size_t, std::size_t> counts;
          for (auto index : keep)
            ++counts[groups.membership[index]];
          ResidentResult::GroupTrace trace{session.trace().size() + 1, groups.groups, {}, {}};
          for (const auto &[group, count] : counts) {
            trace.retained_group_sizes.push_back(count);
            trace.retained_group_means.push_back(groups.mean_predictions[group]);
          }
          result.group_trace.push_back(std::move(trace));
        }
        session.set_quality_threshold(ranking.effective_minimum, ranking.relaxations);
        std::unordered_set<std::size_t> retained(keep.begin(), keep.end());
        std::vector<std::size_t> pool;
        for (std::size_t i = 0; i < states.size(); ++i)
          if (!retained.count(i))
            pool.push_back(i);
        session.select(random.sample(pool, session.random_slots()));
      });
    } else if (request.kind == "quality") {
      session.commit_quality();
      timed("release_pruned", [&] { sampler_->retain(session.live_states()); });
    } else if (request.kind == "done") {
      flush_observations();
      // Include statistical propagation for the terminal states in the cutoff.
      result.exhausted =
          session.exhausted() || std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                         .count() >= options.max_seconds;
      for (auto id : request.ids)
        result.completed.push_back(session.state(id));
      result.heuristic_guard_status = session.heuristic_guard_status();
      if (const auto id = session.heuristic_guard_complete_id()) {
        const auto key = problem_.state_key(session.state(*id));
        for (std::size_t i = 0; i < result.completed.size(); ++i)
          if (problem_.state_key(result.completed[i]) == key) {
            result.heuristic_guard_complete = i;
            break;
          }
      }
      std::vector<StatePtr> prefixes;
      std::unordered_set<std::uint64_t> visited;
      for (const auto &state : result.completed)
        for (auto current = state; current; current = current->parent)
          if (visited.insert(current->handle_id).second)
            prefixes.push_back(current);
      timed("final_refresh", [&] { sampler_->refresh(prefixes); });
      if (options.mode == "heuristic_requests" && !result.completed.empty()) {
        // Score every existing completion against the same latest statistics
        // and profile, including both time and quality. Reuse this snapshot for
        // feasibility, final choice and exported assessments.
        timed("final_predict",
              [&] { result.prediction = predict(result.completed); });
      } else
        for (const auto &state : result.completed) {
          const auto &value = completion_values.at(state->handle_id);
          result.prediction.scores.push_back(value.score);
          result.prediction.times.push_back(value.time);
          result.prediction.accuracies.push_back(value.accuracy);
          result.prediction.time_reasons.push_back(value.reason);
        }
      result.scores = result.prediction.scores;
      validate_scores(result.scores);
      if (result.scores.size() != result.completed.size() ||
          result.prediction.times.size() != result.completed.size() ||
          result.prediction.accuracies.size() != result.completed.size() ||
          result.prediction.time_reasons.size() != result.completed.size())
        throw std::invalid_argument(
            "Prediction count differs from completed states");
      result.trace = session.trace();
      // A constant neutral second axis gives request search a scalar frontier;
      // exported accuracy stays unknown and never affects its eligibility.
      result.selection = select_completions(
          result.scores, result.prediction.times,
          options.mode == "heuristic_requests"
              ? std::vector<std::optional<double>>(result.completed.size(), 0.)
              : result.prediction.accuracies,
          options.mode == "heuristic_requests" ? std::nullopt
                                               : options.minimum_accuracy);
      result.expansions = session.expansions();
      last_expansions_ = result.expansions;
      result.action_budget_exhausted = session.action_budget_exhausted();
      result.timings["beam_search"] = {
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count(),
          1.};
      return result;
    } else
      throw std::logic_error("Unknown resident search phase");
  }
}
}
