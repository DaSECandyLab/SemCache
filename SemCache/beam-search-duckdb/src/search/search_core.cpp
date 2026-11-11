#include "search_core.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace semcache {
void validate_scores(const Scores &scores) {
  for (const auto &row : scores)
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument(
            "Ranking heads must return two finite scores");
}
Fronts pareto_fronts(const Scores &scores) {
  validate_scores(scores);
  std::map<std::array<double, 2>, std::vector<std::size_t>,
           std::greater<std::array<double, 2>>>
      points;
  std::set<double, std::greater<double>> ys;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    points[scores[i]].push_back(i);
    ys.insert(scores[i][1]);
  }
  std::map<double, std::size_t> ranks;
  std::size_t rank = 0;
  for (double y : ys)
    ranks[y] = ++rank;
  std::vector<std::size_t> tree(rank + 1, 0);
  Fronts fronts;
  for (const auto &item : points) {
    auto pos = ranks.at(item.first[1]);
    std::size_t depth = 0;
    for (auto p = pos; p; p -= p & -p)
      depth = std::max(depth, tree[p]);
    if (depth == fronts.size())
      fronts.emplace_back();
    fronts[depth].insert(fronts[depth].end(), item.second.begin(),
                         item.second.end());
    for (auto p = pos; p < tree.size(); p += p & -p)
      tree[p] = std::max(tree[p], depth + 1);
  }
  for (auto &front : fronts)
    std::sort(front.begin(), front.end());
  return fronts;
}
std::vector<std::size_t> trim(const Scores &scores, std::size_t limit,
                              const IndexSampler &sample,
                              const Fronts *supplied_fronts) {
  validate_scores(scores);
  auto generated = supplied_fronts ? Fronts{} : pareto_fronts(scores);
  const auto &fronts = supplied_fronts ? *supplied_fronts : generated;
  std::vector<std::size_t> retained;
  if (!limit)
    return retained;
  for (const auto &front : fronts) {
    const auto slots = limit - retained.size();
    if (front.size() <= slots) {
      retained.insert(retained.end(), front.begin(), front.end());
    } else {
      const auto chosen = sample(front, slots);
      retained.insert(retained.end(), chosen.begin(), chosen.end());
      break;
    }
    if (retained.size() == limit)
      break;
  }
  return retained;
}
SearchSession::SearchSession(const StateProblem &problem, StatePtr initial,
                             const std::vector<StatePtr> &completed,
                             std::size_t beam_width, std::size_t max_expansions,
                             double max_seconds, bool training,
                             double exploration_fraction, double elapsed,
                             ActionPriorities action_priorities,
                             IndexSampler sample, std::string mode,
                             std::optional<std::string> fixed_model)
    : problem_(problem), beam_width_(beam_width),
      max_expansions_(max_expansions), max_seconds_(max_seconds),
      exploration_fraction_(exploration_fraction), training_(training),
      start_(std::chrono::steady_clock::now() -
             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                 std::chrono::duration<double>(elapsed))) {
  if (!beam_width || !max_expansions || !std::isfinite(max_seconds) ||
      max_seconds <= 0 || !std::isfinite(exploration_fraction) ||
      exploration_fraction < 0 || exploration_fraction >= 1 ||
      !std::isfinite(elapsed) || elapsed < 0)
    throw std::invalid_argument("Invalid native search budgets");
  adaptive_cap_ = beam_width_;
  if (mode != "heuristic_requests" && mode != "heuristic_value" &&
      mode != "value")
    throw std::invalid_argument("Unknown Beam search policy");
  request_guided_ = mode == "heuristic_requests";
  fixed_model_ = request_guided_ ? std::move(fixed_model) : std::nullopt;
  if (!training_ && (mode == "value" || exploration_fraction_ != 0.))
    throw std::invalid_argument(
        "Pure value guidance and exploration require training");
  action_priorities_ = std::move(action_priorities);
  sample_ = std::move(sample);
  if (!sample_)
    throw std::invalid_argument("A Pareto cutoff sampler is required");
  for (const auto &entry : action_priorities_)
    for (auto value : entry.second)
      if (!std::isfinite(value))
        throw std::invalid_argument("Action priorities must be finite");
  states_[0] = std::move(initial);
  if (problem_.complete(states_[0])) {
    completed_.push_back(0);
    if (heuristic_guard_enabled_)
      guard_complete_id_ = 0;
  } else {
    beam_.push_back(0);
    if (heuristic_guard_enabled_)
      guard_parent_id_ = 0;
  }
  for (const auto &state : completed) {
    if (!problem_.complete(state))
      throw std::invalid_argument(
          "Completed search entry requires a complete state");
    states_[next_id_] = state;
    completed_.push_back(next_id_++);
  }
}
void SearchSession::validate_problem(const StateProblem &problem) const {
  if (!problem_.same_owner(problem))
    throw std::invalid_argument(
        "Search session belongs to another native problem");
}
std::string SearchSession::heuristic_guard_status() const {
  if (!heuristic_guard_enabled_)
    return "disabled";
  if (guard_complete_id_)
    return "complete";
  if (phase_ == "done" || !guard_parent_id_)
    return "incomplete";
  return "active";
}
bool SearchSession::budget_reached() const {
  return expansions_ >= max_expansions_ ||
         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_)
                 .count() >= max_seconds_;
}
std::size_t SearchSession::adaptive_beam_width() {
  if (candidates_.empty() || beam_width_ <= 1 || expansions_ >= max_expansions_)
    return std::max<std::size_t>(1, adaptive_cap_);
  std::size_t remaining_layers = 0;
  for (const auto id : candidates_)
    remaining_layers =
        std::max(remaining_layers, states_.at(id)->remaining().size());
  const auto available = max_expansions_ - expansions_;
  // Estimate the remaining expansion work for the currently retained paths.
  // A terminal layer is not itself a reason to shrink the beam: do so only
  // when the projected work cannot fit in the remaining expansion budget.
  const auto parents = std::max<std::size_t>(1, beam_.size());
  const auto observed_fanout =
      std::max<std::size_t>(1, (candidates_.size() + parents - 1) / parents);
  // Action fan-out falls as operators are consumed; use a conservative three-quarter remaining horizon.
  const auto horizon = std::max<std::size_t>(1, remaining_layers);
  const auto projected_layers = std::max<std::size_t>(1, (3 * horizon + 1) / 4);
  const auto reserve =
      projected_layers * observed_fanout * (horizon <= 4 ? 2 : 1);
  const auto projected = reserve * adaptive_cap_;
  if (available < projected) {
    const auto target =
        std::clamp<std::size_t>(available / reserve, 1, beam_width_);
    if (target < adaptive_cap_) {
      // Keep the budget guard, but spread the reduction over successive
      // pruning points so one noisy fan-out estimate cannot collapse the beam.
      const auto maximum_step =
          std::max<std::size_t>(1, (adaptive_cap_ + 9) / 10);
      const auto step = std::min(maximum_step, adaptive_cap_ - target);
      adaptive_cap_ -= step;
    }
  }
  return adaptive_cap_;
}
void SearchSession::finish() {
  std::unordered_set<std::string> unique;
  std::vector<std::size_t> completed;
  for (auto id : completed_)
    if (unique.insert(problem_.state_key(states_.at(id))).second)
      completed.push_back(id);
  completed_ = std::move(completed);
  phase_ = "done";
}
SearchRequest SearchSession::advance() {
  if (phase_ == "observe")
    throw std::logic_error("Observation must be committed before advancing");
  if (phase_ == "ranked")
    throw std::logic_error("Selection must be committed before advancing");
  if (phase_ == "score")
    return {"score", 0, 0, {}, {}, candidates_};
  if (phase_ == "quality")
    return {"quality", 0, 0, {}, {}, keep_};
  if (phase_ == "done")
    return {"done", 0, 0, {}, {}, completed_};
  while (parent_pos_ < beam_.size() && !exhausted_) {
    auto parent_id = beam_[parent_pos_];
    if (!actions_loaded_) {
      if (budget_reached()) {
        exhausted_ = true;
        break;
      }
      try {
        actions_ =
            problem_.actions(states_.at(parent_id), false, request_guided_,
                             &action_priorities_, fixed_model_);
      } catch (const ActionBudgetExceeded &) {
        exhausted_ = action_budget_exhausted_ = true;
        break;
      }
      action_pos_ = 0;
      actions_loaded_ = true;
    }
    if (action_pos_ == actions_.size()) {
      ++parent_pos_;
      actions_loaded_ = false;
      continue;
    }
    if (budget_reached()) {
      exhausted_ = true;
      break;
    }
    auto action = actions_[action_pos_++];
    auto child = problem_.transition(states_.at(parent_id), action);
    pending_id_ = next_id_++;
    states_[pending_id_] = child;
    ++expansions_;
    phase_ = "observe";
    return {"observe", parent_id, pending_id_, child, action, {}};
  }
  if (candidates_.empty()) {
    finish();
    return {"done", 0, 0, {}, {}, completed_};
  }
  phase_ = "score";
  return {"score", 0, 0, {}, {}, candidates_};
}
bool SearchSession::commit_observation(std::size_t child_id) {
  if (phase_ != "observe" || child_id != pending_id_)
    throw std::logic_error("Unexpected observation commit");
  auto child = states_.at(child_id);
  bool accepted = true;
  if (problem_.complete(child))
    completed_.push_back(child_id);
  else {
    std::string history;
    for (const auto &action : child->history())
      history += std::to_string(action.size()) + ":" + action;
    if (histories_.insert(history).second)
      candidates_.push_back(child_id);
    else {
      states_.erase(child_id);
      accepted = false;
    }
  }
  // Follow one existing heuristic trajectory continuously. Once its first
  // accepted child is chosen, siblings cannot replace that choice.
  if (accepted && guard_parent_id_ == beam_[parent_pos_] &&
      !guard_candidate_id_ && !guard_complete_id_) {
    if (problem_.complete(child))
      guard_complete_id_ = child_id;
    else
      guard_candidate_id_ = child_id;
  }
  phase_ = "expand";
  return accepted;
}
std::vector<std::size_t>
SearchSession::rank(const Scores &scores, std::size_t /*sampling_slots*/,
                    const std::vector<std::size_t> &excluded,
                    std::optional<std::size_t> guard_index,
                    const std::vector<std::size_t> &order,
                    std::optional<std::size_t> retention_limit) {
  if (phase_ != "score")
    throw std::logic_error("Scores submitted outside scoring phase");
  if (scores.size() != candidates_.size())
    throw std::invalid_argument(
        "PlanValueModel returned the wrong number of scores");
  scores_ = scores;
  validate_scores(scores_);
  excluded_.assign(candidates_.size(), false);
  for (auto i : excluded) {
    if (i >= candidates_.size())
      throw std::invalid_argument("Quality exclusion index is out of range");
    excluded_[i] = true;
  }
  if (!guard_index && guard_candidate_id_) {
    const auto found =
        std::find(candidates_.begin(), candidates_.end(), *guard_candidate_id_);
    if (found != candidates_.end()) {
      const auto index = static_cast<std::size_t>(found - candidates_.begin());
      if (!excluded_[index])
        guard_index = index;
    }
  }
  if (guard_index &&
      (*guard_index >= candidates_.size() || excluded_[*guard_index]))
    throw std::invalid_argument(
        "Heuristic guard must be an eligible candidate");
  const auto available =
      candidates_.size() - std::count(excluded_.begin(), excluded_.end(), true);
  if (!order.empty()) {
    std::vector<bool> ordered(candidates_.size(), false);
    std::vector<std::size_t> eligible_order;
    eligible_order.reserve(available);
    for (auto index : order) {
      if (index >= candidates_.size())
        throw std::invalid_argument("Layer order index is out of range");
      if (!ordered[index] && !excluded_[index]) {
        ordered[index] = true;
        eligible_order.push_back(index);
      }
    }
    // A ranking callback may omit an unknown candidate; retain it after all
    // explicitly ordered candidates instead of silently dropping its path.
    for (std::size_t index = 0; index < candidates_.size(); ++index)
      if (!ordered[index] && !excluded_[index])
        eligible_order.push_back(index);
    fronts_ = {std::move(eligible_order)};
  } else if (excluded.empty())
    fronts_ = pareto_fronts(scores_);
  else {
    Scores eligible;
    std::vector<std::size_t> indices;
    eligible.reserve(available);
    indices.reserve(available);
    for (std::size_t i = 0; i < scores_.size(); ++i)
      if (!excluded_[i]) {
        eligible.push_back(scores_[i]);
        indices.push_back(i);
      }
    fronts_ = pareto_fronts(eligible);
    for (auto &front : fronts_)
      for (auto &index : front)
        index = indices[index];
  }
  const auto layer_beam_width = std::min(
      adaptive_beam_width(), retention_limit.value_or(beam_width_));
  if (!layer_beam_width && available)
    throw std::invalid_argument("Layer retention limit must be positive");
  random_slots_ = training_ && available > layer_beam_width
                      ? static_cast<std::size_t>(
                            std::ceil(layer_beam_width * exploration_fraction_))
                      : 0;
  heuristic_guard_ = guard_index ? 1 : 0;
  const auto take_ordered = [&](std::size_t limit,
                                std::optional<std::size_t> skip = {}) {
    std::vector<std::size_t> selected;
    selected.reserve(std::min(limit, available));
    // The normalized order covers every eligible candidate exactly once.
    // Keep only its leading entries, with no Pareto reserve or replacement.
    for (auto index : fronts_.front()) {
      if (selected.size() == limit)
        break;
      if (index < candidates_.size() && !excluded_[index] &&
          (!skip || index != *skip))
        selected.push_back(index);
    }
    return selected;
  };
  if (guard_index) {
    // The caller supplies the heuristic choice. Reserve it within the model
    // quota before Pareto truncation, including when exploration requests the
    // entire beam. Keep the original fronts intact for layer diagnostics.
    random_slots_ = std::min(random_slots_, layer_beam_width - 1);
    auto remaining_fronts = fronts_;
    for (auto &front : remaining_fronts)
      front.erase(std::remove(front.begin(), front.end(), *guard_index),
                  front.end());
    keep_ = {*guard_index};
    auto model =
        order.empty()
            ? trim(scores_, layer_beam_width - random_slots_ - 1, sample_,
                   &remaining_fronts)
            : take_ordered(layer_beam_width - random_slots_ - 1, guard_index);
    keep_.insert(keep_.end(), model.begin(), model.end());
  } else if (order.empty())
    keep_ = trim(scores_, layer_beam_width - random_slots_, sample_, &fronts_);
  else
    keep_ = take_ordered(layer_beam_width - random_slots_);
  random_slots_ = std::min(random_slots_, available - keep_.size());
  phase_ = "ranked";
  return keep_;
}
std::vector<std::size_t>
SearchSession::select(const std::vector<std::size_t> &exploration) {
  if (phase_ != "ranked")
    throw std::logic_error("Selection submitted outside ranking phase");
  std::unordered_set<std::size_t> selected(keep_.begin(), keep_.end());
  const auto available =
      candidates_.size() - std::count(excluded_.begin(), excluded_.end(), true);
  auto expected = std::min(random_slots_, available - keep_.size());
  if (exploration.size() != expected)
    throw std::invalid_argument("Wrong exploration sample count");
  for (auto i : exploration) {
    if (i >= candidates_.size() || excluded_[i] || !selected.insert(i).second)
      throw std::invalid_argument("Invalid exploration index");
  }
  keep_.insert(keep_.end(), exploration.begin(), exploration.end());
  for (auto &i : keep_)
    i = candidates_[i];
  phase_ = "quality";
  return keep_;
}
void SearchSession::commit_quality() {
  if (phase_ != "quality")
    throw std::logic_error("Quality submitted outside quality phase");
  LayerTrace row{
      ++depth_, candidates_.size(), keep_.size(), completed_.size(), {}};
  row.beam_width = keep_.size();
  row.cost_pruned = cost_pruned_;
  row.quality_pruned = std::count(excluded_.begin(), excluded_.end(), true) - cost_pruned_;
  cost_pruned_ = 0;
  row.heuristic_guard = heuristic_guard_;
  row.random_exploration = random_slots_;
  row.accuracy_threshold = quality_threshold_;
  row.accuracy_relaxations = quality_relaxations_;
  for (const auto &front : fronts_)
    row.fronts.push_back(front.size());
  trace_.push_back(std::move(row));
  beam_ = keep_;
  if (heuristic_guard_enabled_) {
    if (guard_candidate_id_ && std::find(beam_.begin(), beam_.end(),
                                         *guard_candidate_id_) != beam_.end())
      guard_parent_id_ = guard_candidate_id_;
    else
      guard_parent_id_.reset();
    guard_candidate_id_.reset();
  }
  std::unordered_set<std::size_t> live(completed_.begin(), completed_.end());
  live.insert(beam_.begin(), beam_.end());
  for (auto it = states_.begin(); it != states_.end();) {
    if (!live.count(it->first))
      it = states_.erase(it);
    else
      ++it;
  }
  candidates_.clear();
  histories_.clear();
  keep_.clear();
  actions_.clear();
  scores_.clear();
  fronts_.clear();
  excluded_.clear();
  parent_pos_ = action_pos_ = 0;
  actions_loaded_ = false;
  phase_ = "expand";
  if (exhausted_ || beam_.empty())
    finish();
}
}
