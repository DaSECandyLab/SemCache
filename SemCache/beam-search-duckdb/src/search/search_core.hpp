#pragma once
#include "state_core.hpp"
#include <chrono>
#include <functional>
#include <map>
#include <optional>
#include <unordered_set>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace semcache {
using Scores = std::vector<std::array<double, 2>>;
using Fronts = std::vector<std::vector<std::size_t>>;
void validate_scores(const Scores &scores);
Fronts pareto_fronts(const Scores &scores);
using IndexSampler = std::function<std::vector<std::size_t>(
    const std::vector<std::size_t> &, std::size_t)>;
std::vector<std::size_t> trim(const Scores &scores, std::size_t limit,
                              const IndexSampler &sample,
                              const Fronts *fronts = nullptr);
struct SearchRequest {
  std::string kind;
  std::size_t parent_id = 0;
  std::size_t child_id = 0;
  StatePtr child;
  Action action;
  std::vector<std::size_t> ids;
};
struct LayerTrace {
  std::size_t depth, enumerated, retained, complete;
  std::size_t beam_width = 0;
  std::vector<std::size_t> fronts;
  std::size_t quality_pruned = 0;
  std::size_t cost_pruned = 0;
  std::size_t heuristic_guard = 0, random_exploration = 0;
  std::optional<double> accuracy_threshold;
  std::size_t accuracy_relaxations = 0;
};
class SearchSession {
public:
  SearchSession(const StateProblem &problem, StatePtr initial,
                const std::vector<StatePtr> &completed, std::size_t beam_width,
                std::size_t max_expansions, double max_seconds, bool training,
                double exploration_fraction, double elapsed,
                ActionPriorities action_priorities, IndexSampler sample,
                std::string mode = "heuristic_requests",
                std::optional<std::string> fixed_model = {});
  void validate_problem(const StateProblem &problem) const;
  SearchRequest advance();
  bool commit_observation(std::size_t child_id);
  // guard_index is an eligible candidate position supplied by the caller's
  // heuristic. It consumes one deterministic slot before truncation.
  std::vector<std::size_t> rank(const Scores &scores,
                                std::size_t sampling_slots = 0,
                                const std::vector<std::size_t> &excluded = {},
                                std::optional<std::size_t> guard_index = {},
                                const std::vector<std::size_t> &order = {},
                                std::optional<std::size_t> retention_limit = {});
  std::vector<std::size_t> select(const std::vector<std::size_t> &exploration);
  void commit_quality();
  void set_cost_pruned(std::size_t count) { cost_pruned_ = count; }
  void set_quality_threshold(std::optional<double> threshold, std::size_t relaxations) {
    quality_threshold_ = threshold;
    quality_relaxations_ = relaxations;
  }
  const std::vector<std::size_t> &completed_ids() const { return completed_; }
  const std::vector<LayerTrace> &trace() const { return trace_; }
  std::size_t expansions() const { return expansions_; }
  // True only when candidate expansion stopped because a budget was reached.
  bool exhausted() const { return exhausted_; }
  bool action_budget_exhausted() const { return action_budget_exhausted_; }
  // Actual exploration capacity after deterministic retention and exclusions.
  std::size_t random_slots() const { return random_slots_; }
  std::size_t resident_states() const { return states_.size(); }
  std::string heuristic_guard_status() const;
  std::optional<std::size_t> heuristic_guard_complete_id() const {
    return guard_complete_id_;
  }
  StatePtr state(std::size_t id) const { return states_.at(id); }
  std::vector<StatePtr> live_states() const {
    std::vector<StatePtr> result;
    result.reserve(states_.size());
    for (const auto &entry : states_)
      result.push_back(entry.second);
    return result;
  }

private:
  StateProblem problem_;
  std::unordered_map<std::size_t, StatePtr> states_;
  std::vector<std::size_t> beam_, completed_, candidates_, keep_;
  std::unordered_set<std::string> histories_;
  std::vector<Action> actions_;
  ActionPriorities action_priorities_;
  IndexSampler sample_;
  std::size_t next_id_ = 1, parent_pos_ = 0, action_pos_ = 0, pending_id_ = 0;
  std::size_t beam_width_, max_expansions_,
      expansions_ = 0, depth_ = 0, random_slots_ = 0, heuristic_guard_ = 0;
  std::size_t adaptive_cap_ = 0;
  double max_seconds_, exploration_fraction_;
  bool training_, exhausted_ = false, action_budget_exhausted_ = false,
                  actions_loaded_ = false;
  bool heuristic_guard_enabled_ = false;
  bool request_guided_ = false;
  std::optional<std::string> fixed_model_;
  std::optional<std::size_t> guard_parent_id_, guard_candidate_id_,
      guard_complete_id_;
  std::string phase_ = "expand";
  std::chrono::steady_clock::time_point start_;
  Scores scores_;
  Fronts fronts_;
  std::vector<bool> excluded_;
  std::vector<LayerTrace> trace_;
  std::optional<double> quality_threshold_;
  std::size_t quality_relaxations_ = 0;
  std::size_t cost_pruned_ = 0;
  bool budget_reached() const;
  std::size_t adaptive_beam_width();
  void finish();
};
}
