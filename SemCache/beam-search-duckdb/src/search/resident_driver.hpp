#pragma once
#include "../sampling/environment.hpp"
#include "resident_random.hpp"
#include "search_core.hpp"
#include <functional>
#include <map>

namespace semcache {
struct ResidentOptions {
  std::size_t beam_width, max_expansions;
  double max_seconds;
  bool training;
  double exploration_fraction;
  std::optional<double> minimum_accuracy;
  bool trace_timings = false;
  ActionPriorities action_priorities;
  std::string mode = "heuristic_requests";
  std::optional<std::string> fixed_model;
  std::size_t planner_workers = 1;
  std::string preference = "fastest";
  std::string layer_pruning = "pareto";
  std::size_t max_structure_groups = 15, max_implementations_per_group = 6;
};
struct ResidentPrediction {
  Scores scores;
  std::vector<std::optional<double>> times, accuracies;
  std::vector<std::optional<std::string>> time_reasons;
  // Position within this prediction batch, rather than a resident state ID.
  std::optional<std::size_t> guard_index;
};
struct LayerRanking {
  Scores scores;
  std::vector<std::size_t> excluded;
  // Optional deterministic order for non-Pareto layer pruning.
  std::vector<std::size_t> order;
  std::optional<double> effective_minimum;
  std::size_t relaxations = 0;
};
// Keep both objectives for Pareto layers; apply constraints to completions.
LayerRanking rank_layer(const ResidentPrediction &prediction,
                        const std::string &preference,
                        std::optional<double> minimum,
                        const std::string &layer_pruning = "pareto");
struct CompletionSelection {
  std::vector<std::size_t> selected;
  std::vector<std::string> statuses;
  std::optional<bool> constraint_satisfied;
  std::optional<std::size_t> accurate_index;
};
CompletionSelection
select_completions(const Scores &scores,
                   const std::vector<std::optional<double>> &times,
                   const std::vector<std::optional<double>> &accuracies,
                   std::optional<double> minimum);
struct ResidentResult {
  std::vector<StatePtr> completed;
  Scores scores;
  ResidentPrediction prediction;
  CompletionSelection selection;
  std::vector<LayerTrace> trace;
  std::size_t expansions = 0;
  bool exhausted = false, action_budget_exhausted = false;
  std::string heuristic_guard_status = "disabled";
  std::optional<std::size_t> heuristic_guard_complete;
  std::map<std::string, std::array<double, 2>> timings;
  std::map<std::string, std::vector<std::array<std::int64_t, 2>>> timing_spans;
  struct StateDiagnostic {
    std::string kind;
    std::size_t layer;
    StatePtr state;
    std::array<std::int64_t, 2> enumerate_span, observe_span;
    std::map<Id, std::shared_ptr<const sampling::Json>> statistics;
    std::vector<std::array<std::int64_t, 2>> observe_spans{};
  };
  std::vector<StateDiagnostic> state_diagnostics;
  struct GroupTrace {
    std::size_t depth, candidate_groups;
    std::vector<std::size_t> retained_group_sizes;
    std::vector<std::array<double, 2>> retained_group_means;
  };
  std::vector<GroupTrace> group_trace;
  sampling::Json join_filter_trace = sampling::Json::array();
};
using ResidentPredictor =
    std::function<ResidentPrediction(const std::vector<StatePtr> &)>;
class NativeSearchEngine {
public:
  NativeSearchEngine(const StateProblem &problem,
                     std::shared_ptr<sampling::SearchEnvironment> sampler)
      : problem_(problem), sampler_(std::move(sampler)) {}
  ResidentResult search(const ResidentOptions &options, ResidentRandom &random,
                        const std::vector<StatePtr> &original,
                        const ResidentPredictor &predict);
  const std::shared_ptr<sampling::SearchEnvironment> &sampler() const {
    return sampler_;
  }
  std::size_t expansions() const { return last_expansions_; }

private:
  StateProblem problem_;
  std::shared_ptr<sampling::SearchEnvironment> sampler_;
  std::size_t last_expansions_ = 0;
};
}
