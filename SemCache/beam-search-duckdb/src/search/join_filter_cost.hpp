#pragma once

#include "../sampling/environment.hpp"

namespace semcache {
struct JoinFilterDecision {
  std::size_t filter_index, join_index;
  Id filter_operator, join_node;
  double input_rows, peer_rows, join_output, filter_output;
  double join_seconds_per_row, filter_seconds_per_row;
  double join_first_seconds, filter_first_seconds;
};

struct JoinFilterPruning {
  std::vector<std::size_t> excluded;
  std::vector<JoinFilterDecision> decisions;
};

JoinFilterPruning
prune_join_filter_cost(const StateProblem &problem,
                       const std::vector<StatePtr> &states,
                       const std::vector<sampling::Json> &statistics,
                       const sampling::Json &calibration);
sampling::Json join_filter_decisions(const JoinFilterPruning &result);
std::vector<sampling::Json>
join_filter_statistics(const sampling::SearchEnvironment &environment,
                       const std::vector<StatePtr> &states);
}
