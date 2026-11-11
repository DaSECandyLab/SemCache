#pragma once

#include "state_core.hpp"
#include <vector>

namespace semcache {
struct ResidentPrediction;

struct GroupedPruning {
  std::vector<std::size_t> order;
  std::vector<std::size_t> membership;
  std::vector<std::array<double, 2>> mean_predictions;
  std::size_t groups = 0;
};

// Return only admitted candidates, interleaving groups for budget reductions.
// Prediction pairs and executable state identities are never modified.
GroupedPruning prune_structure_groups(const StateProblem &problem,
                                      const std::vector<StatePtr> &states,
                                      const ResidentPrediction &prediction,
                                      std::size_t maximum_groups,
                                      std::size_t per_group);
}
