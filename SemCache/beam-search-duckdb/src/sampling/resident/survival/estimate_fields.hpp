#pragma once
#include "../../statistics/statistics.hpp"

namespace semcache::sampling::resident_detail {
// Intrinsic estimate fields are replaced, not merged with older derived
// values. Branch-specific annotations are retained.
inline void clear_estimate_fields(Json &info) {
  for (const auto *name :
       {"conditional_point_estimate", "conditional_effective_samples",
        "conditional_known_coverage", "conditional_count",
        "retained_condition_count",   "retained_conditions",
        "conditional_relaxed",        "conditional_basis",
        "estimated_selectivity",      "survival_estimated",
        "survival_estimation_basis",  "shared_selectivity",
        "shared_source_contexts",     "shared_prior_strength",
        "shared_prior_rate",          "shared_prior_version",
        "fallback_selectivity",       "selectivity_fusion",
        "depth_selectivity_sources",  "depth_selectivity_fusion"})
    info.erase(name);
  if (auto it = info.find("reliability"); it != info.end() && it->is_object()) {
    it->erase("output_rows");
    if (it->empty())
      info.erase(it);
  }
}
inline void replace_estimate_fields(Json &info, const Json &observation) {
  clear_estimate_fields(info);
  info.update(observation);
}
}
