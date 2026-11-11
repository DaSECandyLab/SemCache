#pragma once
#include "../statistics/statistics.hpp"
#include <cmath>
#include <map>
#include <string>

namespace semcache::sampling {
inline bool prefix_population_boundary(const std::string &kind) {
  // Group/aggregate outputs form a new value population. Their row-count
  // ratio is not a survival probability for the consumed input fields.
  return kind.find("AGGREGATE") != std::string::npos;
}

inline bool prefix_preserving_operator(const std::string &kind) {
  // CTE_SCAN only rebinds the producer's fields; it has no predicate.
  return kind.find("PROJECTION") != std::string::npos || kind == "ORDER_BY" ||
         kind == "CROSS_PRODUCT" || kind == "CTE_SCAN";
}

// Missing depths use the operator's overall rate without creating observations
// or borrowing a different depth's conditional rate.
inline Json
complete_prefix_selectivity(const std::map<std::string, bool> &support,
                            const Json &evidence, const Json &scalar,
                            bool preserving = false,
                            bool empty_population = false,
                            bool scalar_floor = false) {
  auto finite = [](const Json &value) {
    return value.is_number() && std::isfinite(value.get<double>());
  };
  auto measured = [&](const std::string &depth) -> const Json * {
    if (!evidence.is_object() || !evidence.contains(depth))
      return nullptr;
    const auto &entry = evidence.at(depth);
    return entry.is_object() && finite(entry.value("rate", Json(nullptr))) &&
                   entry.value("observed", std::size_t(0)) > 0
               ? &entry
               : nullptr;
  };
  // For joins this is the whole source-row population, not pair acceptance.
  const auto zero = scalar_floor ? nullptr : measured("0");
  const Json overall = zero ? zero->at("rate") : scalar;
  Json result = Json::object();
  for (const auto &[key, supported] : support) {
    Json entry = {{"rate", nullptr},
                  {"source", "unknown"},
                  {"observed", 0},
                  {"effective_samples", 0.},
                  {"anchor_depths", Json::array()}};
    const auto exact = scalar_floor ? nullptr : measured(key);
    if (exact) {
      entry = *exact;
      entry["source"] = "measured";
      entry["effective_samples"] = exact->value("effective_samples", 0.);
      entry["anchor_depths"] = Json::array({key});
    } else if (!supported || empty_population) {
      entry["rate"] = 0.;
      entry["source"] = "empty_population";
    } else if (preserving) {
      entry["rate"] = 1.;
      entry["source"] = "exact";
    } else if (finite(overall)) {
      entry["rate"] = overall;
      entry["source"] = "operator_scalar";
      if (scalar_floor)
        entry["floor_applied"] = true;
      entry["anchor_depths"] = Json::array({"0"});
    }
    result[key] = std::move(entry);
  }
  return result;
}

inline std::map<std::string, bool> prefix_depth_support(const Json &summary) {
  const bool populated =
      !summary.is_object() || summary.value("count", Json(nullptr)) != 0;
  std::map<std::string, bool> result{{"0", populated}};
  if (summary.is_object() && summary.contains("sketches"))
    for (auto d = summary.at("sketches").begin();
         d != summary.at("sketches").end(); ++d)
      result[d.key()] =
          populated && (d.key() == "0" ||
                        !d.value().value("entries", Json::array()).empty());
  return result;
}
}
