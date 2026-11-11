#pragma once

// Hot-path cache-history feature reduction shared by the native sampler and
// the Python value-model adapter.  This header has no DuckDB or Python
// dependency; it consumes the versioned JSON history already imported by the
// native sampler and returns the fixed twelve-number §4.2 vector.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace semcache::statistics {
using Json = nlohmann::ordered_json;

inline bool finite_nonnegative(const Json &value) {
  if (!value.is_number())
    return false;
  const auto number = value.get<double>();
  return std::isfinite(number) && number >= 0;
}

inline double log_value(const Json &value) {
  return std::log1p(std::max(0., value.get<double>()));
}

// Return [mean, max, compatible_count, age, request_distance,
// character_distance, mean_mask, max_mask, count_mask, age_mask,
// request_mask, character_mask].  Unknown history coordinates deliberately
// keep the same masks as the Python reference implementation.
class ExistingHistoryFeatures {
public:
  ExistingHistoryFeatures(const Json &pre, const std::string &model)
      : pre_(pre), model_(model) {}

  // Observe one source while the caller is already walking history_ to build
  // l_existing.  The pointers remain valid because history_ is immutable for
  // the duration of cache_edges().
  void observe(const Json &entry) {
    if (entry.value("model_id", std::string()) != model_)
      return;
    if (!entry.contains("pre") || entry.at("pre").is_null()) {
      complete_ = false;
      return;
    }
    for (const auto *key :
         {"age_seconds", "requests_since", "characters_since"}) {
      const auto value = entry.value(key, Json(nullptr));
      if (!finite_nonnegative(value))
        continue;
      auto &scale = std::strcmp(key, "age_seconds") == 0 ? age_scale_
                    : std::strcmp(key, "requests_since") == 0
                        ? request_scale_
                        : character_scale_;
      scale = std::max(scale, value.get<double>());
    }
    if (entry.at("pre") == pre_)
      compatible_.push_back(&entry);
  }

  Json finish(const Json &matches_value) const {
    Json result = Json::array();
    for (int i = 0; i < 12; ++i)
      result.push_back(0.);
    if (pre_.is_null() || !complete_)
      return result;

    static const Json empty_matches = Json::object();
    const Json &matches =
        matches_value.is_object() ? matches_value : empty_matches;
    result[2] = std::log1p(static_cast<double>(compatible_.size()));
    result[8] = 1.;
    bool matches_complete = true;
    for (const auto *entry : compatible_) {
      const auto id = entry->value("id", std::string());
      const auto value = matches.value(id, Json(nullptr));
      if (!finite_nonnegative(value)) {
        matches_complete = false;
        break;
      }
    }
    if (!matches_complete || compatible_.empty()) {
      result[6] = matches_complete ? 1. : 0.;
      result[7] = matches_complete ? 1. : 0.;
      return result;
    }

    const auto strongest = *std::min_element(
        compatible_.begin(), compatible_.end(),
        [&](const Json *left, const Json *right) {
          const auto left_value =
              matches.value(left->value("id", ""), Json(0.)).get<double>();
          const auto right_value =
              matches.value(right->value("id", ""), Json(0.)).get<double>();
          if (left_value != right_value)
            return left_value > right_value;
          return left->value("id", "") < right->value("id", "");
        });
    double sum = 0.;
    for (const auto *entry : compatible_)
      sum += matches.at(entry->at("id").get<std::string>()).get<double>();
    result[0] = std::log1p(sum / static_cast<double>(compatible_.size()));
    result[1] = log_value(matches.at(strongest->at("id").get<std::string>()));
    result[6] = result[7] = 1.;
    const auto distance = [&](const Json *entry, const char *key, double scale,
                              std::size_t index) {
      const auto value = entry->value(key, Json(nullptr));
      if (finite_nonnegative(value)) {
        result[index] = scale > 0. ? value.get<double>() / scale : 0.;
        result[index + 6] = 1.;
      }
    };
    distance(strongest, "age_seconds", age_scale_, 3);
    distance(strongest, "requests_since", request_scale_, 4);
    distance(strongest, "characters_since", character_scale_, 5);
    return result;
  }

private:
  const Json &pre_;
  std::string model_;
  std::vector<const Json *> compatible_;
  double age_scale_ = 0., request_scale_ = 0., character_scale_ = 0.;
  bool complete_ = true;
};

inline Json existing_history_features(const Json &history, const Json &estimate,
                                      const std::string &model) {
  if (!estimate.contains("pre") || estimate.at("pre").is_null() ||
      !history.is_array())
    return Json::array({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  ExistingHistoryFeatures features(estimate.at("pre"), model);
  for (const auto &entry : history)
    features.observe(entry);
  static const Json empty_matches = Json::object();
  const auto it = estimate.find("l_existing");
  return features.finish(it != estimate.end() ? *it : empty_matches);
}
}
