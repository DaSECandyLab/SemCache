#pragma once
#include "inputs.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semcache::sampling {
// Inclusion of a source occupying `copies` rows in a uniform m-of-N prefix.
inline double source_inclusion(std::size_t population, std::size_t copies,
                               std::size_t draws) {
  draws = std::min(draws, population);
  copies = std::min(copies, population);
  if (draws > population - copies)
    return 1.;
  double log_miss = 0.;
  for (std::size_t i = 0; i < draws; ++i)
    log_miss += std::log1p(-double(copies) / double(population - i));
  return -std::expm1(log_miss);
}
// Private sampling metadata, outside user values. Presence also identifies a
// finite intermediate sample, even when its normalization factor equals one.
inline double sample_factor(const Json &row) {
  const auto value = row.value("_sample_weight_factor", 1.);
  if (!std::isfinite(value) || value <= 0)
    throw std::invalid_argument("Sample expansion must be finite and positive");
  return value;
}
inline void sample_factor(Json &row, double factor) {
  if (!std::isfinite(factor) || factor <= 0)
    throw std::invalid_argument("Sample expansion must be finite and positive");
  row["_sample_weight_factor"] = factor;
}
inline bool sampled_population(const Json &rows) {
  if (rows.is_object()) {
    const auto &parts = rows.at("product");
    return sampled_population(parts[0]) || sampled_population(parts[1]);
  }
  return std::any_of(rows.begin(), rows.end(), [](const Json &row) {
    return row.contains("_sample_weight_factor") ||
           row.value("_sample_match_incomplete", false);
  });
}
inline void merge_sample_origins(Json &target, const Json &source) {
  if (source.value("_sample_match_incomplete", false))
    target["_sample_match_incomplete"] = true;
  if (source.value("_sample_origin_approximate", false))
    target["_sample_origin_approximate"] = true;
  if (!source.contains("_sample_origin_probabilities"))
    return;
  if (!target.contains("_sample_origin_probabilities"))
    target["_sample_origin_probabilities"] = Json::object();
  auto &out = target["_sample_origin_probabilities"];
  for (auto it = source.at("_sample_origin_probabilities").begin();
       it != source.at("_sample_origin_probabilities").end(); ++it) {
    if (!out.contains(it.key()) || out[it.key()].is_null())
      out[it.key()] = it.value();
    else if (!it.value().is_null())
      out[it.key()] =
          std::max(out[it.key()].get<double>(), it.value().get<double>());
  }
}
inline void copy_sample_metadata(Json &target, const Json &source) {
  if (source.contains("_sample_weight_factor"))
    sample_factor(target, sample_factor(source));
  merge_sample_origins(target, source);
}
}
