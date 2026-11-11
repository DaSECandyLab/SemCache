#include "relational_estimate.hpp"
#include <cmath>

namespace semcache::sampling {
namespace {
std::optional<double> rate(const Json &value) {
  if (!value.is_number())
    return {};
  const auto n = value.get<double>();
  return std::isfinite(n) && n >= 0. && n <= 1. ? std::optional<double>(n)
                                                : std::nullopt;
}
}
Json mean_relational_estimate(const Json &sampled, const Json &duckdb) {
  const auto sample_rate =
      sampled.value("sampled_selectivity_complete", false)
          ? rate(sampled.value("sampling_selectivity",
                               sampled.value("selectivity", Json(nullptr))))
          : std::nullopt;
  const auto duckdb_rate = rate(duckdb.value(
      "duckdb_selectivity", duckdb.value("selectivity", Json(nullptr))));
  Json result = duckdb;
  result.update(sampled);
  // Historical fusion metadata is not a new source observation.
  for (const auto *key : {"selectivity_fusion", "estimated_selectivity",
                          "survival_estimation_basis"})
    result.erase(key);
  if (!sample_rate && !duckdb_rate) {
    result["selectivity"] = nullptr;
    result["output_rows"] = nullptr;
    return result;
  }
  const double mean = sample_rate && duckdb_rate
                          ? .5 * *sample_rate + .5 * *duckdb_rate
                      : sample_rate ? *sample_rate
                                    : *duckdb_rate;
  result["sampling_selectivity"] =
      sample_rate ? Json(*sample_rate) : Json(nullptr);
  result["duckdb_selectivity"] =
      duckdb_rate ? Json(*duckdb_rate) : Json(nullptr);
  for (const auto *key : {"input_rows", "output_rows", "population_input_rows"})
    if (duckdb.contains(key))
      result[std::string("duckdb_") + key] = duckdb.at(key);
  const auto inputs =
      sample_rate ? sampled.value("population_input_rows", Json(nullptr))
                  : duckdb.value("population_input_rows", Json(nullptr));
  if (inputs.is_array() && !inputs.empty()) {
    double input = 1.;
    bool known = true;
    for (const auto &side : inputs) {
      known &= side.is_number() && std::isfinite(side.get<double>()) &&
               side.get<double>() >= 0.;
      if (known)
        input *= side.get<double>();
    }
    if (known && std::isfinite(input)) {
      result["input_rows"] = input;
      result["population_input_rows"] = inputs;
    }
  }
  result["selectivity"] = mean;
  result["estimated_selectivity"] = mean;
  result["selectivity_fusion"] = {
      {"method", "arithmetic_mean"},
      {"weights",
       {{"sampling", sample_rate ? (duckdb_rate ? .5 : 1.) : 0.},
        {"duckdb", duckdb_rate ? (sample_rate ? .5 : 1.) : 0.}}}};
  result["survival_estimated"] = true;
  result["cardinality_basis"] =
      sample_rate && duckdb_rate ? "arithmetic mean of sampled and DuckDB "
                                   "selectivities on the same input domain"
      : sample_rate ? "sampled selectivity; DuckDB estimate unavailable"
                    : "DuckDB selectivity; population sample unavailable";
  result["survival_estimation_basis"] = result.at("cardinality_basis");
  const auto input = result.value("input_rows", Json(nullptr));
  result["output_rows"] = input.is_number() &&
                                  std::isfinite(input.get<double>()) &&
                                  input.get<double>() >= 0.
                              ? Json(input.get<double>() * mean)
                              : Json(nullptr);
  result["population_output_rows"] = result.at("output_rows");
  return result;
}
}
