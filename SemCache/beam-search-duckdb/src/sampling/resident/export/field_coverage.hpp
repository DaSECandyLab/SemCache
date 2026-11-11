#pragma once
#include "../../statistics/statistics.hpp"
#include "../../../search/state_core.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <tuple>

namespace semcache::sampling::resident_detail {
// Diagnostic-only accounting: it never supplies priors or changes sampling.
class FieldDepthCoverage {
  enum Source {
    unknown,
    scalar,
    interpolated,
    independent,
    shared,
    exact,
    measured
  };
  using Coordinate = std::tuple<Id, std::uint64_t, std::string, std::string>;
  using Logical = std::tuple<Id, std::string, std::string>;
  std::map<Coordinate, std::pair<Source, bool>> contexts_;
  std::map<Logical, Source> logical_;
  std::array<std::size_t, 5> weighted_{};
  std::set<std::tuple<Id, std::uint64_t, std::string>> pending_;
  std::set<Coordinate> finite_zeros_;
  static Source source(const std::string &s) {
    if (s == "direct" || s == "measured")
      return measured;
    if (s == "exact")
      return exact;
    if (s == "shared")
      return shared;
    if (s == "fallback" || s == "independent")
      return independent;
    if (s == "interpolated")
      return interpolated;
    if (s == "scalar" || s == "prior" || s == "duckdb")
      return scalar;
    return unknown;
  }
  static Json counts(const std::array<std::size_t, 7> &values,
                     std::size_t total) {
    return {{"total", total},
            {"measured", values[measured]},
            {"exact", values[exact]},
            {"shared", values[shared]},
            {"independent", values[independent]},
            {"interpolated", values[interpolated]},
            {"scalar", values[scalar]},
            {"unknown", values[unknown]}};
  }

public:
  void pending(Id oid, std::uint64_t serial, const std::string &field) {
    pending_.emplace(oid, serial, field);
  }
  void resolved(Id oid, std::uint64_t serial, const std::string &field) {
    pending_.erase({oid, serial, field});
  }
  void add(Id oid, std::uint64_t serial, const std::string &field,
           const std::string &depth, const std::string &s, bool usable,
           bool finite_zero = false, const Json *weights = nullptr) {
    const auto value = source(s);
    contexts_.try_emplace({oid, serial, field, depth}, value, usable);
    if (finite_zero)
      finite_zeros_.emplace(oid, serial, field, depth);
    if (weights && weights->is_object()) {
      for (const auto &[name, index] :
           std::array<std::pair<const char *, std::size_t>, 5>{
               {{"direct", 0},
                {"shared", 1},
                {"interpolated", 2},
                {"scalar", 3},
                {"fallback", 4}}}) {
        const auto found = weights->find(name);
        if (found != weights->end() && found->is_number() &&
            found->get<double>() > 0)
          ++weighted_[index];
      }
      if (weights->value("duckdb", 0.) > 0)
        ++weighted_[3];
    } else if (value >= scalar && value <= measured) {
      // Generated-field descriptors only retain a source label. Preserve a
      // one-hot contribution for those records until they carry full fusion
      // weights as well.
      const auto index = value == shared         ? 1
                         : value == interpolated ? 2
                         : value == scalar       ? 3
                         : value == independent  ? 4
                                                 : 0;
      ++weighted_[index];
    }
    auto &slot = logical_[{oid, field, depth}];
    slot = std::max(slot, value);
  }
  Json json() const {
    std::array<std::size_t, 7> counts_{}, logical_counts{};
    std::size_t usable = 0;
    for (const auto &[_, entry] : contexts_) {
      ++counts_[entry.first];
      usable += entry.second;
    }
    for (const auto &[_, value] : logical_)
      ++logical_counts[value];
    return {
        {"field_depth_coverage_protocol", 2},
        {"field_depth_coverage_unit",
         "physical_observation/field/reachable_positive_depth"},
        {"field_depth_count", contexts_.size()},
        {"field_depth_context_count", contexts_.size()},
        {"logical_field_depth_count", logical_.size()},
        {"sampled_field_depth_count", counts_[measured]},
        {"measured_field_depth_count", counts_[measured]},
        {"exact_field_depth_count", counts_[exact]},
        {"shared_field_depth_count", counts_[shared]},
        {"fallback_field_depth_count", counts_[independent]},
        {"interpolated_field_depth_count", counts_[interpolated]},
        {"scalar_field_depth_count", counts_[scalar]},
        {"estimated_field_depth_count",
         counts_[interpolated] + counts_[scalar]},
        {"unknown_field_depth_count", counts_[unknown]},
        {"direct_weighted_field_depth_count", weighted_[0]},
        {"shared_weighted_field_depth_count", weighted_[1]},
        {"interpolated_weighted_field_depth_count", weighted_[2]},
        {"scalar_weighted_field_depth_count", weighted_[3]},
        {"estimated_weighted_field_depth_count", weighted_[2] + weighted_[3]},
        {"fallback_weighted_field_depth_count", weighted_[4]},
        {"finite_zero_field_depth_count", finite_zeros_.size()},
        {"pending_field_count", pending_.size()},
        {"usable_output_survival_depth_count", usable},
        {"unknown_output_survival_depth_count", contexts_.size() - usable},
        {"field_depth_coverage", counts(counts_, contexts_.size())},
        {"logical_field_depth_coverage",
         counts(logical_counts, logical_.size())}};
  }
};
}
