#pragma once
#include "../../statistics/statistics.hpp"
#include "decision_mask.hpp"
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace semcache::sampling::resident_detail {
struct ConditionEstimate {
  double rate = 0, effective = 0, known_coverage = 0;
  std::size_t samples = 0, conditions = 0;
  bool relaxed = false;
  std::vector<std::string> retained;
};

// Rows belong to the sampler's immutable strata, not to a State. The pool
// stores pointers and compact answer masks; it never owns a second text copy.
class FilterPool {
  std::vector<const Json *> rows_;
  std::vector<double> weights_;
  std::vector<std::size_t> strata_;
  std::map<std::size_t, double> stratum_totals_;
  double total_weight_ = 0;
  std::unordered_map<std::string, std::size_t> coordinates_;
  std::map<std::string, DecisionMask> tasks_;
  std::uint64_t version_ = 0;

public:
  FilterPool(std::vector<const Json *> rows, std::vector<double> weights,
             std::vector<std::size_t> strata);
  std::optional<std::size_t> coordinate(const std::string &id) const;
  bool compatible(std::size_t coordinate, const Json &row,
                  const std::set<std::string> &fields) const;
  std::optional<bool> decision(const std::string &task,
                               std::size_t coordinate) const;
  void publish(const std::string &task, std::size_t coordinate, bool passed);
  std::optional<ConditionEstimate>
  estimate(const std::string &task, const std::vector<std::string> &conditions,
           double minimum, bool relax) const;
  std::uint64_t version() const { return version_; }
};
}
