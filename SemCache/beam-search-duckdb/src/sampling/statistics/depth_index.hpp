#pragma once
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace semcache::sampling {
// The population sum retains the original ascending reduction order; each
// immutable field/depth computes it once, preserving prepared float evidence.
class DepthPopulation {
  std::map<std::size_t, double> lengths_;
  mutable std::map<std::size_t, double> totals_;

public:
  explicit DepthPopulation(std::map<std::size_t, double> lengths)
      : lengths_(std::move(lengths)) {}
  double at_least(std::size_t depth) const {
    auto found = totals_.find(depth);
    if (found != totals_.end())
      return found->second;
    double count = 0;
    for (auto it = lengths_.lower_bound(depth); it != lengths_.end(); ++it)
      count += it->second;
    totals_.emplace(depth, count);
    return count;
  }
};
class DepthCounts {
  std::vector<std::size_t> lengths_, passed_;

public:
  explicit DepthCounts(std::vector<std::pair<std::size_t, bool>> rows) {
    std::sort(rows.begin(), rows.end());
    lengths_.reserve(rows.size());
    passed_.resize(rows.size() + 1);
    for (const auto &row : rows)
      lengths_.push_back(row.first);
    for (std::size_t i = rows.size(); i; --i)
      passed_[i - 1] = passed_[i] + rows[i - 1].second;
  }
  std::pair<std::size_t, std::size_t> at_least(std::size_t depth) const {
    auto index = std::lower_bound(lengths_.begin(), lengths_.end(), depth) -
                 lengths_.begin();
    return {lengths_.size() - index, passed_[index]};
  }
};
class DepthCohort {
  std::vector<std::pair<std::string, std::size_t>> members_;

public:
  explicit DepthCohort(std::vector<std::pair<std::string, std::size_t>> members)
      : members_(std::move(members)) {}
  DepthCounts
  observe(const std::set<std::string> &reached,
          const std::optional<std::set<std::string>> &eligible) const {
    std::vector<std::pair<std::size_t, bool>> rows;
    rows.reserve(members_.size());
    for (const auto &member : members_)
      if (!eligible || eligible->count(member.first))
        rows.emplace_back(member.second, reached.count(member.first));
    return DepthCounts(std::move(rows));
  }
};
}
