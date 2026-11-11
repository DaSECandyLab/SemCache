#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace semcache::sampling::resident_detail {
// A cosine kernel on sparse binary conditions plus one constant coordinate.
// Only scalar summaries live here. No state, row, prompt or sketch is owned.
class ContextPrior {
public:
  using Conditions = std::vector<std::uint64_t>;
  struct Estimate {
    double rate, strength;
    std::size_t donors, lookups;
  };

  bool publish(const Conditions &conditions, double rate, double effective) {
    if (!std::isfinite(rate) || rate < 0 || rate > 1 ||
        !std::isfinite(effective) || effective <= 0)
      return false;
    auto [entry, inserted] = entries_.try_emplace(conditions);
    if (!inserted && entry->second.effective >= effective)
      return false;
    if (!inserted)
      accumulate(conditions, entry->second, -1.);
    entry->second = {effective, (effective * rate + .5) / (effective + 1.)};
    accumulate(conditions, entry->second, 1.);
    ++version_;
    return true;
  }

  std::optional<Estimate> estimate(const Conditions &conditions,
                                   double local_effective) const {
    Totals total = base_;
    std::size_t lookups = 0, donors = entries_.size();
    for (const auto c : conditions) {
      ++lookups;
      const auto it = coordinates_.find(c);
      if (it != coordinates_.end()) {
        total.mass += it->second.mass;
        total.passed += it->second.passed;
      }
    }
    const double norm = std::sqrt(1. + conditions.size());
    const auto local = entries_.find(conditions);
    if (local != entries_.end() && local->second.effective <= local_effective) {
      --donors;
      const double mass = local->second.effective * norm;
      total.mass -= mass;
      total.passed -= mass * local->second.rate;
    }
    if (!donors || !(total.mass > 0))
      return std::nullopt;
    return Estimate{std::clamp(total.passed / total.mass, 0., 1.),
                    total.mass / norm / donors, donors, lookups};
  }

  std::uint64_t version() const { return version_; }
  std::size_t size() const { return entries_.size(); }

private:
  struct ConditionHash {
    std::size_t operator()(const Conditions &conditions) const {
      std::size_t hash = 0;
      for (const auto c : conditions)
        hash ^= c + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
      return hash;
    }
  };
  struct Evidence {
    double effective = 0., rate = 0.;
  };
  struct Totals {
    double mass = 0., passed = 0.;
  };
  void accumulate(const Conditions &conditions, const Evidence &source,
                  double sign) {
    const double mass =
        sign * source.effective / std::sqrt(1. + conditions.size());
    base_.mass += mass;
    base_.passed += mass * source.rate;
    for (const auto c : conditions) {
      auto &coordinate = coordinates_[c];
      coordinate.mass += mass;
      coordinate.passed += mass * source.rate;
    }
  }
  std::unordered_map<Conditions, Evidence, ConditionHash> entries_;
  std::unordered_map<std::uint64_t, Totals> coordinates_;
  Totals base_;
  std::uint64_t version_ = 0;
};

struct EvidenceFamily {
  std::uint64_t task = 0;
  std::vector<std::vector<std::string>> sides;
  ContextPrior::Conditions generators, barriers;
  bool operator<(const EvidenceFamily &other) const {
    return std::tie(task, sides, generators, barriers) <
           std::tie(other.task, other.sides, other.generators, other.barriers);
  }
};

struct EvidenceNode {
  ContextPrior::Conditions conditions, input_conditions, generators, barriers;
  ContextPrior *prior = nullptr;
  struct DepthPriors {
    using Key = std::tuple<std::string, std::string, std::string>;
    std::map<Key, ContextPrior> fields;
    std::uint64_t version = 0;
    std::map<std::string, std::uint64_t> exposure_versions;
    std::uint64_t version_for(const std::string &exposure) const {
      if (exposure.empty())
        return version;
      const auto found = exposure_versions.find(exposure);
      return found == exposure_versions.end() ? 0 : found->second;
    }
  };
  DepthPriors *depth_priors = nullptr;
};
}
