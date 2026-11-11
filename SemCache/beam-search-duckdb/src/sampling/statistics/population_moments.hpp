#pragma once
#include "../common/inputs.hpp"
#include <optional>

namespace semcache::sampling {
// A publication's population mass and effective size share the same weights.
// Store only two numbers and a weak lifetime guard, including for large probe
// banks. Direct mutable calls and explicit accuracy quotas stay uncached.
class PopulationMoments {
public:
  using Result = std::pair<double, double>;

private:
  struct Entry {
    std::weak_ptr<const Json> owner;
    std::optional<Result> value;
  };
  std::unordered_map<const Json *, Entry> entries_;
  std::uint64_t cohort_ = 0;

public:
  void reset() {
    entries_.clear();
    cohort_ = 0;
  }
  void cohort(std::uint64_t version) {
    if (cohort_ == version)
      return;
    for (auto &entry : entries_)
      entry.second.value.reset();
    cohort_ = version;
  }
  void register_population(const SharedSamples &rows) {
    auto found = entries_.find(rows.get());
    if (found == entries_.end() || found->second.owner.expired()) {
      if (entries_.size() >= 4096) {
        for (auto it = entries_.begin(); it != entries_.end();)
          if (it->second.owner.expired())
            it = entries_.erase(it);
          else
            ++it;
        if (entries_.size() >= 4096)
          entries_.erase(entries_.begin());
      }
      entries_.insert_or_assign(rows.get(), Entry{rows, {}});
    }
    if (rows->is_object())
      for (const auto &part : rows->at("product"))
        register_population(SharedSamples(rows, &part));
  }
  template <class Compute> Result get(const Json &rows, Compute compute) {
    auto found = entries_.find(&rows);
    if (found == entries_.end() || found->second.owner.expired())
      return compute();
    if (found->second.value)
      return *found->second.value;
    auto result = compute();
    // Recursive product evaluation must not hold an iterator across calls.
    found = entries_.find(&rows);
    if (found != entries_.end() && !found->second.owner.expired())
      found->second.value = result;
    return result;
  }
};
}
