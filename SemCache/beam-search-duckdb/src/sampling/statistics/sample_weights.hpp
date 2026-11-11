#pragma once
#include "../common/inputs.hpp"
#include <optional>

namespace semcache::sampling {
// Quota-dependent row weights share only owned, immutable populations. An
// admission change or rolled-back trial resets all pointer entries together.
class RowWeights {
  struct Entry {
    std::optional<double> weight;
  };
  std::unordered_map<const Json *, Entry> rows_;
  struct Owner {
    std::size_t retained_rows;
    std::vector<const Json *> rows;
  };
  std::map<SharedSamples, Owner, std::owner_less<SharedSamples>> owners_;
  std::deque<SharedSamples> order_;
  std::size_t retained_rows_ = 0;
  std::unordered_map<const Json *, std::weak_ptr<const Json>> populations_;
  std::size_t limit_, hits_ = 0, misses_ = 0;
  std::uint64_t cohort_ = 0;

  void evict_oldest() {
    const auto &owner = order_.front();
    auto found = owners_.find(owner);
    for (auto row : found->second.rows)
      rows_.erase(row);
    // The weak registration index is capped separately. Remove aliases of the
    // evicted allocation so a still-live caller can register it again.
    for (auto population = populations_.begin();
         population != populations_.end();)
      if (!population->second.owner_before(owner) &&
          !owner.owner_before(population->second))
        population = populations_.erase(population);
      else
        ++population;
    retained_rows_ -= found->second.retained_rows;
    owners_.erase(found);
    order_.pop_front();
  }

public:
  explicit RowWeights(std::size_t limit = 65536) : limit_(limit) {}
  void reset() {
    rows_.clear();
    owners_.clear();
    order_.clear();
    retained_rows_ = 0;
    populations_.clear();
    cohort_ = 0;
  }
  void cohort(std::uint64_t version) {
    if (cohort_ != version) {
      reset();
      cohort_ = version;
    }
  }
  void register_population(const SharedSamples &population,
                           std::size_t retained_rows = 0) {
    if (!retained_rows)
      retained_rows = PopulationFingerprints::materialized_rows(*population);
    if (!retained_rows || !limit_ || retained_rows > limit_)
      return;
    auto owner = owners_.find(population);
    if (owner == owners_.end()) {
      // One alias retains its whole allocation. Evict only enough complete
      // owners to admit the next population, keeping other evaluated rows.
      while (!order_.empty() && retained_rows > limit_ - retained_rows_)
        evict_oldest();
      owner = owners_.emplace(population, Owner{retained_rows, {}}).first;
      order_.push_back(population);
      retained_rows_ += retained_rows;
    }
    auto found = populations_.find(population.get());
    if (found != populations_.end() && !found->second.expired())
      return;
    if (population->is_object()) {
      const auto &parts = population->at("product");
      for (const auto &part : parts)
        register_population(SharedSamples(population, &part), retained_rows);
    } else {
      if (population->size() > limit_ || !limit_)
        return;
      for (const auto &row : *population) {
        if (rows_.count(&row))
          continue;
        rows_.emplace(&row, Entry{{}});
        owner->second.rows.push_back(&row);
      }
    }
    if (populations_.size() >= 1024)
      populations_.clear();
    populations_[population.get()] = population;
  }
  template <typename Evaluate>
  double weight(const Json &row, Evaluate &&evaluate) {
    auto found = rows_.find(&row);
    if (found != rows_.end() && found->second.weight) {
      ++hits_;
      return *found->second.weight;
    }
    auto value = evaluate(row);
    ++misses_;
    if (found != rows_.end())
      found->second.weight = value;
    return value;
  }
  std::size_t hits() const { return hits_; }
  std::size_t misses() const { return misses_; }
};
}
