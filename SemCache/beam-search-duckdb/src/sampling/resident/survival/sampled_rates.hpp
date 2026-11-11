#pragma once
#include "source_evidence.hpp"
#include <cstdint>
#include <deque>
#include <type_traits>
#include <unordered_map>

namespace semcache::sampling::resident_detail {
// Store scalars, not JSON trees: a hit avoids metadata preparation, depth
// reduction and a deep clone of the cached result. JSON is rendered only at
// the existing publication boundary, preserving its complete public schema.
struct SampledStratumRate {
  double population;
  std::size_t observed, passed;
};
struct SampledDepthRate {
  std::size_t depth;
  std::optional<double> rate;
  double population, effective_samples;
  std::size_t observed, passed;
  bool reliable, match_incomplete;
  std::vector<SampledStratumRate> strata;
};
using SampledRates = std::vector<SampledDepthRate>;

// Local, length-delimited binary identity, with exact probability type/bits.
// No pointer identity or digest-only match: hash collisions compare all bytes.
class SampledRateKey {
  std::string bytes_;

public:
  template <typename T> void scalar(const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes_.append(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  void text(const std::string &value) {
    scalar(value.size());
    bytes_.append(value);
  }
  void probability(const Json &value) {
    scalar(value.type());
    if (value.is_number_unsigned())
      scalar(value.get<std::uint64_t>());
    else if (value.is_number_integer())
      scalar(value.get<std::int64_t>());
    else if (value.is_number_float())
      scalar(value.get<double>());
    else if (!value.is_null())
      text(value.dump());
  }
  void sources(const SourceEvidence::Inclusion &values) {
    scalar(values.size());
    for (const auto &entry : values) {
      text(entry.first);
      probability(entry.second);
    }
  }
  const std::string &bytes() const { return bytes_; }
};

class SampledRateCache {
  struct Entry {
    std::string key;
    std::shared_ptr<const SampledRates> rates;
    std::size_t hash, bytes;
  };
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<const Entry>>>
      bank_;
  std::deque<std::shared_ptr<const Entry>> order_;
  std::size_t retained_ = 0;
  const std::size_t max_entries_, max_bytes_;

public:
  explicit SampledRateCache(std::size_t max_entries = 512,
                            std::size_t max_bytes = 8 * 1024 * 1024)
      : max_entries_(max_entries), max_bytes_(max_bytes) {}
  std::shared_ptr<const SampledRates> find(const SampledRateKey &key,
                                           std::size_t hash) const {
    auto found = bank_.find(hash);
    if (found != bank_.end())
      for (const auto &entry : found->second)
        if (entry->key == key.bytes())
          return entry->rates;
    return {};
  }
  void remember(const SampledRateKey &key, std::size_t hash,
                std::shared_ptr<const SampledRates> rates) {
    std::size_t bytes = sizeof(Entry) + key.bytes().size() +
                        sizeof(SampledRates) +
                        rates->size() * sizeof(SampledDepthRate);
    for (const auto &rate : *rates)
      bytes += rate.strata.size() * sizeof(SampledStratumRate);
    if (!max_entries_ || bytes > max_bytes_)
      return;
    while (order_.size() >= max_entries_ || retained_ > max_bytes_ - bytes) {
      const auto oldest = order_.front();
      auto &bucket = bank_.at(oldest->hash);
      bucket.erase(std::find(bucket.begin(), bucket.end(), oldest));
      if (bucket.empty())
        bank_.erase(oldest->hash);
      retained_ -= oldest->bytes;
      order_.pop_front();
    }
    auto entry = std::make_shared<const Entry>(
        Entry{key.bytes(), std::move(rates), hash, bytes});
    bank_[hash].push_back(entry);
    order_.push_back(std::move(entry));
    retained_ += bytes;
  }
};

inline Json publish_sampled_rates(const SampledRates &rates) {
  Json result = Json::object();
  for (const auto &value : rates) {
    Json strata = Json::array();
    for (const auto &stratum : value.strata)
      strata.push_back({{"population", stratum.population},
                        {"observed", stratum.observed},
                        {"passed", stratum.passed}});
    result[std::to_string(value.depth)] = {
        {"rate", value.rate ? Json(*value.rate) : Json(nullptr)},
        {"population", value.population},
        {"observed", value.observed},
        {"passed", value.passed},
        {"effective_samples", value.effective_samples},
        {"reliable", value.reliable},
        {"match_incomplete", value.match_incomplete},
        {"strata", std::move(strata)},
        {"precision_basis", "actual distinct-source effective samples; "
                            "unexhausted matches remain a survival "
                            "approximation, no per-node interval"}};
  }
  return result;
}
}
