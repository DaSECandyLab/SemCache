#pragma once
#include "statistics.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <unordered_map>

namespace semcache::sampling {
struct TextMetric {
  std::string encoded;
  std::size_t characters;
};
// Search-local text caching bounds both entry count and encoded/raw bytes.
class TextMetrics {
  std::unordered_map<std::string, std::shared_ptr<const TextMetric>> entries_;
  std::deque<std::string> order_;
  std::size_t bytes_ = 0, hits_ = 0, misses_ = 0;
  static constexpr std::size_t capacity = 1024, byte_budget = 4 * 1024 * 1024;

public:
  std::shared_ptr<const TextMetric> measure(const std::string &value) {
    auto found = entries_.find(value);
    if (found != entries_.end()) {
      ++hits_;
      return found->second;
    }
    auto text = serialize_value(Json(value));
    auto metric = std::make_shared<const TextMetric>(
        TextMetric{text, character_length(text)});
    ++misses_;
    auto bytes = value.size() + text.size();
    if (bytes <= byte_budget) {
      while (!order_.empty() &&
             (entries_.size() >= capacity || bytes_ + bytes > byte_budget)) {
        const auto &old = order_.front();
        bytes_ -= old.size() + entries_.at(old)->encoded.size();
        entries_.erase(old);
        order_.pop_front();
      }
      entries_.emplace(value, metric);
      order_.push_back(value);
      bytes_ += bytes;
    }
    return metric;
  }
  std::size_t hits() const { return hits_; }
  std::size_t misses() const { return misses_; }
};
struct ValueMetric {
  std::size_t identity, characters, bytes, bin;
  bool is_null;
};
// One observation owns this bank. Scalar equality includes its exact type and
// float bits; frequency identity follows serialized equality. Compound values
// are read afresh, so callers never acquire a mutable-value cache contract.
class ValueMetrics {
public:
  explicit ValueMetrics(TextMetrics *texts = nullptr) : text_metrics_(texts) {}
  ValueMetric measure(const Json &value) {
    if (value.is_string())
      return cached(texts_, value.get_ref<const std::string &>(), value);
    if (value.is_number_unsigned())
      return cached(unsigned_, value.get<std::uint64_t>(), value);
    if (value.is_number_integer())
      return cached(integers_, value.get<std::int64_t>(), value);
    if (value.is_number_float()) {
      const auto number = value.get<double>();
      std::uint64_t bits;
      std::memcpy(&bits, &number, sizeof(bits));
      return cached(floats_, bits, value);
    }
    if (value.is_null() || value.is_boolean()) {
      auto &slot = special_[value.is_null() ? 0 : value.get<bool>() ? 2 : 1];
      if (!slot)
        slot = fresh(value);
      return *slot;
    }
    return fresh(value);
  }
  std::size_t serializations() const { return serializations_; }

private:
  std::unordered_map<std::string, ValueMetric> texts_;
  std::unordered_map<std::int64_t, ValueMetric> integers_;
  std::unordered_map<std::uint64_t, ValueMetric> unsigned_, floats_;
  std::array<std::optional<ValueMetric>, 3> special_;
  std::unordered_map<std::string, std::size_t> identities_;
  std::size_t serializations_ = 0;
  TextMetrics *text_metrics_;

  template <typename Map, typename Key>
  ValueMetric cached(Map &cache, const Key &key, const Json &value) {
    auto found = cache.find(key);
    if (found != cache.end())
      return found->second;
    auto metric = fresh(value);
    cache.emplace(key, metric);
    return metric;
  }
  ValueMetric fresh(const Json &value) {
    std::shared_ptr<const TextMetric> measured;
    std::string encoded;
    if (text_metrics_ && value.is_string())
      measured = text_metrics_->measure(value.get_ref<const std::string &>());
    else
      encoded = serialize_value(value);
    const auto &text = measured ? measured->encoded : encoded;
    ++serializations_;
    auto identity = identities_.emplace(text, identities_.size()).first->second;
    const auto length =
        measured ? measured->characters : character_length(text);
    constexpr std::array<std::size_t, 7> bounds{
        {8, 32, 128, 512, 2048, 8192, 32768}};
    std::size_t bin = 0;
    while (bin < bounds.size() && length > bounds[bin])
      ++bin;
    return {identity, length, text.size(), bin, value.is_null()};
  }
};
}
