#pragma once
#include "../statistics/statistics.hpp"
#include <deque>
#include <iterator>
#include <unordered_map>

namespace semcache::sampling {
using SharedSamples = std::shared_ptr<const Json>;
using OwnedInputs = std::vector<SharedSamples>;
// Read-only SQL/sample stages borrow rows. Owned ingress keeps its populations
// alive; direct compatibility calls borrow only for their enclosing call.
class SampleInputs {
  struct Entry {
    SharedSamples owner;
    const Json *rows;
  };
  std::vector<Entry> entries_;

public:
  SampleInputs() = default;
  SampleInputs(const Json &) = delete;
  SampleInputs(const std::vector<Json> &rows) {
    entries_.reserve(rows.size());
    for (const auto &row : rows)
      entries_.push_back({{}, &row});
  }
  SampleInputs(const OwnedInputs &rows) {
    entries_.reserve(rows.size());
    for (const auto &row : rows)
      entries_.push_back({row, row.get()});
  }
  SampleInputs(std::initializer_list<const Json *> rows) {
    entries_.reserve(rows.size());
    for (auto row : rows)
      entries_.push_back({{}, row});
  }
  const Json &operator[](std::size_t index) const {
    return *entries_.at(index).rows;
  }
  const Json &at(std::size_t index) const { return (*this)[index]; }
  const SharedSamples &owner(std::size_t index) const {
    return entries_.at(index).owner;
  }
  std::size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }
  class Iterator {
    std::vector<Entry>::const_iterator current_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Json;
    using difference_type = std::ptrdiff_t;
    using pointer = const Json *;
    using reference = const Json &;
    explicit Iterator(std::vector<Entry>::const_iterator current)
        : current_(current) {}
    reference operator*() const { return *current_->rows; }
    Iterator &operator++() {
      ++current_;
      return *this;
    }
    bool operator!=(const Iterator &other) const {
      return current_ != other.current_;
    }
    bool operator==(const Iterator &other) const { return !(*this != other); }
  };
  Iterator begin() const { return Iterator(entries_.begin()); }
  Iterator end() const { return Iterator(entries_.end()); }
};
// Native published boundaries are immutable. Weak owners reject recycled
// addresses without retaining whole row banks. Direct borrowed calls retain
// their content semantics and cannot enter the publication shortcut.
class PopulationVersions {
  struct Entry {
    std::weak_ptr<const Json> owner;
    std::uint64_t serial;
  };
  std::unordered_map<const Json *, Entry> entries_;
  std::uint64_t serial_ = 0;

public:
  std::string key(const SampleInputs &inputs) {
    std::string result;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto &owner = inputs.owner(i);
      if (!owner) {
        result += "c" + digest(inputs[i]) + '/';
        continue;
      }
      auto found = entries_.find(owner.get());
      if (found == entries_.end() || found->second.owner.expired()) {
        if (entries_.size() >= 65536) {
          for (auto it = entries_.begin(); it != entries_.end();)
            if (it->second.owner.expired())
              it = entries_.erase(it);
            else
              ++it;
        }
        found = entries_.insert_or_assign(owner.get(), Entry{owner, ++serial_})
                    .first;
      }
      result += 'v' + std::to_string(found->second.serial) + '/';
    }
    return result;
  }
};
// Exact content keys are measured once only for owned immutable populations.
// Strong references prevent address reuse, and bounded physical leaf rows avoid
// retaining unbounded factorized populations. Direct mutable calls always read.
class PopulationFingerprints {
  struct Entry {
    SharedSamples owner;
    std::string key;
    std::size_t rows;
  };
  std::unordered_map<const Json *, Entry> entries_;
  std::deque<const Json *> order_;
  std::size_t capacity_, row_budget_, rows_ = 0, measurements_ = 0, hits_ = 0;

public:
  static std::size_t materialized_rows(const Json &rows) {
    if (rows.is_object() && rows.contains("product"))
      return materialized_rows(rows["product"][0]) +
             materialized_rows(rows["product"][1]);
    return rows.size();
  }

  explicit PopulationFingerprints(std::size_t capacity = 1024,
                                  std::size_t row_budget = 65536)
      : capacity_(capacity), row_budget_(row_budget) {}
  std::string fingerprint(const SampleInputs &inputs, std::size_t side,
                          std::size_t retained_rows = 0) {
    auto &owner = inputs.owner(side);
    if (owner) {
      auto found = entries_.find(owner.get());
      if (found != entries_.end()) {
        ++hits_;
        return found->second.key;
      }
    }
    auto key = digest(inputs[side]);
    ++measurements_;
    const auto count =
        retained_rows ? retained_rows : materialized_rows(inputs[side]);
    if (owner && capacity_ && count <= row_budget_) {
      while (!order_.empty() &&
             (entries_.size() >= capacity_ || rows_ + count > row_budget_)) {
        rows_ -= entries_.at(order_.front()).rows;
        entries_.erase(order_.front());
        order_.pop_front();
      }
      entries_.emplace(owner.get(), Entry{owner, key, count});
      order_.push_back(owner.get());
      rows_ += count;
    }
    return key;
  }
  Json fingerprints(const SampleInputs &inputs) {
    Json keys = Json::array();
    for (std::size_t side = 0; side < inputs.size(); ++side)
      keys.push_back(fingerprint(inputs, side));
    return keys;
  }
  std::size_t measurements() const { return measurements_; }
  std::size_t hits() const { return hits_; }
};
}
