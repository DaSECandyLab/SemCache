#pragma once
#include "../common/inputs.hpp"

namespace semcache::sampling {
// SQL observes ordered IDs, values and field order, never sampling evidence.
// Only immutable published owners can skip measuring content. Direct borrowed
// populations are measured every time, including after in-place mutation.
class SQLPayloads {
  struct Entry {
    SharedSamples owner;
    std::string key;
    std::size_t rows;
  };
  std::unordered_map<const Json *, Entry> entries_;
  std::deque<const Json *> order_;
  std::size_t rows_ = 0, measurements_ = 0, hits_ = 0;

  static std::string ordered_value(const Json &value) {
    if (!value.is_structured())
      return canonical(value);
    std::string result = value.is_array() ? "[" : "{";
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (result.size() > 1)
        result += ',';
      if (value.is_object())
        result += Json(it.key()).dump() + ':';
      result += ordered_value(it.value());
    }
    return result + (value.is_array() ? "]" : "}");
  }

public:
  std::string fingerprint(const Json &rows, const SharedSamples &owner = {},
                          std::size_t retained_rows = 0) {
    if (owner) {
      auto found = entries_.find(owner.get());
      if (found != entries_.end()) {
        ++hits_;
        return found->second.key;
      }
    }
    std::string content;
    if (rows.is_array()) {
      content = "[";
      for (const auto &row : rows) {
        if (content.size() > 1)
          content += ',';
        content += '[' + ordered_value(row.at("id")) + ',' +
                   ordered_value(row.at("values")) + ']';
      }
      content += ']';
    } else {
      const auto &parts = rows.at("product");
      if (!parts.is_array() || parts.size() != 2)
        throw std::invalid_argument("Invalid binary sample product");
      auto count = retained_rows
                       ? retained_rows
                       : PopulationFingerprints::materialized_rows(rows);
      content =
          "product/" +
          fingerprint(parts[0],
                      owner ? SharedSamples(owner, &parts[0]) : SharedSamples{},
                      count) +
          '/' +
          fingerprint(parts[1],
                      owner ? SharedSamples(owner, &parts[1]) : SharedSamples{},
                      count);
    }
    auto key = digest_text(content);
    ++measurements_;
    auto count = retained_rows
                     ? retained_rows
                     : PopulationFingerprints::materialized_rows(rows);
    if (owner && count <= 65536) {
      while (!order_.empty() &&
             (entries_.size() >= 1024 || rows_ + count > 65536)) {
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
      keys.push_back(fingerprint(inputs[side], inputs.owner(side)));
    return keys;
  }
  std::size_t measurements() const { return measurements_; }
  std::size_t hits() const { return hits_; }
  void clear() {
    entries_.clear();
    order_.clear();
    rows_ = 0;
  }
};
}
