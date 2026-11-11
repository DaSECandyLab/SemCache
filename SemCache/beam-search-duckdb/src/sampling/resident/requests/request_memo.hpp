#pragma once
#include "../../statistics/value_metrics.hpp"

namespace semcache::sampling::resident_detail {
// One context owns the plan and exact scalar dictionary. The key remains the
// original content hash; repeated task arguments avoid encoding/hash work.
class RequestMemo {
  RequestKeyPlan plan_;
  ValueMetrics metrics_;
  std::unordered_map<std::size_t, std::string> cells_;
  std::map<std::vector<std::size_t>, std::string> keys_;
  std::size_t hits_ = 0;

public:
  explicit RequestMemo(RequestKeyPlan plan) : plan_(std::move(plan)) {}
  template <class Fetch> std::string key(const Fetch &fetch) {
    std::vector<std::size_t> identity;
    std::vector<const std::string *> cells;
    for (const auto &slot : plan_.slots()) {
      const auto &value = fetch(slot.first, slot.second);
      const auto id = metrics_.measure(value).identity;
      auto found = cells_.find(id);
      if (found == cells_.end())
        found = cells_.emplace(id, serialize_value(value)).first;
      identity.push_back(id);
      cells.push_back(&found->second);
    }
    auto found = keys_.find(identity);
    if (found != keys_.end()) {
      ++hits_;
      return found->second;
    }
    Json values = Json::array();
    for (const auto *cell : cells)
      values.push_back(*cell);
    auto key = plan_.from_serialized(values);
    keys_.emplace(std::move(identity), key);
    return key;
  }
  std::size_t hits() const { return hits_; }
};
}
