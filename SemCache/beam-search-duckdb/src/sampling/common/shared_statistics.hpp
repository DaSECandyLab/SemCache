#pragma once
#include "inputs.hpp"

namespace semcache::sampling {
// Published evidence and sibling states share a block until a writer changes
// it. Borrowed exports retain the block and force subsequent writes to detach.
class SharedStatistics {
public:
  SharedStatistics() : value_(empty()) {}
  explicit SharedStatistics(Json value)
      : value_(std::make_shared<Json>(std::move(value))) {}
  const Json &read() const { return *value_; }
  Json &write() {
    if (!value_.unique())
      value_ = std::make_shared<Json>(*value_);
    return *value_;
  }
  SharedSamples handle() const { return value_; }

private:
  static const std::shared_ptr<Json> &empty() {
    static const auto value = std::make_shared<Json>(Json::object());
    return value;
  }
  std::shared_ptr<Json> value_;
};
}
