#pragma once
#include <algorithm>
#include <functional>
#include <cstdint>
#include <map>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace semcache::sampling {
using Json = nlohmann::ordered_json;
std::string canonical(const Json &value);
std::string digest(const Json &value);
std::string digest_text(const std::string &value);
std::string serialize_value(const Json &value);
std::size_t character_length(const std::string &value);
Json build_summaries(const Json &fields, const Json &config);
struct NativeFieldValues {
  std::string name;
  std::size_t count;
  std::function<std::string(std::size_t)> value;
};
Json build_summaries(const std::vector<NativeFieldValues> &fields,
                     const Json &config);
double effective_samples(const std::vector<double> &weights);
Json survival_rate(const Json &alpha, const std::string &depth = "");
Json self_reuse(const Json &summary, const Json &alpha,
                const Json &requests = nullptr);
struct CrossDependence {
  bool same_field = false;
  std::shared_ptr<const Json> common;
};
Json cross_reuse(const Json &source, const Json &target,
                 const Json &source_alpha, const Json &target_alpha,
                 const CrossDependence &dependence = {},
                 const Json &requests = nullptr);
// Published owned summary handles are versions; changing survivals reevaluates
// the retained numeric terms without decoding or rehashing prefix entries.
class ReuseCache {
public:
  ReuseCache();
  ~ReuseCache();
  Json diagnostics() const;
  Json self(const std::shared_ptr<const Json> &summary, const Json &alpha,
            const Json &requests = nullptr);
  Json cross(const std::shared_ptr<const Json> &source,
             const std::shared_ptr<const Json> &target,
             const Json &source_alpha, const Json &target_alpha,
             const CrossDependence &dependence = {},
             const Json &requests = nullptr);
  Json join(const std::shared_ptr<const Json> &left,
            const std::shared_ptr<const Json> &right,
            const Json &left_total_length, const Json &left_alpha,
            const Json &right_alpha, const Json &left_requests = nullptr,
            const Json &right_requests = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
std::string prefix_namespace(const Json &op, const Json &action,
                             const std::string &system,
                             const std::string &specification);
Json prefix_length(const std::string &prefix);
// Resolve task/layout and argument slots once; borrow immutable row values.
class RequestKeyPlan {
public:
  RequestKeyPlan(const Json &op, const std::string &model, const Json &layout,
                 bool two_inputs);
  std::string operator()(const Json &left, const Json *right = nullptr) const;

  // A factorized population can supply borrowed cells without constructing
  // whole rows. All callers share the exact serialization/hash contract.
  template <typename Fetch> std::string from_values(const Fetch &fetch) const {
    Json values = Json::array();
    for (const auto &slot : slots_)
      values.push_back(serialize_value(fetch(slot.first, slot.second)));
    return from_serialized(values);
  }
  const auto &slots() const { return slots_; }
  std::string from_serialized(const Json &values) const {
    return digest_text(prefix_ + canonical(values) + "]");
  }

private:
  std::string prefix_;
  std::vector<std::pair<std::string, bool>> slots_;
};
std::string request_key(const Json &op, const std::vector<Json> &unit,
                        const std::string &model, const Json &layout);
Json propagate(const Json &op, const Json &prior,
               const std::vector<Json> &sizes);
void propagate_inplace(const Json &op, Json &statistics,
                       const std::vector<Json> &sizes);

struct RequestResult {
  bool success;
  Json value;
};
// Request spending is separate from per-parameter result memoization. A single
// physical prompt may publish several decisions; cache hits spend no request.
class RequestLedger {
public:
  explicit RequestLedger(std::size_t budget) : budget_(budget) {}
  bool charge(std::size_t requests, bool accuracy = false);
  void publish(const std::string &key, bool success, const Json &value);
  const RequestResult *find(const std::string &key) const;
  std::size_t used() const { return selection_used_ + accuracy_used_; }
  std::size_t remaining() const { return budget_ - selection_used_; }
  std::size_t selection_used() const { return selection_used_; }
  std::size_t accuracy_used() const { return accuracy_used_; }
  void allow_additional_selection(std::size_t requests) {
    if (requests > std::numeric_limits<std::size_t>::max() - budget_)
      throw std::overflow_error("Sampling request allowance overflow");
    budget_ += requests;
  }

private:
  std::size_t budget_;
  std::size_t selection_used_ = 0, accuracy_used_ = 0;
  std::map<std::string, RequestResult> values_;
};

}
