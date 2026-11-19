#pragma once

#include "environment.hpp"
#include "../common/accuracy.hpp"
#include "../common/exact_integer.hpp"
#include "../common/expression_bindings.hpp"
#include "../common/prefix_selectivity.hpp"
#include "../statistics/statistics.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace semcache::sampling {
namespace profile_detail {

using Value = std::shared_ptr<const Json>;
using FieldMap = std::map<std::string, Value>;

struct Record {
  FieldMap fields;
  mutable Value scalar_statistics;
  mutable Value feature_statistics;
  bool contains(const std::string &key) const { return fields.count(key); }
  const Json &get(const std::string &key) const;
  Value reference(const std::string &key) const;
  void set(const std::string &key, Json value);
  void share(const std::string &key, Value value);
  void merge(const Record &other);
  void erase(const std::string &key);
  Json json(bool raw = false) const;
  Value scalar() const;
  Json feature_json() const;
  Value features() const;
};

using Evidence = std::shared_ptr<const Record>;
using Bank = std::map<std::string, Evidence>;

Evidence empty_record();
Evidence find(const Bank &bank, const std::string &key);
Record import_record(Value owner, const Json &raw);
const Json &value_reference(const Json &object, const std::string &key, const Json &fallback);
const Json &input_schema(const Json &op);
Bank import_bank(Value owner, const Json &raw);
bool number(const Json &value, bool nonnegative = false);
bool flag(const Record &record, const std::string &name);
std::string kind(const Json &op);
bool semantic(const Json &op);
std::string field_key(const Json &field);
Json width(const NodePtr &node, const Json &registry, const Json &summaries);
bool compatible(const Json &a, const Json &b);

struct Choice {
  Id operator_id, node_id;
  Json action;
  std::string key;
  std::string implementation_key;
  std::string layout_key;
};

struct NodeIndex {
  std::vector<Id> ids;
  std::unordered_map<Id, std::size_t> slots;
  explicit NodeIndex(const std::set<Id> &values) : ids(values.begin(), values.end()) {
    for (std::size_t i = 0; i < ids.size(); ++i)
      slots.emplace(ids[i], i);
  }
};

struct StateData {
  StatePtr state;
  std::vector<Evidence> node_records;
  std::vector<std::shared_ptr<const Choice>> actions;
  std::vector<NodePtr> nodes;
  std::vector<std::size_t> node_slots;
  std::vector<NodePtr> node_by_slot;
  std::map<Id, double> runtime_selectivities;
};

struct SurvivalMemoKey {
  std::vector<std::uint64_t> inputs;
  std::string name;
  std::string expression;
  std::string expression_type;
  std::uintptr_t summary_ptr = 0;
  std::uintptr_t single_ptr = 0;
  bool operator==(const SurvivalMemoKey &other) const {
    return summary_ptr == other.summary_ptr &&
           single_ptr == other.single_ptr &&
           inputs == other.inputs &&
           name == other.name &&
           expression == other.expression &&
           expression_type == other.expression_type;
  }
};

struct SurvivalMemoKeyHash {
  std::size_t operator()(const SurvivalMemoKey &key) const noexcept {
    std::size_t seed = std::hash<std::uintptr_t>{}(key.summary_ptr);
    auto mix = [&](std::size_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    mix(std::hash<std::uintptr_t>{}(key.single_ptr));
    mix(key.inputs.size());
    for (const auto id : key.inputs)
      mix(std::hash<std::uint64_t>{}(id));
    mix(std::hash<std::string>{}(key.name));
    mix(std::hash<std::string>{}(key.expression));
    mix(std::hash<std::string>{}(key.expression_type));
    return seed;
  }
};

} // namespace profile_detail

struct ProfileEnvironment::Impl {
  profile_detail::Value imported;
  const Json &profile_, &query, &fields;
  Json history_;
  profile_detail::Value history_owner;
  std::shared_ptr<const void> identity_token = std::make_shared<const int>(0);
  mutable std::recursive_mutex api_mutex;
  Json remaining_selectivity_ = Json::object();
  std::map<Id, Json> ops, logical;
  std::shared_ptr<const profile_detail::NodeIndex> node_index;
  profile_detail::Bank initial, node_priors, operator_priors;
  mutable std::map<Id, profile_detail::Evidence> scan_children;
  std::unordered_map<
      Id, std::unordered_map<std::string, std::shared_ptr<const profile_detail::Choice>>>
      choice_cache;
  std::size_t choice_cache_size = 0;
  bool sample_depth_selectivity = true;
  bool runtime_observed = false;
  std::vector<profile_detail::Record> sources;
  struct SourceGroup {
    std::unordered_map<std::string, std::vector<const profile_detail::Record *>> prefixes;
    bool unknown_prefix = false;
  };
  std::unordered_map<std::string, SourceGroup> source_index;
  bool source_index_compatible = true;
  std::unordered_map<std::uint64_t, profile_detail::StateData> states;
  ReuseCache reuse;
  std::map<std::pair<Id, std::string>, profile_detail::Value> prefixes;
  std::map<std::string, profile_detail::Value> prefix_lengths;
  struct SurvivalMemo {
    std::vector<std::uint64_t> inputs;
    profile_detail::Value summary, value;
  };
  mutable std::unordered_map<profile_detail::SurvivalMemoKey, SurvivalMemo,
                             profile_detail::SurvivalMemoKeyHash>
      survival_memo;
  std::map<std::string, profile_detail::Value> prefix_cooccurrences;
  std::map<std::string, std::pair<double, std::size_t>> metrics;
  std::vector<std::unique_ptr<Impl>> observation_workers;

  explicit Impl(const Json &config);
  explicit Impl(const Impl &source);
  void index_sources();
  std::set<Id> descendants(const std::vector<NodePtr> &nodes, Id target) const;
  const NodePtr &indexed_node(const profile_detail::StateData &data, Id id) const;
  const profile_detail::Evidence &node_record(const profile_detail::StateData &data, Id id) const;
  const profile_detail::Record &child(const profile_detail::StateData &data, const NodePtr &node) const;
  CrossDependence cache_dependence(const profile_detail::Record &source,
                                   const profile_detail::Record &target) const;
  void cache_edges(profile_detail::Record &result, const profile_detail::StateData &context,
                   const std::vector<NodePtr> &nodes, const NodePtr &node,
                   const profile_detail::Choice &chosen);
  bool has_field(const NodePtr &node, const std::string &name) const;
  Json predecessor_survival(const profile_detail::StateData &context,
                            const profile_detail::StateData &structure, const NodePtr &node,
                            const std::string &name, const std::string &depth,
                            const std::string &expression = "",
                            const std::string &expression_type = "") const;
  profile_detail::Value incoming_survival(const profile_detail::StateData &context,
                                          const profile_detail::StateData &structure,
                                          const NodePtr &node, const std::string &name,
                                          profile_detail::Value summary_owner,
                                          profile_detail::Value single_owner = {}) const;
  void estimate_survivals(profile_detail::Record &result, const profile_detail::StateData &context,
                          const profile_detail::StateData &structure, const NodePtr &node,
                          const profile_detail::Choice &chosen) const;
  profile_detail::Record estimate(const profile_detail::StateData &context,
                                  const profile_detail::StateData &structure,
                                  const NodePtr &node, const profile_detail::Choice &chosen);
  std::shared_ptr<const profile_detail::Choice> choice(const StatePtr &state);
  profile_detail::StateData build_state(const StatePtr &state, const profile_detail::StateData *parent);
  void observe(const StatePtr &state);
  Json statistics(const StatePtr &state, bool raw) const;
};

} // namespace semcache::sampling
