#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace semcache {
using Id = std::int64_t;
using Ids = std::vector<Id>;
using Keys = std::set<std::string>;
using ActionPriority = std::array<double, 3>;
using ActionPriorities = std::map<Id, ActionPriority>;
const ActionPriority &action_priority(const ActionPriorities &priorities,
                                      Id oid);

struct Action {
  Id operator_id = 0;
  std::size_t domain_index = 0;
  Ids inputs;
  std::optional<Id> pair_for;
  // Preserve Python-ingress canonical values (e.g. 0.0 == 0) without
  // weakening the integer IDs used by resident generated actions.
  std::string ingress_json;
};
struct PlanNode {
  std::uint64_t handle_id = 0;
  Id node_id = 0;
  Ids children;
  std::size_t height = 0;
  std::optional<Id> operator_id;
  std::vector<std::size_t> fields;
  Keys relations;
  std::optional<Id> pending_join;
  std::vector<std::vector<std::size_t>> pair_columns;
};
using NodePtr = std::shared_ptr<PlanNode>;
struct ProblemData;
struct State {
  std::uint64_t handle_id = 0;
  std::shared_ptr<const ProblemData> owner;
  std::shared_ptr<State> parent;
  NodePtr added;
  std::vector<NodePtr>
      roots; // Original leaf/construction order, not node-id order.
  std::set<Id> done;
  std::size_t depth = 0;
  Action action;
  std::string action_json;
  std::vector<NodePtr> nodes() const;
  std::size_t node_count() const;
  NodePtr node_at(std::size_t index) const;
  Ids root_ids() const;
  Ids remaining() const;
  std::vector<std::string> history() const;
  std::string structure_key() const;
};
using StatePtr = std::shared_ptr<State>;

struct LogicalStructure {
  std::string key;
  std::set<std::string> features;
};

struct ActionBudgetExceeded : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class StateProblem {
public:
  explicit StateProblem(const nlohmann::json &config);
  bool same_owner(const StateProblem &other) const {
    return data_ == other.data_;
  }
  StatePtr initial() const;
  void update_resources(const nlohmann::json &models,
                        const nlohmann::json &constraints,
                        const nlohmann::json &order,
                        const nlohmann::json &hashes);
  bool memory_feasible(const StatePtr &state,
                       const std::optional<std::string> &model) const;
  bool complete(const StatePtr &state) const;
  std::vector<Ids> input_choices(const StatePtr &state, Id oid) const;
  // A non-null priority map restricts enumeration to the first feasible group;
  // max_actions applies only to that group, including every tied operator.
  std::vector<Action>
  actions(const StatePtr &state, bool prioritize_relational = false,
          bool fixed_layout = false,
          const ActionPriorities *highest_priorities = nullptr,
          const std::optional<std::string> &fixed_model = {}) const;
  bool feasible(const StatePtr &state, const Action &action) const;
  // Typed action is validated; parent and node ownership stay resident.
  StatePtr transition(const StatePtr &state, const Action &action) const;
  Action implementation(Id operator_id, const std::string &canonical) const;
  Action parse_native_action(const StatePtr &state,
                             const std::string &action_json) const;
  std::string action_json(const Action &action) const;
  std::optional<Id> pair_target(const StatePtr &state, Id oid,
                                const Ids &inputs) const;
  std::string state_key(const StatePtr &state) const;
  // Coarse quota groups only; never used to merge executable states.
  LogicalStructure logical_structure(const StatePtr &state) const;
  bool relational_inner_join(Id operator_id) const;
  bool semantic_filter(Id operator_id) const;
  const std::string &operator_kind(Id operator_id) const;

private:
  std::shared_ptr<ProblemData> data_;
  void check(const StatePtr &state) const;
  bool matching_implementation(const StatePtr &state, Id oid,
                               std::size_t domain_index) const;
  std::vector<Ids> native_inputs(const StatePtr &state, Id oid) const;
  std::vector<Ids> tree_inputs(const StatePtr &state, Id oid) const;
  std::vector<Ids> logical_inputs(const StatePtr &state, Id oid) const;
  const std::vector<Ids> &input_choices_ref(const StatePtr &state,
                                            Id oid) const;
  const std::vector<std::size_t> &
  output_fields(const StatePtr &state, Id oid,
                const std::vector<NodePtr> &inputs) const;
  bool preserves(const StatePtr &state, Id oid, const Keys &fields,
                 const Keys &relations, std::optional<Id> pair_for,
                 const Ids &inputs) const;
};
}
