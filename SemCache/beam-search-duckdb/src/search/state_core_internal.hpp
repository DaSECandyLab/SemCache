#pragma once
#include "state_core.hpp"
#include "../sampling/common/exact_integer.hpp"
#include "nlohmann/json.hpp"
#include "resources.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace semcache {
using Json = nlohmann::json;
namespace state_detail {
extern std::atomic<std::uint64_t> next_handle;
struct LogicalNode {
  Id id;
  std::string type;
  std::string shape_label;
  std::string join_type;
  Ids children, pair_targets;
  std::vector<std::size_t> fields;
  Keys required, left_required, right_required, left_scope, right_scope;
  std::vector<Keys> pair_required;
  bool commutative = false;
};
struct DomainAction {
  Json value;
  std::string canonical;
  std::optional<std::string> model;
  bool fixed_layout = true;
  std::string implementation;
  bool can_append_inputs = false;
  std::string in_prefix, in_suffix;
  std::string pair_prefix, pair_mid, pair_suffix;
};
struct Operator {
  Id id, node;
  std::string kind;
  bool semantic, ordinary_filter;
  std::set<Id> depends;
  Ids pair_targets;
  Ids duplicates;
  Keys required, scope, left_scope, right_scope;
  std::vector<std::size_t> produced, output;
  std::vector<DomainAction> domain;
  std::unordered_map<std::string, std::size_t> domain_index;
};
struct Region {
  Id id, base = 0;
  Ids nodes, bases;
  std::set<Id> members, allowed;
};
struct Memory {
  bool loaded = false;
  std::array<std::optional<ResourceNumber>, 4> bytes;
  std::size_t id = 0;
  std::uint64_t hash = 0;
};
template <class T> T read(const Json &d, const char *name) {
  return d.at(name).get<T>();
}
template <class C> bool contains(const C &c, const typename C::key_type &v) {
  return c.find(v) != c.end();
}
template <class T> bool subset(const std::set<T> &a, const std::set<T> &b) {
  return std::includes(b.begin(), b.end(), a.begin(), a.end());
}
template <class T> bool intersects(const std::set<T> &a, const std::set<T> &b) {
  for (const auto &v : a)
    if (contains(b, v))
      return true;
  return false;
}
template <class T> void extend(std::set<T> &a, const std::set<T> &b) {
  a.insert(b.begin(), b.end());
}
std::set<Id> idset(const Ids &ids);
NodePtr root(const StatePtr &state, Id id);
// Search states are immutable construction prefixes. Keep only one parent's
// structural results per thread, so sibling layouts share legality work without
// retaining search states or mixing concurrent search sessions. The global
// handle is never reused; resource metadata is deliberately excluded.
struct ParentStructureMemo {
  const ProblemData *owner = nullptr;
  std::uint64_t handle = 0;
  std::map<Id, std::vector<Ids>> choices;
  std::map<Ids, std::vector<NodePtr>> selected;
  std::map<std::pair<Id, Ids>, std::vector<std::size_t>> fields;
  std::map<std::pair<Id, std::optional<Id>>, std::map<Id, Id>>
      pending_components;
  std::map<Id, std::optional<std::string>> duplicate_implementations;
  std::unordered_map<Id, NodePtr> nodes;
};
ParentStructureMemo &parent_structure(const StatePtr &state);
const std::vector<NodePtr> &selected(const StatePtr &state, const Ids &ids);
std::vector<Ids> permutations(const std::vector<NodePtr> &nodes, std::size_t arity);
std::optional<ResourceNumber> resource_number(const Json &value);
std::vector<Id> chosen_nodes(const StatePtr &state, const std::set<Id> &members);
} // namespace state_detail
using namespace state_detail;

struct ProblemData {
  bool native;
  std::size_t max_actions;
  std::map<Id, LogicalNode> logical;
  std::map<Id, Operator> ops;
  std::vector<Id> op_order;
  std::vector<Ids> groups;
  std::vector<Region> trees, predicates;
  std::map<Id, std::size_t> tree_nodes, predicate_nodes;
  std::vector<std::string> field_keys;
  std::vector<NodePtr> leaves;
  std::map<std::string, Memory> models;
  std::optional<ResourceNumber> gpu_limit, cpu_limit;
  std::vector<std::string> model_order;
  ResourceSet loaded_models;
  Keys output_required;
  Keys keys(const std::vector<std::size_t> &fields) const {
    Keys result;
    for (auto f : fields)
      result.insert(field_keys.at(f));
    return result;
  }
  Keys keys(const std::vector<NodePtr> &nodes) const {
    Keys result;
    for (const auto &n : nodes)
      extend(result, keys(n->fields));
    return result;
  }
  bool roles(const LogicalNode &node, const std::vector<NodePtr> &nodes,
             const StatePtr &state = {}) const {
    if (!node.pair_required.empty() && nodes.size() == 1) {
      if (node.pair_required.size() != 2 || !state)
        return false;
      auto &built = parent_structure(state).nodes;
      if (built.empty())
        for (const auto &n : state->nodes())
          built.emplace(n->node_id, n);
      auto children = [&](Id id) -> const Ids & {
        const auto found = built.find(id);
        return found != built.end() && !found->second->children.empty()
                   ? found->second->children
                   : logical.at(id).children;
      };
      auto fields = [&](Id id) {
        const auto found = built.find(id);
        return keys(found != built.end() ? found->second->fields
                                         : logical.at(id).fields);
      };
      auto id = nodes[0]->node_id;
      bool candidate_pipeline = false;
      for (std::size_t depth = 0; depth < logical.size(); ++depth) {
        const auto &source = logical.at(id);
        const auto &inputs = children(id);
        if ((source.type == "FILTER" || source.type == "SEMANTIC_FILTER" ||
             source.type == "MODEL_BINDING_ORDER") &&
            inputs.size() == 1) {
          candidate_pipeline |= source.type != "FILTER";
          id = inputs[0];
          continue;
        }
        if ((source.type == "COMPARISON_JOIN" || source.type == "ANY_JOIN" ||
             (source.type == "CROSS_PRODUCT" && candidate_pipeline)) &&
            inputs.size() == 2) {
          auto available = fields(inputs[0]);
          extend(available, fields(inputs[1]));
          return subset(node.pair_required[0], available) &&
                 subset(node.pair_required[1], available);
        }
        return (source.type == "COMPARISON_JOIN" || source.type == "ANY_JOIN" ||
                source.type == "CROSS_PRODUCT") &&
               inputs.size() == 2 &&
               subset(node.pair_required[0], fields(inputs[0])) &&
               subset(node.pair_required[1], fields(inputs[1]));
      }
      return false;
    }
    if (nodes.size() != 2)
      return true;
    const auto left = keys(nodes[0]->fields), right = keys(nodes[1]->fields);
    return (subset(node.left_required, left) &&
            subset(node.right_required, right)) ||
           (node.commutative && subset(node.left_required, right) &&
            subset(node.right_required, left));
  }
};
} // namespace semcache
