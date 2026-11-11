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
namespace {
std::atomic<std::uint64_t> next_handle{1};
struct LogicalNode {
  Id id;
  std::string type;
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
std::set<Id> idset(const Ids &ids) { return {ids.begin(), ids.end()}; }
NodePtr root(const StatePtr &state, Id id) {
  for (const auto &n : state->roots)
    if (n->node_id == id)
      return n;
  return nullptr;
}
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
ParentStructureMemo &parent_structure(const StatePtr &state) {
  thread_local ParentStructureMemo memo;
  if (memo.owner != state->owner.get() || memo.handle != state->handle_id) {
    memo.choices.clear();
    memo.nodes.clear();
    memo.selected.clear();
    memo.fields.clear();
    memo.pending_components.clear();
    memo.duplicate_implementations.clear();
    memo.owner = state->owner.get();
    memo.handle = state->handle_id;
  }
  return memo;
}
const std::vector<NodePtr> &selected(const StatePtr &state, const Ids &ids) {
  auto &cache = parent_structure(state).selected;
  const auto previous = cache.find(ids);
  if (previous != cache.end())
    return previous->second;
  std::vector<NodePtr> result;
  result.reserve(ids.size());
  for (auto id : ids) {
    auto n = root(state, id);
    if (!n)
      throw std::invalid_argument("Illegal state transition");
    result.push_back(n);
  }
  return cache.emplace(ids, std::move(result)).first->second;
}
std::vector<Ids> permutations(const std::vector<NodePtr> &nodes,
                              std::size_t arity) {
  std::vector<Ids> result;
  Ids current;
  std::vector<bool> used(nodes.size(), false);
  std::function<void()> visit = [&]() {
    if (current.size() == arity) {
      result.push_back(current);
      return;
    }
    for (std::size_t i = 0; i < nodes.size(); ++i)
      if (!used[i]) {
        used[i] = true;
        current.push_back(nodes[i]->node_id);
        visit();
        current.pop_back();
        used[i] = false;
      }
  };
  visit();
  return result;
}
std::optional<ResourceNumber> resource_number(const Json &value) {
  ResourceNumber result;
  if (value.is_number_integer() || sampling::exact_integer(value)) {
    result.integer = ResourceNumber::Integer(sampling::exact_integer(value)
                                                 ? sampling::integer_text(value)
                                                 : value.dump());
    if (result.integer < 0)
      return {};
  } else if (value.is_number_float()) {
    result.floating = true;
    result.real = value.get<double>();
    if (!std::isfinite(result.real) || result.real < 0)
      return {};
  } else
    return {};
  return result;
}
std::vector<Id> chosen_nodes(const StatePtr &state,
                             const std::set<Id> &members) {
  std::vector<Id> result;
  for (auto current = state; current->parent; current = current->parent)
    if (contains(members, current->added->node_id))
      result.push_back(current->added->node_id);
  std::reverse(result.begin(), result.end());
  return result;
}
}

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

std::vector<NodePtr> State::nodes() const {
  std::vector<NodePtr> result = owner->leaves;
  std::vector<NodePtr> additions;
  for (auto s = this; s->parent; s = s->parent.get())
    additions.push_back(s->added);
  result.insert(result.end(), additions.rbegin(), additions.rend());
  return result;
}
std::size_t State::node_count() const { return owner->leaves.size() + depth; }
NodePtr State::node_at(std::size_t index) const {
  if (index < owner->leaves.size())
    return owner->leaves[index];
  const auto offset = index - owner->leaves.size();
  if (offset >= depth)
    throw std::out_of_range("State node index out of range");
  auto current = this;
  while (current->depth > offset + 1)
    current = current->parent.get();
  return current->added;
}
Ids State::root_ids() const {
  Ids result;
  for (const auto &n : roots)
    result.push_back(n->node_id);
  if (depth)
    std::sort(result.begin(), result.end());
  return result;
}
Ids State::remaining() const {
  Ids result;
  for (const auto &entry : owner->ops)
    if (!contains(done, entry.first))
      result.push_back(entry.second.node);
  std::sort(result.begin(), result.end());
  return result;
}
std::vector<std::string> State::history() const {
  std::vector<std::string> result;
  for (auto s = this; s->parent; s = s->parent.get())
    result.push_back(s->action_json);
  std::reverse(result.begin(), result.end());
  return result;
}
std::string State::structure_key() const {
  auto all = nodes();
  std::sort(all.begin(), all.end(), [](const NodePtr &a, const NodePtr &b) {
    return a->node_id < b->node_id;
  });
  Json result = Json::array();
  for (const auto &n : all)
    result.push_back(Json::array({n->node_id, n->children}));
  return result.dump();
}

StateProblem::StateProblem(const Json &config)
    : data_(std::make_shared<ProblemData>()) {
  auto &d = *data_;
  d.native = read<bool>(config, "native");
  d.max_actions = read<std::size_t>(config, "max_actions");
  d.field_keys = read<std::vector<std::string>>(config, "field_keys");
  d.groups = read<std::vector<Ids>>(config, "groups");
  d.output_required = read<Keys>(config, "output_required");
  for (auto item : config["nodes"]) {
    const auto &row = item;
    LogicalNode n;
    n.id = read<Id>(row, "node_id");
    n.type = read<std::string>(row, "type");
    n.join_type = row.value("join_type", std::string());
    n.children = read<Ids>(row, "children");
    n.fields = read<std::vector<std::size_t>>(row, "fields");
    n.required = read<Keys>(row, "required");
    n.left_required = read<Keys>(row, "left_required");
    n.right_required = read<Keys>(row, "right_required");
    n.pair_required = read<std::vector<Keys>>(row, "pair_required");
    n.left_scope = read<Keys>(row, "left_scope");
    n.right_scope = read<Keys>(row, "right_scope");
    n.commutative = read<bool>(row, "commutative");
    n.pair_targets = read<Ids>(row, "pair_targets");
    d.logical.emplace(n.id, std::move(n));
  }
  for (auto item : config["operators"]) {
    const auto &row = item;
    Operator o;
    o.id = read<Id>(row, "operator_id");
    o.node = read<Id>(row, "node_id");
    o.kind = read<std::string>(row, "kind");
    o.semantic = read<bool>(row, "semantic");
    o.duplicates = read<Ids>(row, "duplicates");
    o.ordinary_filter = read<bool>(row, "ordinary_filter");
    o.depends = read<std::set<Id>>(row, "depends_on");
    o.required = read<Keys>(row, "required");
    o.scope = read<Keys>(row, "scope");
    o.left_scope = read<Keys>(row, "left_scope");
    o.right_scope = read<Keys>(row, "right_scope");
    o.produced = read<std::vector<std::size_t>>(row, "produced");
    o.output = read<std::vector<std::size_t>>(row, "output");
    o.pair_targets = read<Ids>(row, "pair_targets");
    for (const auto &text : read<std::vector<std::string>>(row, "domain")) {
      DomainAction a{Json::parse(text), text, {}};
      if (a.value.contains("model_id")) {
        a.model = a.value["model_id"].get<std::string>();
        a.fixed_layout = row.contains("fixed_layout") &&
                         a.value.value("layout", Json()) == row["fixed_layout"];
        a.implementation = Json{{"model_id", a.value.at("model_id")},
                                {"layout", a.value.at("layout")}}
                               .dump();
      }
      o.domain_index.emplace(text, o.domain.size());
      o.domain.push_back(std::move(a));
    }
    d.op_order.push_back(o.id);
    d.ops.emplace(o.id, std::move(o));
  }
  for (auto item : config["leaves"]) {
    const auto &row = item;
    auto n = std::make_shared<PlanNode>();
    n->handle_id = next_handle++;
    n->node_id = read<Id>(row, "node_id");
    n->fields = read<std::vector<std::size_t>>(row, "fields");
    n->relations = read<Keys>(row, "relations");
    d.leaves.push_back(std::move(n));
  }
  for (const char *kind : {"tree_regions", "predicate_regions"}) {
    bool tree = std::string(kind) == "tree_regions";
    auto &regions = tree ? d.trees : d.predicates;
    auto &index = tree ? d.tree_nodes : d.predicate_nodes;
    for (auto item : config[kind]) {
      const auto &row = item;
      Region r;
      r.id = read<Id>(row, "region_id");
      r.nodes = read<Ids>(row, "nodes");
      r.members = idset(r.nodes);
      r.allowed = r.members;
      if (tree) {
        r.bases = read<Ids>(row, "base_nodes");
        extend(r.allowed, idset(r.bases));
      } else
        r.base = read<Id>(row, "base_node_id");
      for (auto id : r.nodes)
        index[id] = regions.size();
      regions.push_back(std::move(r));
    }
  }
  update_resources(config.at("models"), config.at("constraints"),
                   config.value("model_order", Json::array()),
                   config.value("model_hashes", Json::object()));
}
void StateProblem::update_resources(const Json &models, const Json &constraints,
                                    const Json &order, const Json &hashes) {
  auto &d = *data_;
  d.models.clear();
  d.model_order.clear();
  d.gpu_limit = constraints.contains("gpu_memory_bytes")
                    ? resource_number(constraints["gpu_memory_bytes"])
                    : std::nullopt;
  d.cpu_limit = constraints.contains("cpu_memory_bytes")
                    ? resource_number(constraints["cpu_memory_bytes"])
                    : std::nullopt;
  d.loaded_models = ResourceSet();
  static const std::array<const char *, 4> names{
      "gpu_weight_bytes", "gpu_kv_reserve_bytes", "cpu_weight_bytes",
      "cpu_kv_reserve_bytes"};
  auto model_order = order.get<std::vector<std::string>>();
  if (model_order.empty())
    for (auto it = models.begin(); it != models.end(); ++it)
      model_order.push_back(it.key());
  if (model_order.size() != models.size())
    throw std::invalid_argument("Model order differs from model catalog");
  std::set<std::string> seen;
  for (const auto &name : model_order) {
    if (!seen.insert(name).second || !models.contains(name))
      throw std::invalid_argument("Invalid model order");
    const auto &row = models.at(name);
    Memory m;
    m.loaded = row.value("loaded", false);
    for (std::size_t i = 0; i < names.size(); ++i)
      m.bytes[i] = row.contains(names[i]) ? resource_number(row[names[i]])
                                          : std::nullopt;
    m.id = d.model_order.size();
    m.hash = hashes.contains(name) ? hashes.at(name).get<std::uint64_t>()
                                   : std::hash<std::string>{}(name);
    if (m.loaded)
      d.loaded_models.add(m.id, m.hash);
    d.model_order.push_back(name);
    d.models.emplace(name, std::move(m));
  }
}
StatePtr StateProblem::initial() const {
  auto s = std::make_shared<State>();
  s->handle_id = next_handle++;
  s->owner = data_;
  s->roots = data_->leaves;
  return s;
}
void StateProblem::check(const StatePtr &state) const {
  if (!state || state->owner.get() != data_.get())
    throw std::invalid_argument("State belongs to another native problem");
}
bool StateProblem::complete(const StatePtr &state) const {
  check(state);
  return state->done.size() == data_->ops.size() && state->roots.size() == 1;
}
bool StateProblem::feasible(const StatePtr &state, const Action &action) const {
  const auto &base =
      data_->ops.at(action.operator_id).domain.at(action.domain_index);
  return matching_implementation(state, action.operator_id,
                                 action.domain_index) &&
         memory_feasible(state, base.model);
}
bool StateProblem::matching_implementation(const StatePtr &state, Id oid,
                                           std::size_t domain_index) const {
  check(state);
  const auto &op = data_->ops.at(oid);
  if (op.duplicates.empty())
    return true;
  auto &cache = parent_structure(state).duplicate_implementations;
  auto found = cache.find(op.duplicates.front());
  if (found == cache.end()) {
    std::optional<std::string> expected;
    for (auto current = state; current->parent; current = current->parent) {
      const auto &action = current->action;
      if (std::find(op.duplicates.begin(), op.duplicates.end(),
                    action.operator_id) != op.duplicates.end()) {
        expected = data_->ops.at(action.operator_id)
                       .domain.at(action.domain_index)
                       .implementation;
        break;
      }
    }
    found = cache.emplace(op.duplicates.front(), std::move(expected)).first;
  }
  return !found->second ||
         *found->second == op.domain.at(domain_index).implementation;
}
bool StateProblem::memory_feasible(
    const StatePtr &state, const std::optional<std::string> &model) const {
  check(state);
  const auto &d = *data_;
  if (!d.gpu_limit && !d.cpu_limit)
    return true;
  ResourceSet used;
  std::vector<std::string> previous;
  for (auto current = state; current->parent; current = current->parent) {
    const auto &a = current->action;
    const auto &name = d.ops.at(a.operator_id).domain.at(a.domain_index).model;
    if (name)
      previous.push_back(*name);
  }
  for (auto it = previous.rbegin(); it != previous.rend(); ++it) {
    auto found = d.models.find(*it);
    if (found == d.models.end())
      return false;
    used.add(found->second.id, found->second.hash);
  }
  ResourceSet candidate;
  if (model) {
    auto found = d.models.find(*model);
    if (found == d.models.end())
      return false;
    candidate.add(found->second.id, found->second.hash);
  }
  used.merge(candidate);
  used.merge(d.loaded_models);
  for (std::size_t device = 0; device < 2; ++device) {
    const auto &limit = device == 0 ? d.gpu_limit : d.cpu_limit;
    if (!limit)
      continue;
    ResourceNumber total;
    for (const auto &entry : used.entries())
      if (entry.present) {
        const auto &memory = d.models.at(d.model_order.at(entry.id));
        for (std::size_t i = device * 2; i < device * 2 + 2; ++i) {
          if (!memory.bytes[i])
            return false;
          total.add(*memory.bytes[i]);
        }
      }
    if (total.greater(*limit))
      return false;
  }
  return true;
}
std::optional<Id> StateProblem::pair_target(const StatePtr &state, Id oid,
                                            const Ids &ids) const {
  const auto &d = *data_;
  const auto &node = d.logical.at(d.ops.at(oid).node);
  if (ids.size() != 2 || node.type != "FILTER")
    return {};
  const auto &inputs = selected(state, ids);
  for (const auto &n : inputs)
    if (n->pending_join || !intersects(node.required, d.keys(n->fields)))
      return {};
  auto available = d.keys(inputs);
  for (auto target : node.pair_targets) {
    bool done = false;
    for (auto id : state->done)
      if (d.ops.at(id).node == target) {
        done = true;
        break;
      }
    if (done)
      continue;
    const auto &join = d.logical.at(target);
    if (!subset(join.required, available))
      continue;
    if (d.native ? d.roles(join, inputs)
                 : (subset(join.left_scope, inputs[0]->relations) &&
                    subset(join.right_scope, inputs[1]->relations)))
      return target;
  }
  return {};
}
bool StateProblem::preserves(const StatePtr &state, Id oid, const Keys &fields,
                             const Keys &relations, std::optional<Id> pair_for,
                             const Ids &inputs) const {
  const auto &d = *data_;
  for (auto id : d.op_order) {
    const auto &op = d.ops.at(id);
    if (id == oid || contains(state->done, id) ||
        (pair_for && op.node == *pair_for))
      continue;
    if (d.native) {
      const auto &n = d.logical.at(op.node);
      if (intersects(n.left_required, fields) &&
          intersects(n.right_required, fields))
        return false;
    } else if (op.kind.find("join") != std::string::npos) {
      if (intersects(op.left_scope, relations) &&
          intersects(op.right_scope, relations))
        return false;
      if (!op.required.empty() && subset(op.required, fields))
        return false;
    }
  }
  // Contracting inputs connected by other pending joins creates a cycle:
  // e.g. CROSS(A,C) leaves A=B and B=C competing to consume the same roots.
  // Such a prefix cannot execute each binary join exactly once. Check the
  // transitive connection, not only a directly swallowed predicate above.
  auto &cache = parent_structure(state).pending_components;
  const auto key = std::make_pair(oid, pair_for);
  auto found = cache.find(key);
  if (found == cache.end()) {
    std::map<Id, Id> components;
    std::map<Id, Keys> root_fields;
    for (const auto &n : state->roots) {
      components.emplace(n->node_id, n->node_id);
      root_fields.emplace(n->node_id, d.keys(n->fields));
    }
    auto representative = [&](Id id) {
      while (components.at(id) != id)
        id = components.at(id);
      return id;
    };
    for (auto id : d.op_order) {
      const auto &op = d.ops.at(id);
      if (id == oid || contains(state->done, id) ||
          (pair_for && op.node == *pair_for))
        continue;
      const auto &left =
          d.native ? d.logical.at(op.node).left_required : op.left_scope;
      const auto &right =
          d.native ? d.logical.at(op.node).right_required : op.right_scope;
      if (left.empty() || right.empty())
        continue;
      Ids owners[2];
      for (const auto &n : state->roots) {
        const auto &available =
            d.native ? root_fields.at(n->node_id) : n->relations;
        if (subset(left, available))
          owners[0].push_back(n->node_id);
        if (subset(right, available))
          owners[1].push_back(n->node_id);
      }
      // Unbound or ambiguous sides cannot establish a definite connection.
      if (owners[0].size() == 1 && owners[1].size() == 1) {
        const auto a = representative(owners[0][0]);
        const auto b = representative(owners[1][0]);
        components[b] = a;
      }
    }
    for (auto &entry : components)
      entry.second = representative(entry.first);
    found = cache.emplace(key, std::move(components)).first;
  }
  return inputs.size() != 2 ||
         found->second.at(inputs[0]) != found->second.at(inputs[1]);
}
const std::vector<std::size_t> &
StateProblem::output_fields(const StatePtr &state, Id oid,
                            const std::vector<NodePtr> &inputs) const {
  const auto &d = *data_;
  const auto &op = d.ops.at(oid);
  const auto &node = d.logical.at(op.node);
  if (d.native && !contains(d.tree_nodes, op.node) &&
      !contains(d.predicate_nodes, op.node))
    return node.fields;
  if (!d.native && op.kind.find("projection") != std::string::npos &&
      !op.semantic)
    return op.output;
  Ids ids;
  ids.reserve(inputs.size());
  for (const auto &input : inputs)
    ids.push_back(input->node_id);
  auto &cache = parent_structure(state).fields;
  const auto key = std::make_pair(oid, std::move(ids));
  const auto previous = cache.find(key);
  if (previous != cache.end())
    return previous->second;
  std::vector<std::size_t> fields;
  for (const auto &n : inputs)
    fields.insert(fields.end(), n->fields.begin(), n->fields.end());
  if (d.native) {
    Keys old;
    for (auto child : node.children)
      extend(old, d.keys(d.logical.at(child).fields));
    for (auto f : node.fields)
      if (!contains(old, d.field_keys.at(f)))
        fields.push_back(f);
  } else
    fields.insert(fields.end(), op.produced.begin(), op.produced.end());
  return cache.emplace(key, std::move(fields)).first->second;
}
std::vector<Ids> StateProblem::tree_inputs(const StatePtr &state,
                                           Id oid) const {
  const auto &d = *data_;
  const auto &op = d.ops.at(oid);
  const auto &node = d.logical.at(op.node);
  const auto &region = d.trees.at(d.tree_nodes.at(op.node));
  std::vector<NodePtr> roots;
  for (const auto &n : state->roots)
    if (contains(region.allowed, n->node_id))
      roots.push_back(n);
  auto choices = permutations(roots, node.children.size());
  if (node.type == "FILTER" && !node.pair_targets.empty()) {
    auto extra = permutations(roots, 2);
    choices.insert(choices.end(), extra.begin(), extra.end());
  } else if (node.type == "SEMANTIC_JOIN") {
    for (const auto &n : roots)
      if (n->pending_join == std::optional<Id>(op.node))
        choices.push_back({n->node_id});
  }
  auto remaining = region.members;
  remaining.erase(op.node);
  for (auto id : state->done)
    remaining.erase(d.ops.at(id).node);
  auto expected = d.keys(d.logical.at(region.id).fields);
  std::vector<Ids> result;
  for (const auto &ids : choices) {
    const auto &inputs = selected(state, ids);
    Ids pending;
    for (const auto &n : inputs)
      if (n->pending_join)
        pending.push_back(*n->pending_join);
    if (!pending.empty() &&
        !(ids.size() == 1 &&
          (node.type == "FILTER" || node.type == "SEMANTIC_FILTER" ||
           pending == Ids{op.node})))
      continue;
    auto target = ids.size() != node.children.size()
                      ? pair_target(state, oid, ids)
                      : std::optional<Id>{};
    if (ids.size() != node.children.size() && !target && pending.empty())
      continue;
    if (inputs.size() == 2 &&
        intersects(inputs[0]->relations, inputs[1]->relations) &&
        intersects(d.keys(inputs[0]->fields), d.keys(inputs[1]->fields)))
      continue;
    auto available = d.keys(inputs);
    if (!subset(op.required, available) || !d.roles(node, inputs, state))
      continue;
    Keys relations;
    for (const auto &n : inputs)
      extend(relations, n->relations);
    if (ids.size() == 2 &&
        !preserves(state, oid, available, relations, target, ids))
      continue;
    if (remaining.empty()) {
      if (d.keys(output_fields(state, oid, inputs)) != expected)
        continue;
      std::set<Id> all;
      for (const auto &n : roots)
        all.insert(n->node_id);
      if (all != idset(ids))
        continue;
    }
    result.push_back(ids);
  }
  return result;
}
std::vector<Ids> StateProblem::logical_inputs(const StatePtr &state,
                                              Id oid) const {
  const auto &d = *data_;
  const auto &op = d.ops.at(oid);
  bool join = op.kind.find("join") != std::string::npos;
  auto choices = permutations(state->roots, join ? 2 : 1);
  if (op.kind == "filter" && !op.pair_targets.empty()) {
    auto extra = permutations(state->roots, 2);
    choices.insert(choices.end(), extra.begin(), extra.end());
  } else if (join && op.semantic) {
    for (const auto &n : state->roots)
      if (n->pending_join == std::optional<Id>(op.node))
        choices.push_back({n->node_id});
  }
  std::vector<Ids> result;
  for (const auto &ids : choices) {
    const auto &inputs = selected(state, ids);
    Ids pending;
    for (const auto &n : inputs)
      if (n->pending_join)
        pending.push_back(*n->pending_join);
    if (!pending.empty() &&
        !(ids.size() == 1 &&
          (op.kind == "filter" || op.kind == "semantic_filter" ||
           pending == Ids{op.node})))
      continue;
    auto target = ids.size() == 2 && !join ? pair_target(state, oid, ids)
                                           : std::optional<Id>{};
    if (ids.size() == 2 && !join && !target)
      continue;
    const auto available = d.keys(inputs);
    Keys relations;
    for (const auto &n : inputs)
      extend(relations, n->relations);
    if (!subset(op.required, available) || !subset(op.scope, relations))
      continue;
    if (ids.size() == 2) {
      if (intersects(inputs[0]->relations, inputs[1]->relations))
        continue;
      if (!subset(op.left_scope, inputs[0]->relations) ||
          !subset(op.right_scope, inputs[1]->relations))
        continue;
      if (!op.required.empty() &&
          (!intersects(op.required, d.keys(inputs[0]->fields)) ||
           !intersects(op.required, d.keys(inputs[1]->fields))))
        continue;
      if (!preserves(state, oid, available, relations, target, ids))
        continue;
    }
    if (op.kind.find("projection") != std::string::npos) {
      auto available_output = d.keys(output_fields(state, oid, inputs));
      for (const auto &n : state->roots)
        if (std::find(ids.begin(), ids.end(), n->node_id) == ids.end())
          extend(available_output, d.keys(n->fields));
      Keys needed = d.output_required;
      for (const auto &item : d.ops)
        if (item.first != oid && !contains(state->done, item.first)) {
          const auto &other = item.second;
          extend(needed, other.required);
          if (other.semantic)
            extend(available_output, d.keys(other.produced));
          else if (other.kind.find("projection") != std::string::npos)
            extend(available_output, d.keys(other.output));
        }
      if (!subset(needed, available_output))
        continue;
    }
    result.push_back(ids);
  }
  return result;
}
std::vector<Ids> StateProblem::native_inputs(const StatePtr &state,
                                             Id oid) const {
  const auto &d = *data_;
  const auto &op = d.ops.at(oid);
  if (contains(d.tree_nodes, op.node))
    return tree_inputs(state, oid);
  auto roots_ready = [&](const Ids &ids) {
    for (auto id : ids)
      if (!root(state, id))
        return false;
    return true;
  };
  auto first_group = [&]() {
    std::set<Id> first;
    for (const auto &g : d.groups) {
      for (auto id : g)
        if (!contains(state->done, id))
          first.insert(id);
      if (!first.empty())
        break;
    }
    return first;
  };
  auto pred = d.predicate_nodes.find(op.node);
  if (pred != d.predicate_nodes.end()) {
    if (op.semantic && d.trees.empty() && !contains(first_group(), oid))
      return {};
    const auto &region = d.predicates[pred->second];
    auto chosen = chosen_nodes(state, region.members);
    Id current = chosen.empty() ? region.base : chosen.back();
    auto input = root(state, current);
    return input && subset(op.required, d.keys(input->fields))
               ? std::vector<Ids>{{current}}
               : std::vector<Ids>{};
  }
  auto inputs = d.logical.at(op.node).children;
  if (op.semantic && d.trees.empty()) {
    for (const auto &group : d.groups) {
      auto remaining = idset(group);
      for (auto done : state->done)
        remaining.erase(done);
      if (remaining.empty())
        continue;
      if (!contains(remaining, oid))
        return {};
      std::set<Id> members;
      for (auto id : group)
        members.insert(d.ops.at(id).node);
      Id bottom = 0;
      bool found = false;
      for (auto n : members)
        if (!intersects(idset(d.logical.at(n).children), members)) {
          bottom = n;
          found = true;
          break;
        }
      if (!found)
        throw std::invalid_argument("Invalid reorder region");
      auto chosen = chosen_nodes(state, members);
      inputs =
          chosen.empty() ? d.logical.at(bottom).children : Ids{chosen.back()};
      break;
    }
  }
  for (const auto &region : d.trees) {
    auto chosen = chosen_nodes(state, region.members);
    if (std::find(inputs.begin(), inputs.end(), region.id) == inputs.end())
      continue;
    if (chosen.size() != region.nodes.size())
      return {};
    for (auto &id : inputs)
      if (id == region.id)
        id = chosen.back();
  }
  for (const auto &region : d.predicates) {
    auto chosen = chosen_nodes(state, region.members);
    if (std::find(inputs.begin(), inputs.end(), region.id) != inputs.end() &&
        chosen.size() != region.nodes.size())
      return {};
    if (chosen.size() == region.nodes.size())
      for (auto &id : inputs)
        if (id == region.id)
          id = chosen.back();
  }
  // Both fixed and semantic consumers reference the original top of preceding
  // groups. Resolve it to the actual last node after a legal group reorder.
  for (const auto &group : d.groups) {
    bool predicate = false;
    std::set<Id> members;
    for (auto id : group) {
      Id n = d.ops.at(id).node;
      members.insert(n);
      if (contains(d.predicate_nodes, n))
        predicate = true;
    }
    if (predicate)
      continue;
    auto tops = members;
    for (auto n : members)
      for (auto c : d.logical.at(n).children)
        tops.erase(c);
    // An external consumer cannot use the nominal top while other members
    // still need its output. The top may execute first after a group reorder.
    if (!contains(idset(group), oid) && intersects(idset(inputs), tops) &&
        !subset(idset(group), state->done))
      return {};
    auto chosen = chosen_nodes(state, members);
    if (subset(idset(group), state->done) && !chosen.empty())
      for (auto &id : inputs)
        if (contains(tops, id))
          id = chosen.back();
  }
  return roots_ready(inputs) ? std::vector<Ids>{inputs} : std::vector<Ids>{};
}
const std::vector<Ids> &StateProblem::input_choices_ref(const StatePtr &state,
                                                        Id oid) const {
  check(state);
  const auto &op = data_->ops.at(oid);
  auto &cache = parent_structure(state).choices;
  const auto previous = cache.find(oid);
  if (previous != cache.end())
    return previous->second;
  std::vector<Ids> result;
  if (!contains(state->done, oid) && subset(op.depends, state->done))
    result =
        data_->native ? native_inputs(state, oid) : logical_inputs(state, oid);
  return cache.emplace(oid, std::move(result)).first->second;
}
std::vector<Ids> StateProblem::input_choices(const StatePtr &state,
                                             Id oid) const {
  return input_choices_ref(state, oid);
}
const ActionPriority &action_priority(const ActionPriorities &priorities,
                                      Id oid) {
  static const ActionPriority fallback{5., 1., 0.};
  const auto found = priorities.find(oid);
  return found == priorities.end() ? fallback : found->second;
}
std::vector<Action>
StateProblem::actions(const StatePtr &state, bool prioritize_relational,
                      bool fixed_layout,
                      const ActionPriorities *highest_priorities,
                      const std::optional<std::string> &fixed_model) const {
  check(state);
  std::vector<Action> result;
  std::vector<const Operator *> operators;
  for (const auto &item : data_->ops)
    if (!contains(state->done, item.first) &&
        subset(item.second.depends, state->done))
      operators.push_back(&item.second);
  if (highest_priorities) {
    for (const auto &entry : *highest_priorities)
      for (auto value : entry.second)
        if (!std::isfinite(value))
          throw std::invalid_argument("Action priorities must be finite");
  }
  struct Branch {
    const Operator *op;
    bool partitioned;
    std::optional<Id> pair_for;
    ActionPriority priority;
  };
  std::vector<Branch> branches;
  std::map<Id, Id> node_operators;
  for (const auto &entry : data_->ops)
    node_operators.emplace(entry.second.node, entry.first);
  for (const auto *op : operators) {
    const auto priority = highest_priorities
                              ? action_priority(*highest_priorities, op->id)
                              : ActionPriority{};
    const bool partitioned =
        highest_priorities && op->ordinary_filter && !op->pair_targets.empty();
    branches.push_back({op, partitioned, {}, priority});
    if (partitioned) {
      std::set<Id> seen_targets;
      for (const auto target : op->pair_targets) {
        if (!seen_targets.insert(target).second)
          continue;
        const auto found = node_operators.find(target);
        if (found == node_operators.end())
          throw std::invalid_argument(
              "Candidate-pair target has no semantic operator");
        // Binary candidate filtering is still a relational FILTER. Input
        // legality governs readiness; the target join does not change its tier.
        branches.push_back({op, true, target, priority});
      }
    }
  }
  if (highest_priorities)
    std::stable_sort(branches.begin(), branches.end(),
                     [](const Branch &a, const Branch &b) {
                       return a.priority < b.priority;
                     });
  struct InputChoice {
    Ids inputs;
    std::optional<Id> pair_for;
  };
  std::map<Id, std::vector<InputChoice>> input_cache;
  std::map<Id, std::vector<std::size_t>> domain_cache;
  // A model is shared by many field/layout variants. Its memory feasibility
  // depends on the parent state, so evaluate it once for this action set.
  std::map<std::optional<std::string>, bool> feasible_models;
  // Probe relations first without enumerating semantic implementations. Only
  // when no feasible relational action exists do we generate semantic actions.
  for (bool semantic : {false, true}) {
    std::optional<ActionPriority> best;
    for (const auto &branch : branches) {
      const auto *op = branch.op;
      if (prioritize_relational && op->semantic != semantic)
        continue;
      // Stop before enumerating lower groups: their actions must neither
      // consume max_actions nor allocate input/layout combinations.
      if (highest_priorities && best && branch.priority != *best)
        break;
      auto cached_inputs = input_cache.find(op->id);
      if (cached_inputs == input_cache.end()) {
        std::vector<InputChoice> choices;
        const auto &node = data_->logical.at(op->node);
        const bool logical_join =
            !op->semantic &&
            (data_->native ? node.commutative : op->kind == "join");
        std::map<std::pair<Id, Id>, std::size_t> join_pairs;
        auto rank = [&](const Ids &ids) {
          const auto &inputs = selected(state, ids);
          const bool direct =
              data_->native ? subset(node.left_required,
                                     data_->keys(inputs[0]->fields)) &&
                                  subset(node.right_required,
                                         data_->keys(inputs[1]->fields))
                            : subset(op->left_scope, inputs[0]->relations) &&
                                  subset(op->right_scope, inputs[1]->relations);
          return std::make_tuple(!direct, ids[0], ids[1]);
        };
        // Deduplicate new logical candidates, retaining both legal directions
        // for transition validation and replay of previously saved plans.
        for (const auto &ids : input_choices_ref(state, op->id)) {
          if (logical_join && ids.size() == 2) {
            const auto key = std::make_pair(std::min(ids[0], ids[1]),
                                            std::max(ids[0], ids[1]));
            const auto found = join_pairs.find(key);
            if (found != join_pairs.end()) {
              auto &previous = choices[found->second];
              if (rank(ids) < rank(previous.inputs))
                previous = {ids, pair_target(state, op->id, ids)};
              continue;
            }
            join_pairs.emplace(key, choices.size());
          }
          choices.push_back({ids, pair_target(state, op->id, ids)});
        }
        cached_inputs = input_cache.emplace(op->id, std::move(choices)).first;
      }
      std::vector<const InputChoice *> choices;
      for (const auto &choice : cached_inputs->second)
        if (!branch.partitioned || choice.pair_for == branch.pair_for)
          choices.push_back(&choice);
      if (choices.empty())
        continue;
      auto cached_domains = domain_cache.find(op->id);
      if (cached_domains == domain_cache.end()) {
        std::vector<std::size_t> domains;
        for (std::size_t i = 0; i < op->domain.size(); ++i) {
          const auto &domain = op->domain[i];
          if (fixed_layout && !domain.fixed_layout)
            continue;
          if (op->semantic && fixed_model && domain.model != fixed_model)
            continue;
          if (!matching_implementation(state, op->id, i))
            continue;
          auto cached = feasible_models.find(domain.model);
          if (cached == feasible_models.end())
            cached =
                feasible_models
                    .emplace(domain.model, memory_feasible(state, domain.model))
                    .first;
          if (cached->second)
            domains.push_back(i);
        }
        cached_domains = domain_cache.emplace(op->id, std::move(domains)).first;
      }
      const auto &feasible_domains = cached_domains->second;
      if (feasible_domains.empty())
        continue;
      if (highest_priorities)
        best = branch.priority;
      if (choices.size() >
          (data_->max_actions - result.size()) / feasible_domains.size())
        throw ActionBudgetExceeded("Ready action set exceeds max_actions");
      for (const auto *choice : choices) {
        for (auto index : feasible_domains)
          result.push_back(
              Action{op->id, index, choice->inputs, choice->pair_for, {}});
      }
    }
    if (!prioritize_relational || !result.empty())
      return result;
  }
  return result;
}
std::string StateProblem::action_json(const Action &action) const {
  if (!action.ingress_json.empty())
    return action.ingress_json;
  auto value =
      data_->ops.at(action.operator_id).domain.at(action.domain_index).value;
  value["inputs"] = action.inputs;
  if (action.pair_for)
    value["pair_for"] = *action.pair_for;
  return value.dump();
}
Action StateProblem::implementation(Id oid,
                                    const std::string &canonical) const {
  auto op = data_->ops.find(oid);
  if (op == data_->ops.end())
    throw std::invalid_argument("Illegal state transition");
  const auto found = op->second.domain_index.find(canonical);
  if (found == op->second.domain_index.end())
    throw std::invalid_argument("Illegal state transition");
  Action action;
  action.operator_id = oid;
  action.domain_index = found->second;
  return action;
}
StatePtr StateProblem::transition(const StatePtr &state,
                                  const Action &action) const {
  check(state);
  auto it = data_->ops.find(action.operator_id);
  if (it == data_->ops.end() || action.domain_index >= it->second.domain.size())
    throw std::invalid_argument("Illegal state transition");
  const auto &choices = input_choices_ref(state, action.operator_id);
  if (std::find(choices.begin(), choices.end(), action.inputs) ==
          choices.end() ||
      !feasible(state, action))
    throw std::invalid_argument("Illegal state transition");
  auto target = pair_target(state, action.operator_id, action.inputs);
  if (target != action.pair_for)
    throw std::invalid_argument("Illegal candidate-pair target");
  const auto &op = it->second;
  const auto &inputs = selected(state, action.inputs);
  if (inputs.empty())
    throw std::invalid_argument("Illegal state transition");
  auto node = std::make_shared<PlanNode>();
  node->handle_id = next_handle++;
  node->node_id = op.node;
  node->operator_id = op.id;
  node->children = action.inputs;
  for (const auto &n : inputs) {
    node->height = std::max(node->height, n->height + 1);
    extend(node->relations, n->relations);
  }
  node->fields = output_fields(state, op.id, inputs);
  if (target) {
    node->pending_join = target;
    for (const auto &n : inputs)
      node->pair_columns.push_back(n->fields);
  } else if (inputs.size() == 1 &&
             (op.ordinary_filter || op.kind == "semantic_filter")) {
    node->pending_join = inputs[0]->pending_join;
    node->pair_columns = inputs[0]->pair_columns;
  }
  auto child = std::make_shared<State>();
  child->handle_id = next_handle++;
  child->owner = data_;
  child->parent = state;
  child->added = node;
  child->depth = state->depth + 1;
  child->action = action;
  child->action_json = action_json(action);
  child->done = state->done;
  child->done.insert(op.id);
  for (const auto &n : state->roots)
    if (data_->logical.at(op.node).type == "CTE_SCAN" ||
        std::find(action.inputs.begin(), action.inputs.end(), n->node_id) ==
            action.inputs.end())
      child->roots.push_back(n);
  child->roots.push_back(node);
  return child;
}
Action StateProblem::parse_native_action(const StatePtr &state,
                                         const std::string &text) const {
  check(state);
  auto value = Json::parse(text);
  const auto oid = value.at("operator_id").get<Id>();
  const auto &op = data_->ops.at(oid);
  auto base = value;
  base.erase("inputs");
  base.erase("pair_for");
  auto found = op.domain_index.find(base.dump());
  if (found == op.domain_index.end())
    throw std::invalid_argument("Illegal action implementation");
  Action action;
  action.operator_id = oid;
  action.domain_index = found->second;
  action.inputs = value.at("inputs").get<Ids>();
  action.pair_for = pair_target(state, oid, action.inputs);
  if (value.contains("pair_for") &&
      value.at("pair_for").get<Id>() != action.pair_for)
    throw std::invalid_argument("Illegal candidate-pair target");
  return action;
}
std::string StateProblem::state_key(const StatePtr &state) const {
  check(state);
  auto history = state->history();
  std::sort(history.begin(), history.end());
  Json schedule = nullptr;
  if (data_->native && data_->trees.empty()) {
    schedule = Json::array();
    for (const auto &text : state->history()) {
      auto action = Json::parse(text);
      if (data_->ops.at(action["operator_id"].get<Id>()).semantic)
        schedule.push_back(std::move(action));
    }
  }
  return Json::array({history, state->root_ids(), schedule}).dump();
}

bool StateProblem::relational_inner_join(Id oid) const {
  const auto &op = data_->ops.at(oid);
  const auto &node = data_->logical.at(op.node);
  return !op.semantic && node.join_type == "INNER" &&
         (node.type == "COMPARISON_JOIN" || node.type == "ANY_JOIN" || node.type == "JOIN");
}
bool StateProblem::semantic_filter(Id oid) const {
  const auto &op = data_->ops.at(oid);
  return op.semantic && data_->logical.at(op.node).type == "SEMANTIC_FILTER";
}
const std::string &StateProblem::operator_kind(Id oid) const {
  return data_->logical.at(data_->ops.at(oid).node).type;
}

LogicalStructure StateProblem::logical_structure(const StatePtr &state) const {
  check(state);
  LogicalStructure result;
  std::unordered_map<Id, std::string> shapes;
  for (const auto &node : state->nodes()) {
    const auto &logical = data_->logical.at(node->node_id);
    auto label = logical.type;
    // Preserve value-producing boundaries, including generated-column roles.
    if (label.find("PROJECTION") != std::string::npos)
      label += "#" + std::to_string(node->node_id);
    std::vector<std::string> children;
    for (auto child : node->children)
      children.push_back(shapes.at(child));
    if (logical.commutative)
      std::sort(children.begin(), children.end());
    std::string shape = label + "(";
    if (children.empty())
      shape += std::to_string(node->node_id);
    for (const auto &child : children)
      shape += child + ",";
    shape += ")";
    shapes.emplace(node->node_id, shape);
    // Repeated Filter tasks have the same structural label. Their real IDs,
    // fields and implementations remain intact in State and legality checks.
    if (!children.empty())
      result.features.insert(shape);
  }
  std::vector<std::string> roots;
  for (const auto &root : state->roots)
    roots.push_back(shapes.at(root->node_id));
  std::sort(roots.begin(), roots.end());
  for (const auto &root : roots)
    result.key += std::to_string(root.size()) + ":" + root;
  return result;
}
}
