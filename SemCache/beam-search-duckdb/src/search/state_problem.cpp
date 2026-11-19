#include "state_core_internal.hpp"

namespace semcache {
namespace state_detail {
std::atomic<std::uint64_t> next_handle{1};
std::set<Id> idset(const Ids &ids) { return {ids.begin(), ids.end()}; }
NodePtr root(const StatePtr &state, Id id) {
  for (const auto &n : state->roots)
    if (n->node_id == id)
      return n;
  return nullptr;
}
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
} // namespace state_detail

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
    n.shape_label = n.type;
    if (n.shape_label.find("PROJECTION") != std::string::npos)
      n.shape_label += "#" + std::to_string(n.id);
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
      if (!a.value.contains("inputs") && !a.value.contains("pair_for") &&
          a.canonical.find("__SC_IN__") == std::string::npos &&
          a.canonical.find("__SC_PF__") == std::string::npos) {
        static const std::string in_marker = "\"inputs\":\"__SC_IN__\"";
        static const std::string pf_marker = "\"pair_for\":\"__SC_PF__\"";
        Json v1 = a.value;
        v1["inputs"] = "__SC_IN__";
        const auto d1 = v1.dump();
        const auto pos1 = d1.find(in_marker);
        Json v2 = v1;
        v2["pair_for"] = "__SC_PF__";
        const auto d2 = v2.dump();
        const auto p_in = d2.find(in_marker);
        const auto p_pf = d2.find(pf_marker);
        if (pos1 != std::string::npos && p_in != std::string::npos &&
            p_pf != std::string::npos && p_in < p_pf) {
          a.can_append_inputs = true;
          a.in_prefix = d1.substr(0, pos1) + "\"inputs\":[";
          a.in_suffix = "]" + d1.substr(pos1 + in_marker.size());
          a.pair_prefix = d2.substr(0, p_in) + "\"inputs\":[";
          a.pair_mid = "]" + d2.substr(p_in + in_marker.size(),
                                       p_pf - (p_in + in_marker.size())) +
                       "\"pair_for\":";
          a.pair_suffix = d2.substr(p_pf + pf_marker.size());
        }
      }
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
}
