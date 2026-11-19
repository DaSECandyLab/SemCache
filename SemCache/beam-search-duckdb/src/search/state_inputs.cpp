#include "state_core_internal.hpp"

namespace semcache {
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
}
