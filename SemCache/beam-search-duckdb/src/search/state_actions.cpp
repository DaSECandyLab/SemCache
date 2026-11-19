#include "state_core_internal.hpp"
#include <string_view>

namespace semcache {
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
  const auto &entry =
      data_->ops.at(action.operator_id).domain.at(action.domain_index);
  if (entry.can_append_inputs) {
    std::string out;
    if (!action.pair_for) {
      out.reserve(entry.in_prefix.size() + entry.in_suffix.size() +
                  action.inputs.size() * 8);
      out = entry.in_prefix;
      for (std::size_t i = 0; i < action.inputs.size(); ++i) {
        if (i)
          out.push_back(',');
        out += std::to_string(action.inputs[i]);
      }
      out += entry.in_suffix;
    } else {
      out.reserve(entry.pair_prefix.size() + entry.pair_mid.size() +
                  entry.pair_suffix.size() + action.inputs.size() * 8 + 12);
      out = entry.pair_prefix;
      for (std::size_t i = 0; i < action.inputs.size(); ++i) {
        if (i)
          out.push_back(',');
        out += std::to_string(action.inputs[i]);
      }
      out += entry.pair_mid;
      out += std::to_string(*action.pair_for);
      out += entry.pair_suffix;
    }
    return out;
  }
  auto value = entry.value;
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
  std::vector<const State *> chain;
  chain.reserve(state->depth);
  for (auto s = state.get(); s->parent; s = s->parent.get())
    chain.push_back(s);
  std::vector<std::string_view> sorted_history;
  sorted_history.reserve(chain.size());
  for (auto it = chain.rbegin(); it != chain.rend(); ++it)
    sorted_history.emplace_back((*it)->action_json);
  std::sort(sorted_history.begin(), sorted_history.end());
  Json history = Json::array();
  for (const auto sv : sorted_history)
    history.push_back(std::string(sv));
  Json schedule = nullptr;
  if (data_->native && data_->trees.empty()) {
    Json region_orders = Json::object(), model_orders = Json::object();
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      const auto *s = *it;
      const auto oid = s->action.operator_id;
      const auto &op = data_->ops.at(oid);
      if (!op.semantic)
        continue;
      const auto region = data_->predicate_nodes.find(op.node);
      if (region != data_->predicate_nodes.end())
        region_orders[std::to_string(data_->predicates[region->second].id)]
            .push_back(oid);
      const auto &model = op.domain.at(s->action.domain_index).model;
      if (model)
        model_orders[*model].push_back(oid);
    }
    schedule = {{"regions", std::move(region_orders)},
                {"models", std::move(model_orders)}};
  }
  return Json::array({std::move(history), state->root_ids(), std::move(schedule)}).dump();
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
const std::optional<std::string> &
StateProblem::action_model(const Action &action) const {
  return data_->ops.at(action.operator_id)
      .domain.at(action.domain_index)
      .model;
}

std::string StateProblem::logical_structure_key(
    const StatePtr &state,
    std::unordered_map<std::uint64_t, std::string> &shape_memo) const {
  check(state);
  std::vector<const State *> chain;
  const State *cur = state.get();
  while (cur && cur->parent) {
    if (shape_memo.count(cur->added->handle_id))
      break;
    chain.push_back(cur);
    cur = cur->parent.get();
  }
  if (cur && !cur->parent) {
    for (const auto &leaf : cur->roots) {
      if (!shape_memo.count(leaf->handle_id)) {
        const auto &logical = data_->logical.at(leaf->node_id);
        shape_memo.emplace(
            leaf->handle_id,
            logical.shape_label + "(" + std::to_string(leaf->node_id) + ")");
      }
    }
  }
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const auto &node = (*it)->added;
    const auto &parent_roots = (*it)->parent->roots;
    const auto &logical = data_->logical.at(node->node_id);
    std::string shape;
    if (node->children.empty()) {
      shape = logical.shape_label + "(" + std::to_string(node->node_id) + ")";
    } else {
      auto lookup = [&](Id child_id) -> const std::string & {
        for (const auto &r : parent_roots)
          if (r->node_id == child_id)
            return shape_memo.at(r->handle_id);
        throw std::out_of_range("Missing child node shape");
      };
      if (node->children.size() == 1) {
        const auto &c0 = lookup(node->children[0]);
        shape.reserve(logical.shape_label.size() + c0.size() + 4);
        shape = logical.shape_label;
        shape.push_back('(');
        shape += c0;
        shape.append(",)");
      } else if (node->children.size() == 2) {
        const auto &a = lookup(node->children[0]);
        const auto &b = lookup(node->children[1]);
        const bool swap = logical.commutative && b < a;
        shape.reserve(logical.shape_label.size() + a.size() + b.size() + 5);
        shape = logical.shape_label;
        shape.push_back('(');
        shape += (swap ? b : a);
        shape.push_back(',');
        shape += (swap ? a : b);
        shape.append(",)");
      } else {
        std::vector<std::string_view> children;
        children.reserve(node->children.size());
        for (auto cid : node->children)
          children.emplace_back(lookup(cid));
        if (logical.commutative)
          std::sort(children.begin(), children.end());
        shape = logical.shape_label;
        shape.push_back('(');
        for (const auto child : children) {
          shape += child;
          shape.push_back(',');
        }
        shape.push_back(')');
      }
    }
    shape_memo.emplace(node->handle_id, std::move(shape));
  }
  std::vector<std::string_view> roots;
  roots.reserve(state->roots.size());
  std::size_t total = 0;
  for (const auto &root : state->roots) {
    const auto &s = shape_memo.at(root->handle_id);
    roots.emplace_back(s);
    total += s.size() + 8;
  }
  std::sort(roots.begin(), roots.end());
  std::string key;
  key.reserve(total);
  for (const auto root : roots) {
    key += std::to_string(root.size());
    key.push_back(':');
    key += root;
  }
  return key;
}

std::string StateProblem::logical_structure_key(const StatePtr &state) const {
  std::unordered_map<std::uint64_t, std::string> shape_memo;
  return logical_structure_key(state, shape_memo);
}

LogicalStructure StateProblem::logical_structure(const StatePtr &state) const {
  check(state);
  std::unordered_map<std::uint64_t, std::string> shape_memo;
  LogicalStructure result;
  result.key = logical_structure_key(state, shape_memo);
  for (const auto &node : state->nodes())
    if (!node->children.empty())
      result.features.insert(shape_memo.at(node->handle_id));
  return result;
}
}
