#include "environment_internal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semcache::sampling {

using namespace profile_detail;

Json ProfileEnvironment::Impl::predecessor_survival(const StateData &context, const StateData &structure,
                                                    const NodePtr &node, const std::string &name,
                                                    const std::string &depth, const std::string &expression,
                                                    const std::string &expression_type) const {
  if (!node->operator_id)
    return child(context, node).get("output_rows") == 0 ? 0. : 1.;
  const auto &op = ops.at(*node->operator_id);
  if (prefix_population_boundary(kind(op)))
    return child(context, node).get("output_rows") == 0 ? 0. : 1.;
  std::string source_name = name;
  if (!expression.empty())
    for (const auto &descriptor : input_schema(op))
      if (descriptor.value("sample_expression", std::string()) ==
              expression &&
          descriptor.value("type", std::string()) == expression_type) {
        source_name = descriptor.value("value_key", name);
        break;
      }
  if (kind(op) == "SEMANTIC_PROJECTION") {
    std::set<Id> existing;
    for (const auto child_id : node->children)
      for (const auto id : indexed_node(structure, child_id)->fields)
        existing.insert(id);
    for (const auto id : node->fields)
      if (!existing.count(id)) {
        const auto &field = fields.at(id);
        const auto generated = field.is_string() ? field.get<std::string>()
                               : field.contains("name")
                                   ? field.at("name").get<std::string>()
                                   : canonical(field.at("binding"));
        if (generated == name ||
            (!expression.empty() &&
             expression.find('"' + generated + '"') != std::string::npos))
          return 1.;
      }
  }
  const auto &record = child(context, node);
  auto local = record.get("selectivity");
  const auto label = kind(op);
  if (prefix_preserving_operator(label))
    local = 1.;
  if (label.find("JOIN") != std::string::npos) {
    const auto bindings = expression.empty()
                              ? decltype(expression_bindings(expression)){}
                              : expression_bindings(expression);
    for (std::size_t side = 0; side < node->children.size(); ++side) {
      const auto &source = indexed_node(structure, node->children[side]);
      bool contains = has_field(source, name);
      for (const auto id : source->fields) {
        const auto &field = fields.at(id);
        const auto column = field.is_string() ? field.get<std::string>()
                            : field.contains("name")
                                ? field.at("name").get<std::string>()
                                : canonical(field.at("binding"));
        contains |= bindings.count(column) != 0;
      }
      const auto side_rate =
          record.get(side ? "right_selectivity" : "left_selectivity");
      if (contains && number(side_rate, true))
        local = side_rate;
    }
  }
  bool measured = false;
  if (!prefix_preserving_operator(label) &&
      label.find("JOIN") == std::string::npos) {
    const auto rt = context.runtime_selectivities.find(node->node_id);
    if (rt != context.runtime_selectivities.end()) {
      local = rt->second;
      measured = true;
    }
  }
  // Frozen local estimates retain the original field identity. Prefer their
  // complete grid; raw observations remain available as evidence only.
  const auto &dense = record.get("prefix_selectivity_statistics");
  const auto &raw = record.get("depth_selectivity_statistics");
  auto matching = [&](const std::string &key) {
    const auto suffix = ':' + source_name;
    return key.size() >= suffix.size() &&
           key.compare(key.size() - suffix.size(), suffix.size(), suffix) ==
               0;
  };
  if (!measured && dense.is_object())
    for (auto field = dense.begin(); field != dense.end(); ++field) {
      if (!matching(field.key()))
        continue;
      if (field.value().contains(depth)) {
        const auto &entry = field.value().at(depth);
        const auto value = entry.value("rate", Json(nullptr));
        const auto source = entry.value("source", std::string());
        if (number(value, true) &&
            (entry.value("observed", std::size_t(0)) > 0 ||
             source == "exact" || source == "empty_population")) {
          local = value;
          measured = true;
          break;
        }
      }
      // Recompute missing depths and historical interpolation from original
      // measurements and the operator's whole-field rate.
      const auto evidence = raw.is_object()
                                ? raw.value(field.key(), Json::object())
                                : Json::object();
      auto field_scalar = local;
      if (field.value().contains("0")) {
        const auto &zero = field.value().at("0");
        const auto source = zero.value("source", std::string());
        if (zero.value("observed", std::size_t(0)) > 0 ||
            source == "operator_scalar" || source == "exact" ||
            source == "empty_population")
          field_scalar = zero.value("rate", local);
      }
      const auto expanded =
          complete_prefix_selectivity({{depth, true}}, evidence, field_scalar,
                                      prefix_preserving_operator(label));
      const auto value = expanded.at(depth).at("rate");
      if (number(value, true)) {
        local = value;
        measured = true;
        break;
      }
    }
  if (!measured && raw.is_object())
    for (auto field = raw.begin(); field != raw.end(); ++field)
      if (matching(field.key())) {
        const auto expanded =
            complete_prefix_selectivity({{depth, true}}, field.value(), local,
                                        prefix_preserving_operator(label));
        const auto value = expanded.at(depth).at("rate");
        if (number(value, true)) {
          local = value;
          measured = true;
          break;
        }
      }
  Json incoming = 1.;
  for (std::size_t side = 0; side < node->children.size(); ++side) {
    const auto &source = indexed_node(structure, node->children[side]);
    bool contains = has_field(source, name);
    const auto bindings = expression_bindings(expression);
    for (const auto id : source->fields) {
      const auto &field = fields.at(id);
      const auto column = field.is_string() ? field.get<std::string>()
                          : field.contains("name")
                              ? field.at("name").get<std::string>()
                              : canonical(field.at("binding"));
      contains |= bindings.count(column) != 0;
    }
    if (!contains && (expression.empty() || node->children.size() != 1))
      continue;
    const auto inherited = predecessor_survival(
        context, structure, source, name, depth, expression, expression_type);
    incoming = incoming == 0 || inherited == 0 ? Json(0.)
               : number(incoming, true) && number(inherited, true)
                   ? Json(incoming.get<double>() * inherited.get<double>())
                   : Json(nullptr);
    if (!measured && label.find("JOIN") != std::string::npos) {
      const auto side_rate =
          record.get(side ? "right_selectivity" : "left_selectivity");
      if (number(side_rate, true))
        local = side_rate;
    }
  }
  return incoming == 0 || local == 0 ? Json(0.)
         : number(incoming, true) && number(local, true)
             ? Json(incoming.get<double>() * local.get<double>())
             : Json(nullptr);
}

Value ProfileEnvironment::Impl::incoming_survival(const StateData &context, const StateData &structure,
                                                  const NodePtr &node, const std::string &name,
                                                  Value summary_owner, Value single_owner) const {
  static const Value missing = std::make_shared<const Json>(nullptr);
  if (!summary_owner)
    summary_owner = missing;
  const auto &summary = *summary_owner;
  std::string expression, expression_type;
  if (node->operator_id)
    for (const auto &descriptor : input_schema(ops.at(*node->operator_id)))
      if (descriptor.value("value_key", std::string()) == name) {
        expression = descriptor.value("sample_expression", std::string());
        expression_type = descriptor.value("type", std::string());
        break;
      }
  std::vector<std::uint64_t> inputs;
  inputs.reserve(node->children.size());
  for (const auto child : node->children)
    inputs.push_back(indexed_node(structure, child)->handle_id);
  SurvivalMemoKey identity{
      inputs, name, expression, expression_type,
      reinterpret_cast<std::uintptr_t>(summary_owner.get()),
      reinterpret_cast<std::uintptr_t>(single_owner.get())};
  if (!runtime_observed) {
    const auto found = survival_memo.find(identity);
    if (found != survival_memo.end())
      return found->second.value;
  }
  std::vector<NodePtr> sources;
  const auto bindings = expression.empty()
                            ? decltype(expression_bindings(expression)){}
                            : expression_bindings(expression);
  for (const auto child_id : node->children) {
    const auto &source = indexed_node(structure, child_id);
    bool contains = has_field(source, name);
    for (const auto id : source->fields) {
      const auto &field = fields.at(id);
      const auto column = field.is_string() ? field.get<std::string>()
                          : field.contains("name")
                              ? field.at("name").get<std::string>()
                              : canonical(field.at("binding"));
      contains |= bindings.count(column) != 0;
    }
    if (contains || (!expression.empty() && node->children.size() == 1))
      sources.push_back(source);
  }
  auto rate = [&](const std::string &depth) {
    Json value = 1.;
    for (const auto &source : sources) {
      const auto factor =
          predecessor_survival(context, structure, source, name, depth,
                               expression, expression_type);
      value = value == 0 || factor == 0 ? Json(0.)
              : number(value, true) && number(factor, true)
                  ? Json(value.get<double>() * factor.get<double>())
                  : Json(nullptr);
    }
    return value;
  };
  Json depths = Json::object();
  const bool has_single =
      single_owner && single_owner.get() != summary_owner.get() &&
      single_owner->is_object() && single_owner->contains("sketches");
  if (summary.is_object() && summary.contains("sketches"))
    for (auto depth = summary.at("sketches").begin();
         depth != summary.at("sketches").end(); ++depth) {
      if (depth.key() == "0")
        continue;
      if (depth.value().at("entries").empty()) {
        depths[depth.key()] = 0.;
        continue;
      }
      if (has_single) {
        const auto &single_sketches = single_owner->at("sketches");
        if (!single_sketches.contains(depth.key()) ||
            single_sketches.at(depth.key()).at("entries").empty())
          continue;
      }
      depths[depth.key()] = rate(depth.key());
    }
  auto value = std::make_shared<const Json>(
      Json{{"total", rate("0")}, {"depths", std::move(depths)}});
  if (!runtime_observed)
    survival_memo.emplace(
        std::move(identity),
        SurvivalMemo{std::move(inputs), std::move(summary_owner), value});
  return value;
}

void ProfileEnvironment::Impl::estimate_survivals(Record &result, const StateData &context,
                                                  const StateData &structure, const NodePtr &node,
                                                  const Choice &chosen) const {
  if (!sample_depth_selectivity || !semantic(ops.at(chosen.operator_id)))
    return;
  const auto &layout = chosen.action.at("layout");
  const auto &op = ops.at(chosen.operator_id);
  auto describe = [&](const std::string &name) {
    for (const auto &field : input_schema(op))
      if (field.at("name") == name)
        return field.value("value_key", name);
    return name;
  };
  auto first = layout.value("field_order",
                            layout.value("left_field_order", Json::array()));
  if (!first.empty())
    result.share("alpha",
                 incoming_survival(context, structure, node,
                                   describe(first[0].get<std::string>()),
                                   result.reference("summary"),
                                   result.reference("single_summary")));
  const auto right = layout.value("right_field_order", Json::array());
  if (!right.empty())
    result.share("right_alpha",
                 incoming_survival(context, structure, node,
                                   describe(right[0].get<std::string>()),
                                   result.reference("right_summary"),
                                   result.reference("single_right_summary")));
}

Record ProfileEnvironment::Impl::estimate(const StateData &context, const StateData &structure,
                                          const NodePtr &node, const Choice &chosen) {
  const auto &op = ops.at(chosen.operator_id);
  const bool is_semantic = semantic(op);
  Record result = is_semantic ? Record{}
                              : *find(node_priors, std::to_string(chosen.node_id));
  if (is_semantic) {
    const auto source =
        find(operator_priors, std::to_string(chosen.operator_id));
    const auto &implementations = source->get("implementations");
    if (source->get("selectivity_scope") == "operator")
      result = *source;
    else if (implementations.is_object()) {
      const auto measured = implementations.find(chosen.implementation_key);
      if (measured != implementations.end())
        result = import_record(imported, *measured);
    } else if (source->get("sample_model") == chosen.action.at("model_id") &&
               source->get("sample_layout") == chosen.action.at("layout"))
      result = *source;
  }
  if (semantic(op)) {
    // A construction adds each logical operator once. An existing node is
    // its selected/runtime-updated record; before construction only the
    // immutable ingress bank can contain an observation for this action.
    const auto &selected = node_record(context, chosen.node_id);
    const auto own = selected ? selected : find(initial, chosen.key);
    if (own->get("sampling_strategy") == "independent_base")
      result.merge(*own);
    static const Json empty = Json::object(), unknown = nullptr;
    const auto &bank = value_reference(profile_, "accuracy", empty);
    const auto &model = value_reference(
        bank, chosen.action.at("model_id").get<std::string>(), empty);
    const auto &task =
        value_reference(model, op.at("kind").get<std::string>(), unknown);
    const auto &evidence =
        accuracy_evidence(task, chosen.action.at("layout"));
    const auto &accuracy = evidence.is_object()
                               ? value_reference(evidence, "value", unknown)
                               : evidence;
    if (number(accuracy))
      result.set("accuracy", accuracy);
    else
      result.erase("accuracy");
  }
  result.set("sampling_strategy", "independent_base");
  std::vector<const Record *> children;
  Json sizes = Json::array();
  bool known = !node->children.empty();
  for (auto id : node->children) {
    const auto &info = child(context, indexed_node(structure, id));
    auto size = info.contains("output_rows")
                    ? info.get("output_rows")
                    : logical.at(id).value("estimated_rows", Json(nullptr));
    known &= number(size, true);
    sizes.push_back(size);
    children.push_back(&info);
  }
  if (known) {
    // Cardinality propagation touches scalar facts only; distribution/sketch
    // fields keep their immutable aliases when no new merged bank is needed.
    Json scalar = Json::object();
    for (const auto *key :
         {"input_rows", "output_rows", "input_side_rows", "selectivity",
          "relational_cost", "relational_seconds", "cardinality_basis",
          "sampled_selectivity", "sampling_selectivity",
          "sampled_selectivity_complete", "uncertain",
          "survival_estimated", "group_count_estimated", "duckdb_selectivity",
          "estimated_selectivity", "selectivity_fusion",
          "survival_estimation_basis"})
      if (result.contains(key))
        scalar[key] = result.get(key);
    auto propagated = propagate(op, scalar, sizes.get<std::vector<Json>>());
    for (const auto *key :
         {"input_rows", "output_rows", "input_side_rows", "selectivity",
          "relational_cost", "relational_seconds", "cardinality_basis",
          "group_count_estimated", "duckdb_selectivity", "sampling_selectivity",
          "estimated_selectivity", "selectivity_fusion", "survival_estimated",
          "survival_estimation_basis"}) {
      if (propagated.contains(key))
        result.set(key, propagated.at(key));
      else
        result.erase(key);
    }
    Json widths = Json::array();
    double input_width = 0;
    bool width_known = true;
    for (const auto &info : children) {
      widths.push_back(info->get("output_width"));
      width_known &= number(info->get("output_width"));
      if (number(info->get("output_width")))
        input_width += info->get("output_width").get<double>();
    }
    if (width_known) {
      result.set("input_width", input_width);
      result.set("input_side_widths", widths);
    }
    const bool needs_input = !result.contains("input_field_statistics");
    const bool needs_output =
        kind(op).find("PROJECTION") == std::string::npos &&
        !result.contains("output_field_statistics");
    if (needs_input || needs_output) {
      Value field_statistics;
      if (children.size() == 1)
        field_statistics = children[0]->reference("output_field_statistics");
      if (!field_statistics) {
        Json merged = Json::object();
        for (const auto &info : children)
          if (info->get("output_field_statistics").is_object())
            merged.update(info->get("output_field_statistics"));
        field_statistics = std::make_shared<const Json>(std::move(merged));
      }
      if (needs_input)
        result.share("input_field_statistics", field_statistics);
      if (needs_output)
        result.share("output_field_statistics", field_statistics);
    }
    auto output_width =
        width(node, fields, result.get("output_field_statistics"));
    if (!output_width.is_null())
      result.set("output_width", std::move(output_width));
  } else if (!semantic(op)) {
    for (const auto *key : {"input_rows", "output_rows", "relational_cost",
                            "relational_seconds"})
      result.erase(key);
  }
  if (semantic(op) && sample_depth_selectivity &&
      result.get("input_field_summaries").is_object()) {
    const auto &layout = chosen.action.at("layout");
    auto attach = [&](const char *order, const char *slot,
                      const char *single_slot) {
      const auto names = layout.value(order, Json::array());
      if (names.empty())
        return;
      auto vk_of = [&](const Json &name) {
        auto key = name.get<std::string>();
        for (const auto &field : input_schema(op))
          if (field.at("name") == name)
            key = field.value("value_key", key);
        return key;
      };
      std::vector<std::string> vks;
      for (const auto &n : names)
        vks.push_back(vk_of(n));
      const std::string &first_vk = vks[0];
      const std::string joint_key =
          vks.size() >= 2 ? "joint:" + canonical(Json(vks)) : std::string();
      const auto owner = result.reference("input_field_summaries");
      const auto &summaries = *owner;
      const auto &selected =
          (!joint_key.empty() && summaries.contains(joint_key) &&
           summaries.at(joint_key).contains("summary"))
              ? joint_key
              : first_vk;
      if (summaries.contains(first_vk) &&
          summaries.at(first_vk).contains("summary"))
        result.share(single_slot,
                     Value(owner, &summaries.at(first_vk).at("summary")));
      if (summaries.contains(selected) &&
          summaries.at(selected).contains("summary")) {
        result.share(slot, Value(owner, &summaries.at(selected).at("summary")));
        if (std::string(slot) == "summary") {
          if (summaries.at(selected).contains("origin"))
            result.share("cache_field_origin",
                         Value(owner, &summaries.at(selected).at("origin")));
          else
            result.erase("cache_field_origin");
        }
      }
    };
    attach(layout.contains("field_order") ? "field_order"
                                          : "left_field_order",
           "summary", "single_summary");
    attach("right_field_order", "right_summary", "single_right_summary");
  }
  if (semantic(op) && sample_depth_selectivity) {
    estimate_survivals(result, context, structure, node, chosen);
    result.erase("single_summary");
    result.erase("single_right_summary");
    cache_edges(result, context, structure.nodes, node, chosen);
  } else {
    result.erase("single_summary");
    result.erase("single_right_summary");
  }
  return result;
}

std::shared_ptr<const Choice> ProfileEnvironment::Impl::choice(const StatePtr &state) {
  const auto node_id = state->added->node_id;
  auto group = choice_cache.find(node_id);
  if (group != choice_cache.end()) {
    const auto found = group->second.find(state->action_json);
    if (found != group->second.end())
      return found->second;
  }
  auto action = Json::parse(state->action_json);
  auto key = digest(action);
  std::string implementation_key, layout_key;
  if (action.contains("layout")) {
    layout_key = canonical(action.at("layout"));
    if (action.contains("model_id"))
      implementation_key =
          canonical(Json{{"operator_id", state->action.operator_id},
                         {"model_id", action.at("model_id")},
                         {"layout", action.at("layout")}});
  }
  auto result = std::make_shared<const Choice>(
      Choice{state->action.operator_id, node_id, std::move(action),
             std::move(key), std::move(implementation_key),
             std::move(layout_key)});
  // Candidate orders commonly share the exact immutable action. Keep this
  // worker-local catalogue bounded; retained states independently own entries
  // even when their cache references are discarded.
  if (choice_cache_size >= 32768) {
    choice_cache.clear();
    choice_cache_size = 0;
  }
  choice_cache[node_id].emplace(state->action_json, result);
  ++choice_cache_size;
  return result;
}

StateData ProfileEnvironment::Impl::build_state(const StatePtr &state, const StateData *parent) {
  StateData data;
  data.state = state;
  if (!state->parent) {
    data.nodes = state->nodes();
    data.node_slots.reserve(data.nodes.size());
    data.node_by_slot.resize(node_index->ids.size());
    data.node_records.resize(node_index->ids.size());
    for (const auto &node : data.nodes) {
      const auto slot = node_index->slots.at(node->node_id);
      data.node_slots.push_back(slot);
      data.node_by_slot[slot] = node;
      data.node_records[slot] =
          find(initial, "scan:" + std::to_string(node->node_id));
    }
  } else {
    data.nodes = parent->nodes;
    data.nodes.push_back(state->added);
    data.node_slots = parent->node_slots;
    data.node_by_slot = parent->node_by_slot;
    data.node_records = parent->node_records;
    data.actions = parent->actions;
    data.runtime_selectivities = parent->runtime_selectivities;
    auto chosen = choice(state);
    const auto slot = node_index->slots.at(chosen->node_id);
    data.node_slots.push_back(slot);
    data.node_by_slot[slot] = state->added;
    data.node_records[slot] = std::make_shared<const Record>(
        estimate(*parent, data, state->added, *chosen));
    data.actions.push_back(std::move(chosen));
  }
  return data;
}

void ProfileEnvironment::Impl::observe(const StatePtr &state) {
  if (states.count(state->handle_id))
    return;
  if (state->parent)
    observe(state->parent);
  else if (states.empty()) {
    auto ids = std::set<Id>(node_index->ids.begin(), node_index->ids.end());
    for (const auto &node : state->nodes())
      ids.insert(node->node_id);
    if (ids.size() != node_index->ids.size()) {
      observation_workers.clear();
      node_index = std::make_shared<const NodeIndex>(ids);
    }
  }
  const auto *parent =
      state->parent ? &states.at(state->parent->handle_id) : nullptr;
  states.emplace(state->handle_id, build_state(state, parent));
}

Json ProfileEnvironment::Impl::statistics(const StatePtr &state, bool raw) const {
  const auto &data = states.at(state->handle_id);
  Json result = Json::object();
  for (std::size_t slot = 0; slot < data.node_records.size(); ++slot)
    if (const auto &record = data.node_records[slot])
      result[std::to_string(node_index->ids[slot])] = record->json(raw);
  return result;
}

} // namespace semcache::sampling
