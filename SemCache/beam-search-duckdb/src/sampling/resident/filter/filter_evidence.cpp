#include "../detail.hpp"

namespace semcache::sampling {
resident_detail::FilterPool &
NativeSampler::Impl::filter_pool(const std::string &relation) {
  auto found = filter_pools.find(relation);
  if (found != filter_pools.end())
    return *found->second;
  std::vector<const Json *> rows;
  std::vector<double> weights;
  std::vector<std::size_t> strata;
  const auto &source = relations.at(relation);
  for (std::size_t i = 0; i < source.strata.size(); ++i) {
    const auto &h = source.strata[i];
    for (const auto &row : h.rows) {
      rows.push_back(&row);
      weights.push_back(h.population / h.rows.size());
      strata.push_back(i);
    }
  }
  auto pool = std::make_unique<resident_detail::FilterPool>(
      std::move(rows), std::move(weights), std::move(strata));
  return *filter_pools.emplace(relation, std::move(pool)).first->second;
}

const resident_detail::FilterBinding *
NativeSampler::Impl::filter_binding(const Json &op) {
  if (selectivity_estimator == "legacy" ||
      op.value("kind", std::string()) != "SEMANTIC_FILTER" ||
      (settings.value("selectivity_parallel_models", true) && models(op).size() > 1))
    return nullptr;
  const Id id = op.at("operator_id");
  auto found = filter_bindings.find(id);
  if (found != filter_bindings.end())
    return found->second ? &*found->second : nullptr;
  const auto model = models(op)[0];
  RequestKeyPlan plan(op, model, op.at("default_layout"), false);
  resident_detail::FilterBinding binding;
  for (const auto &slot : plan.slots()) {
    if (slot.second || slot.first.rfind("__semantic_input:", 0) == 0) {
      filter_bindings.emplace(id, std::nullopt);
      return nullptr;
    }
    binding.fields.insert(slot.first);
  }
  if (binding.fields.empty()) {
    filter_bindings.emplace(id, std::nullopt);
    return nullptr;
  }
  for (const auto &entry : relations) {
    bool eligible = true;
    for (const auto &field : binding.fields)
      eligible &= entry.second.summaries.count(field) != 0;
    if (!eligible)
      continue;
    if (!binding.relation.empty()) {
      filter_bindings.emplace(id, std::nullopt);
      return nullptr;
    }
    binding.relation = entry.first;
  }
  if (binding.relation.empty()) {
    filter_bindings.emplace(id, std::nullopt);
    return nullptr;
  }
  binding.task = digest(Json::array(
      {binding.relation, model, op.at("task"), op.at("default_layout"),
       op.value("input_field_schema", Json::array()),
       op.value("operator_specification", Json(nullptr))}));
  auto inserted = filter_bindings.emplace(id, std::move(binding)).first;
  return &*inserted->second;
}

std::optional<std::size_t> NativeSampler::Impl::filter_coordinate(
    const resident_detail::FilterBinding &binding, const Json &row) {
  auto &pool = filter_pool(binding.relation);
  for (const auto &origin : row.at("lineage"))
    if (relkey(origin[0]) == binding.relation) {
      const auto position = pool.coordinate(origin[1].get<std::string>());
      if (position && pool.compatible(*position, row, binding.fields))
        return position;
      return std::nullopt;
    }
  return std::nullopt;
}

std::optional<Selection>
NativeSampler::Impl::filter_selection(const Json &op,
                                      const SampleInputs &inputs) {
  const auto *binding = filter_binding(op);
  if (!binding || inputs.size() != 1 || !row_count(inputs[0]))
    return std::nullopt;
  PhaseTimer timer(phase_metrics, "filter_mask_selection", trace_timings, true);
  const auto leaf = leaf_for(inputs[0], binding->fields);
  if (!leaf)
    return std::nullopt;
  const auto &source = leaf_at(inputs[0], *leaf);
  auto &pool = filter_pool(binding->relation);
  auto decisions =
      std::make_shared<resident_detail::DecisionMask>(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto coordinate = filter_coordinate(*binding, source[i]);
    if (!coordinate)
      return std::nullopt;
    const auto answer = pool.decision(binding->task, *coordinate);
    if (!answer)
      return std::nullopt;
    decisions->publish(i, *answer);
  }
  Json output = Json::array();
  decisions->each_accepted([&](std::size_t i) {
    output.push_back(
        resident_detail::restore_row(op, source[i], {true, nullptr}));
  });
  if (!leaf->empty())
    output = replace_leaf(inputs[0], *leaf, output);
  Selection result;
  result.rows = std::make_shared<const Json>(std::move(output));
  result.decisions = std::move(decisions);
  counters["filter_mask_hits"] =
      counters.value("filter_mask_hits", std::size_t(0)) + 1;
  return result;
}

double NativeSampler::Impl::selection_effective(const Json &op,
                                                const SampleInputs &inputs,
                                                double ordinary_effective) {
  const auto *binding = filter_binding(op);
  if (!binding || inputs.size() != 1 || !row_count(inputs[0]))
    return ordinary_effective;
  const auto leaf = leaf_for(inputs[0], binding->fields);
  if (!leaf)
    return ordinary_effective;
  const auto &source = leaf_at(inputs[0], *leaf);
  std::map<std::size_t, double> independent;
  for (const auto &row : source) {
    const auto coordinate = filter_coordinate(*binding, row);
    if (!coordinate)
      return ordinary_effective;
    independent[*coordinate] += weight(row);
  }
  double mass = 0., squares = 0.;
  for (const auto &entry : independent) {
    mass += entry.second;
    squares += entry.second * entry.second;
  }
  return std::min(ordinary_effective, squares ? mass * mass / squares : 0.);
}

void NativeSampler::Impl::record_filter(const NodePtr &node, const Json &op,
                                        const SampleInputs &inputs,
                                        const Selection &selection) {
  if (selectivity_estimator == "legacy" || !selection || inputs.size() != 1)
    return;
  const auto *binding = filter_binding(op);
  const auto kind = op.value("kind", op.value("type", std::string()));
  const bool ordinary = !binding && kind == "FILTER" &&
                        fixed_relation_filter(op) &&
                        node->relations.size() == 1 && !product(inputs[0]);
  if (!binding && !ordinary)
    return;
  PhaseTimer timer(phase_metrics, "filter_mask_publication", trace_timings,
                   true);
  resident_detail::FilterBinding relational;
  if (ordinary) {
    relational.relation = node_relkey(*node->relations.begin());
    relational.task = "relation-filter:" + std::to_string(*node->operator_id);
    for (const auto &field : op["sampling_required_fields"]) {
      if (!field.is_string())
        return;
      const auto name = field.get<std::string>();
      if (!relations.at(relational.relation).summaries.count(name))
        return;
      relational.fields.insert(name);
    }
    binding = &relational;
  }
  const auto leaf = leaf_for(inputs[0], binding->fields);
  if (!leaf)
    return;
  const auto &source = leaf_at(inputs[0], *leaf);
  auto &pool = filter_pool(binding->relation);
  std::set<std::string> passed;
  if (ordinary) {
    if (product(*selection.rows))
      return;
    for (const auto &row : *selection.rows)
      passed.insert(row.at("id").get<std::string>());
  }
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto coordinate = filter_coordinate(*binding, source[i]);
    if (!coordinate)
      continue;
    if (ordinary)
      pool.publish(binding->task, *coordinate,
                   passed.count(source[i].at("id").get<std::string>()) != 0);
    else if (selection.decisions && selection.decisions->known(i))
      pool.publish(binding->task, *coordinate,
                   selection.decisions->accepted(i));
  }
}

std::optional<resident_detail::ConditionEstimate>
NativeSampler::Impl::conditional_selectivity(const NodePtr &node,
                                             const Json &op,
                                             const StateData &data,
                                             CoreObservation &core) {
  const auto *binding = filter_binding(op);
  if (!binding || node->children.size() != 1 || node->relations.size() != 1)
    return std::nullopt;
  auto &pool = filter_pool(binding->relation);
  std::vector<std::string> conditions;
  auto id = node->children[0];
  while (true) {
    const auto &ancestor = data.observations.at(id)->core->node;
    if (!ancestor->operator_id)
      break;
    if (ancestor->children.size() != 1 || ancestor->relations.size() != 1)
      return std::nullopt;
    const auto &stage = execution_op(data.state, ancestor);
    if (const auto *a = filter_binding(stage)) {
      if (a->relation != binding->relation)
        return std::nullopt;
      conditions.push_back(a->task);
    } else if (fixed_relation_filter(stage)) {
      conditions.push_back("relation-filter:" +
                           std::to_string(*ancestor->operator_id));
    } else
      return std::nullopt;
    id = ancestor->children[0];
  }
  core.condition_pool = binding->relation;
  core.condition_version = pool.version();
  PhaseTimer timer(phase_metrics, "conditional_selectivity", trace_timings,
                   true);
  return pool.estimate(binding->task, conditions, input_samples,
                       selectivity_estimator == "ancestor");
}
}
