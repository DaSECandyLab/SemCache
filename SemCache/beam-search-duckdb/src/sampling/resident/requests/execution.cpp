#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::execute(const Json &op, const SampleInputs &inputs,
                                  const Json &action, const Json *pairs,
                                  const std::string &purpose,
                                  const std::function<void()> &prepare) {
  PhaseTimer timer(
      phase_metrics,
      purpose == "selection"
          ? (action.is_null() ? "selection_relational" : "selection_semantic")
          : purpose,
      trace_timings);
  if (!backend)
    throw std::runtime_error("Native sampling requires a native SQL backend");
  if (purpose == "selection")
    counters["selectivity"] = counters["selectivity"].get<std::size_t>() + 1;
  counters["sql_stages"] = counters["sql_stages"].get<std::size_t>() + 1;
  const bool semantic = !action.is_null();
  const auto source = backend;
  const auto observations_before = source->observations().size();
  Json bounded = action;
  std::size_t reserved = 0;
  if (semantic) {
    const auto available = selection_remaining();
    bounded["llm_request_limit"] = available;
    if (state_actor) {
      reserved = available;
      pending_selection += reserved;
    }
  }
  auto release_reservation = [&] {
    if (!reserved)
      return;
    pending_selection -= reserved;
    reserved = 0;
    state_actor->task.notify();
  };
  bool accounted = false, submitted = false;
  auto account = [&] {
    if (!semantic || accounted || !submitted)
      return;
    accounted = true;
    const auto requests =
        source->selection_progress().at("requests").get<std::size_t>();
    if (!ledger.charge(requests))
      throw std::logic_error(
          "Sampling transport exceeded its LLM request allowance");
    selection_requests_by_operator[op.at("operator_id").get<Id>()] += requests;
  };
  auto started = std::chrono::steady_clock::now();
  try {
    Json output = state_backend_call(
        [&] {
          submitted = true;
          return pairs ? source->join_pairs(op, *pairs, bounded)
                       : source->execute(op, inputs, bounded);
        },
        prepare);
    const auto seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    release_reservation();
    backend_seconds += seconds;
    account();
    if ((purpose == "selection" || purpose == "product_calibration") &&
        source->observations().size() > observations_before)
      record_selection_time(op, inputs, action, pairs,
                            source->observations().back(),
                            source->selection_progress());
    if (!state_actor || !semantic) {
      state_turn();
      harvest();
    }
    return output;
  } catch (...) {
    release_reservation();
    backend_seconds += std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    account();
    if (!state_actor || !semantic) {
      state_turn();
      harvest();
    }
    throw;
  }
}

bool NativeSampler::Impl::resolve_groups(const Json &op,
                                         const SampleInputs &inputs,
                                         const std::string &model,
                                         const std::vector<Group> &observed,
                                         const std::function<void()> &prepare) {
  if (settings.value("selectivity_parallel_models", true) && models(op).size() > 1)
    return resolve_parallel_groups(op, inputs, observed, prepare);
  std::set<std::string> claimed;
  if (state_actor)
    for (const auto &group : observed)
      if (!ledger.find(group.key))
        claimed.insert(group.key);
  StateClaim claim(*this, selection_claims, {claimed.begin(), claimed.end()});
  std::vector<const Group *> selected;
  Json representatives = Json::array(), pairs = Json::array();
  std::size_t wanted = 0;
  auto available = selection_remaining();
  // Reserved HTTP capacity is not spent capacity. Wait for actual billing
  // and refunds rather than marking this task exhausted while another owns it.
  while (state_actor && !available && pending_selection) {
    const auto before = pending_selection;
    state_actor->task.wait([&] { return pending_selection < before; });
    available = selection_remaining();
  }
  for (auto &group : observed)
    if (!ledger.find(group.key)) {
      ++wanted;
      if (selected.size() >= available)
        continue;
      selected.push_back(&group);
      if (inputs.size() == 1)
        representatives.push_back(group.unit[0]);
      else
        pairs.push_back(Json::array({group.unit[0], group.unit[1]}));
    }
  if (selected.empty())
    return wanted == 0;
  if (deadline_reached())
    return false;
  Json action = {{"model_id", model}, {"layout", op.at("default_layout")}};
  auto finish = [&] {
    claim.release();
    state_turn();
    if (state_actor)
      harvest();
  };
  try {
    auto output = inputs.size() == 1 ? execute(op, {&representatives}, action,
                                               nullptr, "selection", prepare)
                                     : execute(op, inputs, action, &pairs,
                                               "selection", prepare);
    std::map<std::string, const Json *> by_id;
    for (const auto &r : output)
      by_id.emplace(r.at("id").get<std::string>(), &r);
    auto column = output_column(op);
    const auto progress = backend->selection_progress();
    const auto completed = progress.at("completed_inputs").get<std::size_t>();
    if (completed > selected.size())
      throw std::logic_error(
          "Sampling backend completed more inputs than supplied");
    for (std::size_t i = 0; i < completed; ++i) {
      const auto *group = selected[i];
      auto id = group->unit.size() == 1
                    ? group->unit[0]["id"].get<std::string>()
                    : digest(Json::array(
                          {group->unit[0].at("id"), group->unit[1].at("id")}));
      auto found = by_id.find(id);
      ledger.publish(group->key, found != by_id.end(),
                     found != by_id.end() && !column.empty()
                         ? found->second->at("values").at(column)
                         : Json(nullptr));
    }
    const bool pending = wanted > completed;
    finish();
    return !pending;
  } catch (...) {
    finish();
    throw;
  }
}

const Json &NativeSampler::Impl::execution_op(const StatePtr &state,
                                              const NodePtr &node) const {
  PhaseTimer timer(phase_metrics, "execution_operator", trace_timings, true);
  if (auto found = execution_ops.find(node->handle_id);
      found != execution_ops.end())
    return found->second;
  Json op = ops.at(*node->operator_id), names = Json::array();
  const auto &fields = imported.at("fields");
  auto all = state->nodes();
  std::function<std::string(Id)> shape = [&](Id id) {
    const auto found = std::find_if(all.begin(), all.end(), [&](const auto &n) {
      return n->node_id == id;
    });
    std::string result = std::to_string(id) + '[';
    for (const auto child : (*found)->children)
      result += shape(child) + ',';
    return result + ']';
  };
  if (!semantic_operator(op) && (join_operator(op, node->children.size()) ||
                                 op.value("kind", std::string()) == "FILTER")) {
    auto instance = shape(node->node_id);
    for (const auto field : node->fields)
      instance += 'f' + std::to_string(field);
    op["sampling_instance"] = std::move(instance);
  }
  for (auto child : node->children) {
    auto found = std::find_if(all.begin(), all.end(),
                              [&](auto &n) { return n->node_id == child; });
    Json side = Json::array();
    for (auto id : (*found)->fields) {
      const auto &f = fields.at(id);
      side.push_back(sample_column(f));
    }
    names.push_back(side);
  }
  if (semantic_operator(op)) {
    // An operator's independent sampling contract excludes unrelated fields
    // and the shape/order of earlier predicates. Preserve only real input
    // bindings on each role, so equivalent candidates share the same cohort.
    std::set<std::string> required;
    for (const auto &field : op.value("required_columns", Json::array()))
      required.insert(sample_column(field));
    Json stable = Json::array();
    for (const auto &side : names) {
      std::set<std::string> columns;
      for (const auto &field : side)
        if (required.count(field.get<std::string>()))
          columns.insert(field.get<std::string>());
      stable.push_back(columns);
    }
    names = std::move(stable);
    op["sampling_dependency_fields"] = Json::array();
  }
  op["input_columns"] = names;
  if (node->children.size() == 1 &&
      op.value("kind", std::string()).find("JOIN") != std::string::npos) {
    auto found = std::find_if(all.begin(), all.end(), [&](auto &n) {
      return n->node_id == node->children[0];
    });
    Json partitions = Json::array();
    for (auto &part : (*found)->pair_columns) {
      Json side = Json::array();
      for (auto id : part) {
        const auto &f = fields.at(id);
        side.push_back(sample_column(f));
      }
      partitions.push_back(side);
    }
    // Keep source-column ownership from the candidate pair. Independent
    // samples may slice unused columns; the backend validates that slice and
    // appends prepared arguments on their declared side without losing CAST
    // source columns or admitting fields from a different candidate boundary.
    if (!partitions.empty())
      op["pair_columns"] = partitions;
  }
  return execution_ops.emplace(node->handle_id, std::move(op)).first->second;
}

OwnedInputs NativeSampler::Impl::prepare_inputs(const Json &op,
                                                const OwnedInputs &inputs) {
  PhaseTimer timer(phase_metrics, "prepare_inputs", trace_timings);
  if (!backend || deadline_reached())
    return inputs;
  OwnedInputs result;
  result.reserve(inputs.size());
  for (std::size_t side = 0; side < inputs.size(); ++side) {
    if (deadline_reached()) {
      result.push_back(inputs[side]);
      continue;
    }
    auto &rows = *inputs[side];
    Json generated = Json::array();
    for (const auto &field : op.at("input_field_schema")) {
      if (field.value("value_key", std::string())
              .rfind("__semantic_input:", 0) != 0)
        continue;
      bool right = false;
      for (const auto &name : op.at("right_input_fields"))
        right |= name == field.at("name");
      if (inputs.size() == 1 || right == (side == 1))
        generated.push_back(field);
    }
    if (generated.empty() || !row_count(rows)) {
      result.push_back(inputs[side]);
      continue;
    }
    Json side_op = op;
    side_op["input_field_schema"] = std::move(generated);
    auto prepare_side = [&](const SampleInputs &side) {
      const auto source = backend;
      auto operation = [&] { return source->prepare_inputs(side_op, side); };
      return state_actor ? state_actor->task.outside(operation) : operation();
    };
    if (!product(rows)) {
      auto prepared = prepare_side({&rows});
      result.push_back(std::make_shared<const Json>(std::move(prepared[0])));
      continue;
    }
    std::set<std::string> dependencies;
    for (auto &f : op.value("required_columns", Json::array()))
      dependencies.insert(f.is_string() ? f.get<std::string>() : field_key(f));
    auto leaf = dependencies.empty() ? std::optional<std::vector<int>>{}
                                     : leaf_for(rows, dependencies);
    if (leaf) {
      auto replacement = prepare_side({&leaf_at(rows, *leaf)})[0];
      result.push_back(
          std::make_shared<const Json>(replace_leaf(rows, *leaf, replacement)));
    } else {
      auto prepared = prepare_side({&rows});
      result.push_back(std::make_shared<const Json>(std::move(prepared[0])));
    }
  }
  return result;
}

}
