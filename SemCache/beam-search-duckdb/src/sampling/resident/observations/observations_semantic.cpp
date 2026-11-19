#include "../detail.hpp"
#include "../join/join_candidates.hpp"
#include "../survival/estimate_fields.hpp"
#include <numeric>
#include <unordered_set>

namespace semcache::sampling {
void NativeSampler::Impl::replay_semantic_node(
    const StatePtr &state, const NodePtr &node, StateData &data,
    const std::vector<NodePtr> &nodes, const std::string &nid,
    std::shared_ptr<Observation> &observation,
    std::shared_ptr<CoreObservation> &core) {
  Id oid = *node->operator_id;
  sampling_input_owner = oid;
  const auto &op = execution_op(state, node);
  const auto &action = action_for(state, oid);
  bool semantic = semantic_operator(op);
  const auto kind = op.value("kind", op.value("type", std::string()));
  const bool algebraic_projection =
      semantic && kind == "SEMANTIC_PROJECTION";
  const bool cardinality_boundary =
      !sample_depth_selectivity && !semantic &&
      (kind == "PROJECTION" || kind == "CTE_SCAN" ||
       op.contains("sampling_passthrough_input"));
  const bool estimate_only_relation =
      !sample_depth_selectivity && !semantic;
  OwnedInputs owned_inputs;
  std::vector<Json> sizes;
  bool input_complete = true;
  {
    PhaseTimer timer(phase_metrics, "independent_base_inputs",
                     trace_timings, true);
    auto required = independent_dependencies(op, node);
    // Mapping dependencies must remain bound even in request-count mode.
    // Clearing them reconstructs unrelated Cartesian source domains.
    if (op.value("sampling_cardinality_only", false) ||
        (algebraic_projection && !sample_depth_selectivity &&
         !ensure_prefix_depth_rates))
      required.clear();
    for (std::size_t side = 0; side < node->children.size(); ++side) {
      const auto child = node->children[side];
      const auto &previous = data.observations.at(child);
      if (op.contains("sampling_passthrough_input") &&
          side != op.at("sampling_passthrough_input").get<std::size_t>())
        // Preserve side indices and published statistics without rebuilding
        // the unused producer's independent Cartesian population.
        owned_inputs.push_back(previous->rows);
      else
        owned_inputs.push_back(independent_population(
            state, previous->core->node, required, input_complete,
            semantic ? initial_size : relational_sample_rows));
      sizes.push_back(
          previous->statistics.read().value("output_rows", Json(nullptr)));
    }
  }
  // Bound every consumer, including value-mode relational SQL. Semantic
  // producers above retain their own smaller limit even for a relational
  // consumer that can afford a larger source pool.
  owned_inputs = bound_independent_inputs(op, owned_inputs);
  std::size_t candidates = 1;
  for (const auto &rows : owned_inputs) {
    const auto count = row_count(*rows);
    if (count &&
        candidates > std::numeric_limits<std::size_t>::max() / count)
      throw std::overflow_error("Sample candidate population overflow");
    candidates *= count;
  }
  bool bounded_join =
      semantic &&
      op.value("kind", std::string()).find("JOIN") != std::string::npos &&
      candidates > join_pair_budget;
  if (semantic && input_complete && !bounded_join &&
      (!algebraic_projection || sample_depth_selectivity ||
       ensure_prefix_depth_rates))
    owned_inputs = prepare_inputs(op, owned_inputs);
  SampleInputs inputs(owned_inputs);
  for (const auto &input : owned_inputs)
    row_weights.register_population(input);
  core->input_complete = input_complete;
  Selection emitted;
  if (input_complete) {
    if (algebraic_projection || estimate_only_relation) {
      const auto side =
          op.value("sampling_passthrough_input", std::size_t(0));
      emitted.rows = owned_inputs.at(side);
      if (estimate_only_relation) {
        // Downstream operators rebuild independent inputs. EXPLAIN below
        // supplies cardinalities without materializing unused pair rows.
        if (!op.contains("sampling_passthrough_input"))
          for (std::size_t i = 1; i < owned_inputs.size(); ++i)
            emitted.rows = std::make_shared<const Json>(
                Json{{"product",
                      Json::array({*emitted.rows, *owned_inputs[i]})}});
        emitted.diagnostics["relational_estimate_only"] = true;
        if (cardinality_boundary)
          emitted.diagnostics["cardinality_boundary_algebraic"] = true;
      } else
        emitted.diagnostics["projection_selectivity_algebraic"] = true;
    } else
      emitted = selection(op, inputs, semantic, core->full_join_input);
  }
  observation->complete = bool(emitted);
  if (!emitted.prepared_inputs.empty())
    owned_inputs = emitted.prepared_inputs;
  inputs = SampleInputs(owned_inputs);
  if (!estimate_only_relation)
    record_filter(node, op, inputs, emitted);
  for (const auto &input : owned_inputs)
    population_moments.register_population(input);
  if (emitted.probes)
    population_moments.register_population(emitted.probes);
  observation->rows =
      emitted ? emitted.rows : std::make_shared<const Json>(Json::array());
  core->probe_units = emitted.units;
  core->decisions = emitted.decisions;
  core->join_evidence = emitted.join_evidence;
  core->statistical_rows = emitted.statistical_rows;
  core->join_sample_key = emitted.join_sample_key;
  core->join_probe_limit = emitted.join_probe_limit;
  core->join_stream_key = emitted.join_stream_key;
  core->join_output_limit = emitted.join_output_limit;
  const bool probe_sampled =
      (bool(emitted.probes) || bool(emitted.join_evidence)) &&
      !emitted.diagnostics.value("probe_census", false);
  core->evidence_sampled = probe_sampled;
  for (const auto &rows : inputs)
    core->evidence_sampled |= sampled_population(rows);
  const bool evidence_match_incomplete =
      core->input_match_incomplete || probe_sampled;
  const bool output_sampled =
      emitted.diagnostics.value("output_sampled", false);
  core->sampled = core->evidence_sampled || output_sampled;
  core->match_incomplete = evidence_match_incomplete || output_sampled;
  row_weights.register_population(observation->rows);
  population_moments.register_population(observation->rows);
  std::size_t reached = 1;
  double input_mass = 1;
  for (auto &rows : inputs) {
    reached *= row_count(rows);
    input_mass *= mass(rows);
  }
  if (emitted.join_evidence) {
    reached = emitted.join_evidence->prefix;
    input_mass = emitted.join_evidence->input_mass;
  } else if (emitted.probes) {
    reached = row_count(*emitted.probes);
    input_mass = mass(*emitted.probes);
  }
  bool represented = true;
  bool census = observation->complete && !core->evidence_sampled &&
                !core->input_match_incomplete &&
                (!core->full_join_input || !output_sampled);
  for (auto &raw : node->relations)
    for (auto &h : relations.at(node_relkey(raw)).strata) {
      represented &= h.selected != 0;
      census &= h.selected == h.population;
    }
  const auto observed_mass =
      emitted.diagnostics.find("observed_join_output_mass");
  const auto output_mass = observed_mass == emitted.diagnostics.end()
                               ? mass(*observation->rows)
                               : observed_mass->get<double>();
  Json selectivity =
      reached && observation->complete && represented && input_mass
          ? Json(output_mass / input_mass)
          : Json(nullptr);
  const auto sample_cap = positive("max_operator_sample_rows", 50);
  const bool capped_zero = semantic && selectivity == 0 &&
                           reached >= sample_cap &&
                           kind.find("PROJECTION") == std::string::npos;
  if (capped_zero)
    selectivity = 1. / sample_cap;
  Json expected_input = 1.;
  for (auto &size : sizes)
    if (size.is_null()) {
      expected_input = nullptr;
      break;
    } else
      expected_input = expected_input.get<double>() * size.get<double>();
  Json expected_output = !expected_input.is_null() && !selectivity.is_null()
                             ? Json(expected_input.get<double>() *
                                    selectivity.get<double>())
                             : Json(nullptr),
       relational = Json::object();
  // A finite zero remains an empirical point estimate. Sampling the next
  // operator still uses its independent base cohort, even when estimated
  // cardinality becomes zero.
  const bool finite_zero = selectivity == 0 && !census;
  if (!semantic) {
    Json prior = profile_.contains("node_statistics") &&
                         profile_["node_statistics"].contains(nid)
                     ? profile_["node_statistics"][nid]
                     : Json::object();
    const auto kind = op.value("kind", op.value("type", std::string()));
    const bool algebraic =
        kind == "PROJECTION" || kind == "CROSS_PRODUCT" ||
        kind == "ORDER_BY" || kind == "CTE_SCAN" ||
        op.contains("sampling_passthrough_input") ||
        op.value("scalar_aggregate", false) ||
        (kind == "LIMIT" &&
         finite_number(op.value("limit", Json(nullptr))));
    if (op.value("sampling_cardinality_only", false)) {
      prior["cardinality_basis"] = "DuckDB logical prior; correlated "
                                   "terminal stage not materialized";
      prior["sampling_cardinality_only"] = true;
    } else if (backend && input_complete && !deadline_reached() &&
               !algebraic) {
      PhaseTimer timer(phase_metrics, "relational_estimate", trace_timings,
                       true);
      const auto source = backend;
      prior.update(sql_pipeline.run(
          [&] { return source->estimate_relation(op, inputs); }));
      harvest();
    }
    prior.erase("sampled_selectivity");
    prior.erase("sampled_selectivity_complete");
    relational = propagate(op, prior, sizes);
    selectivity = relational.value("selectivity", Json(nullptr));
    expected_output = relational.value("output_rows", Json(nullptr));
  }
  if (expected_input == 0 && input_complete && observation->complete &&
      !op.contains("sampling_passthrough_input") &&
      op.value("join_type", std::string()) != "LEFT" &&
      op.value("join_type", std::string()) != "MARK")
    expected_output = op.value("scalar_aggregate", false) ? 1. : 0.;
  Json field_info = Json::object(), widths = Json::array();
  OwnedInputs input_boundaries;
  double width = 0;
  bool width_known = true;
  for (std::size_t side = 0; side < inputs.size(); ++side) {
    auto owned = std::make_shared<const Json>(boundary(inputs[side]));
    input_boundaries.push_back(std::move(owned));
    const auto &b = *input_boundaries.back();
    widths.push_back(b[0]);
    field_info.update(b[1]);
    if (b[0].is_null())
      width_known = false;
    else
      width += b[0].get<double>();
  }
  observation->boundary =
      std::make_shared<const Json>(boundary(*observation->rows));
  double eff = emitted.join_evidence ? emitted.join_evidence->effective
               : inputs.empty()      ? 0
                                     : effective(inputs[0]);
  if (!emitted.join_evidence && inputs.size() > 1)
    for (auto &rows : inputs)
      eff = std::min(eff, effective(rows));
  if (emitted.probes)
    eff = std::min(eff, effective(*emitted.probes));
  if (semantic && observation->complete)
    eff = selection_effective(op, inputs, eff);
  if (!observation->complete)
    eff = 0.;
  Json info = {
      {"input_width", width_known ? Json(width) : Json(nullptr)},
      {"output_width", (*observation->boundary)[0]},
      {"width_basis", "serialized scalar bytes"},
      {"input_rows", expected_input},
      {"output_rows", expected_output},
      {"sample_input_rows", observation->complete ? reached : 0},
      {"sample_output_rows", row_count(*observation->rows)},
      {"selectivity", selectivity},
      {"effective_samples", eff},
      {"uncertain", !reached || !observation->complete},
      {"sampling_complete", observation->complete},
      {"population_census", census},
      {"finite_zero", finite_zero},
      {"model_budget_exhausted",
       !observation->complete && !deadline_reached()},
      {"survival_match_incomplete", core->input_match_incomplete},
      {"sample_model", semantic ? Json(models(op)[0]) : Json(nullptr)},
      {"input_side_rows", sizes},
      {"input_side_widths", widths},
      {"input_field_statistics", field_info},
      {"output_field_statistics", observation->complete
                                      ? (*observation->boundary)[1]
                                      : Json::object()}};
  if (capped_zero) {
    info["empirical_selectivity"] = 0.;
    info["selectivity_floor_applied"] = true;
    info["selectivity_sample_cap"] = sample_cap;
    info["reliability"]["selectivity"] = 0.;
  }
  if (semantic) {
    info["sample_layout"] = op.at("default_layout");
    info["sample_model_id"] = models(op)[0];
    info["selectivity_scope"] = "operator";
    if (settings.value("selectivity_parallel_models", true) &&
        models(op).size() > 1) {
      info["sample_model"] = nullptr;
      info["sample_model_id"] = nullptr;
      info["sample_model_ids"] = models(op);
      info["selection_model_assignment"] =
          "completion_driven_idle_model_lanes";
    }
  }
  info.update(emitted.diagnostics);
  if (core->full_join_input)
    info["sampling_full_join_input"] = true;
  if (!semantic) {
    for (auto it = relational.begin(); it != relational.end(); ++it)
      if (it.key() != "input_width" && it.key() != "output_width")
        info[it.key()] = it.value();
  } else {
    double chars = 0, left = 0, right = 0;
    bool known = true;
    auto schema = op.value("input_field_schema", Json::array());
    std::map<std::string, std::string> descriptors;
    for (auto &f : schema)
      descriptors[f["name"].get<std::string>()] =
          f.value("value_key", f["name"].get<std::string>());
    Json names = op.value("input_fields", Json::array()),
         left_names = op.value("left_input_fields", Json::array());
    if (names.empty()) {
      names = left_names;
      for (auto &f : op.value("right_input_fields", Json::array()))
        names.push_back(f);
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      std::string name = names[i],
                  key = descriptors.count(name) ? descriptors[name] : name;
      auto length = field_info.value(key, Json::object())
                        .value("average_length", Json(nullptr));
      if (length.is_null()) {
        known = false;
        break;
      }
      chars += length.get<double>();
      if (i < left_names.size())
        left += length.get<double>();
      else
        right += length.get<double>();
    }
    info["prompt_input_characters"] = known ? Json(chars) : Json(nullptr);
    info["left_data_characters"] =
        known && !left_names.empty() ? Json(left) : Json(nullptr);
    info["right_data_characters"] =
        known && !left_names.empty() ? Json(right) : Json(nullptr);
    if (node->children.size() == 1 &&
        op.value("kind", std::string()).find("JOIN") != std::string::npos)
      info["pair_input"] = true;
  }
  info["selectivity_precision"] =
      !capped_zero &&
      (semantic || join_operator(op, node->children.size())) &&
      !selectivity.is_null() && (census || eff >= min_depth_samples);
  if (semantic && observation->complete &&
      (sample_depth_selectivity || ensure_prefix_depth_rates)) {
    if (reached)
      directly_sampled_operators.insert(oid);
    auto survival =
        emitted.join_evidence
            ? source_depths(node, emitted.join_evidence->arrived, nullptr,
                            input_complete, !core->evidence_sampled,
                            core->input_match_incomplete)
            : depths(node, inputs, std::nullopt, input_complete,
                     !core->evidence_sampled, core->input_match_incomplete);
    auto eligible =
        emitted.probes ? SampleInputs{emitted.probes.get()} : inputs;
    auto passed =
        emitted.join_evidence
            ? source_depths(node, emitted.join_evidence->passed,
                            &emitted.join_evidence->probed,
                            observation->complete, !core->evidence_sampled,
                            evidence_match_incomplete)
            : depths(node,
                     {emitted.statistical_rows
                          ? emitted.statistical_rows.get()
                          : observation->rows.get()},
                     eligible, observation->complete,
                     !core->evidence_sampled, evidence_match_incomplete);
    bool precise = !survival.empty();
    for (auto it = survival.begin(); it != survival.end(); ++it)
      precise &= it.value()["0"]["reliable"].get<bool>();
    info["input_survival_precision"] = precise;
    if (inputs.size() == 2) {
      for (std::size_t side = 0; side < 2; ++side) {
        auto child = std::find_if(nodes.begin(), nodes.end(), [&](auto &n) {
          return n->node_id == node->children[side];
        });
        auto relation = node_relkey(*(*child)->relations.begin());
        Json value = nullptr;
        for (auto it = passed.begin(); it != passed.end(); ++it)
          if (it.key().rfind(relation + ":", 0) == 0) {
            value = it.value()["0"]["rate"];
            break;
          }
        info[side ? "right_selectivity" : "left_selectivity"] = value;
      }
      info["depth_basis"] =
          (emitted.probes || emitted.join_evidence)
              ? "distinct-row match rate among evaluated probes; unprobed "
                "pairs excluded"
              : "distinct-row match rate against sampled opposite side";
    }
    info["depth_survival_statistics"] = std::move(survival);
    info["depth_selectivity_statistics"] = std::move(passed);
  }
  core->inputs = std::move(owned_inputs);
  if (semantic && (sample_depth_selectivity || ensure_prefix_depth_rates)) {
    auto expression = expression_depths(op, *core);
    auto &bank = info["depth_selectivity_statistics"];
    if (bank.is_null())
      bank = Json::object();
    bank.update(expression);
    core->expression_statistics =
        std::make_shared<const Json>(std::move(expression));
  }
  estimate_survivals(node, op, data, info, *core);
  if (semantic && sample_depth_selectivity) {
    core->statistics = std::make_shared<const Json>(info);
    info.update(cache_inputs(op, action, inputs, input_complete,
                             *observation, &input_boundaries));
  }
  observation->statistics = SharedStatistics(std::move(info));
  core->input_boundaries = std::move(input_boundaries);
}
}
