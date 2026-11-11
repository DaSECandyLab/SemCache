#include "detail.hpp"
#include "join/join_candidates.hpp"
#include "survival/estimate_fields.hpp"
#include <numeric>
#include <unordered_set>

namespace semcache::sampling {
const Json &NativeSampler::Impl::action_for(const StatePtr &state,
                                            Id oid) const {
  for (auto s = state; s && s->parent; s = s->parent)
    if (s->action.operator_id == oid) {
      auto found = state_actions.find(s->handle_id);
      if (found == state_actions.end())
        found = state_actions.emplace(s->handle_id, Json::parse(s->action_json))
                    .first;
      return found->second;
    }
  static const Json empty = Json::object();
  return empty;
}

std::string NativeSampler::Impl::state_token(const NodePtr &node,
                                             const StateData &data) const {
  PhaseTimer timer(phase_metrics, "observation_token", trace_timings, true);
  // Configuration/prior/operator banks are immutable within this sampler.
  // Owned node handles identify the action/layout, while versions identify
  // mutable admission cohorts and immutable child observation publication.
  std::string token = std::to_string(node->handle_id);
  for (auto &raw : node->relations) {
    auto key = node_relkey(raw);
    token += '|' + key + ':' + std::to_string(relations.at(key).version);
  }
  for (auto child : node->children)
    token += '|' + std::to_string(data.observations.at(child)->serial);
  return token;
}

std::string NativeSampler::Impl::core_token(const NodePtr &node,
                                            const StateData &data) const {
  PhaseTimer timer(phase_metrics, "physical_core_token", trace_timings, true);
  std::string token = node->operator_id
                          ? "op" + std::to_string(*node->operator_id)
                          : "scan" + std::to_string(node->node_id);
  for (auto &raw : node->relations) {
    auto key = node_relkey(raw);
    token += '|' + key + ':' + std::to_string(relations.at(key).version);
  }
  for (auto child : node->children)
    token += '|' + std::to_string(data.observations.at(child)->core->serial);
  for (auto field : node->fields)
    token += 'f' + std::to_string(field);
  for (auto &side : node->pair_columns) {
    token += 'p';
    for (auto field : side)
      token += std::to_string(field) + ',';
  }
  return token;
}

namespace {
struct ProjectionSlice {
  Json op;
  std::set<std::string> dependencies;
};
ProjectionSlice projection_slice(const Json &description,
                                 const std::vector<Json> &outputs,
                                 const std::set<std::string> &available,
                                 const std::set<std::string> &required) {
  ProjectionSlice result{description, {}};
  const auto sql =
      description.value("sql", description.value("sampling_sql", ""));
  // These templates are emitted by the bound DuckDB AST exporter. Split only
  // their SELECT list; quotes and nesting protect commas inside expressions.
  std::vector<std::string> expressions;
  std::size_t start = 7, from = std::string::npos;
  char quote = 0;
  std::size_t nesting = 0;
  if (sql.rfind("SELECT ", 0) != 0)
    throw std::invalid_argument(
        "Bound projection sampling requires SELECT SQL");
  for (std::size_t i = start; i < sql.size(); ++i) {
    const auto c = sql[i];
    if (quote) {
      if (c == quote) {
        if (i + 1 < sql.size() && sql[i + 1] == quote)
          ++i;
        else
          quote = 0;
      }
      continue;
    }
    if (c == '\'' || c == '"') {
      quote = c;
      continue;
    }
    if (c == '(' || c == '[' || c == '{')
      ++nesting;
    else if (c == ')' || c == ']' || c == '}')
      --nesting;
    else if (!nesting && c == ',') {
      expressions.push_back(sql.substr(start, i - start));
      start = i + 1;
    } else if (!nesting && sql.compare(i, 6, " FROM ") == 0) {
      expressions.push_back(sql.substr(start, i - start));
      from = i;
      break;
    }
  }
  if (from == std::string::npos || expressions.size() != outputs.size() + 1)
    throw std::invalid_argument(
        "Bound projection sampling schema differs from SELECT outputs");
  std::string selected = "SELECT " + expressions.front();
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (!required.count(sample_column(outputs[i])) &&
        !required.count(field_key(outputs[i])))
      continue;
    const auto &expression = expressions[i + 1];
    selected += ',' + expression;
    // Only source binding identifiers before the emitted final AS alias are
    // dependencies. Literal text and unselected generated outputs add none.
    const auto alias = expression.rfind(" AS ");
    const auto end = alias == std::string::npos ? expression.size() : alias;
    for (std::size_t pos = 0; pos < end; ++pos) {
      if (expression[pos] == '\'') {
        for (++pos; pos < end; ++pos)
          if (expression[pos] == '\'') {
            if (pos + 1 < end && expression[pos + 1] == '\'')
              ++pos;
            else
              break;
          }
      } else if (expression[pos] == '"') {
        std::string name;
        for (++pos; pos < end; ++pos) {
          if (expression[pos] == '"') {
            if (pos + 1 < end && expression[pos + 1] == '"') {
              name += '"';
              ++pos;
            } else
              break;
          } else
            name += expression[pos];
        }
        if (available.count(name))
          result.dependencies.insert(std::move(name));
      }
    }
  }
  selected += sql.substr(from);
  result.op["sql"] = selected;
  result.op["sampling_sql"] = selected;
  result.op["required_columns"] = result.dependencies;
  return result;
}
}

std::set<std::string>
NativeSampler::Impl::independent_dependencies(const Json &op,
                                              const NodePtr &node) const {
  std::set<std::string> required;
  for (const auto &field : op.value("required_columns", Json::array())) {
    required.insert(sample_column(field));
    required.insert(field_key(field));
  }
  for (const auto &field : input_keys(op))
    required.insert(field);
  // A mapping with no explicit expressions still needs its output bindings.
  // An empty dependency set otherwise expands every unrelated source factor.
  const auto kind = op.value("kind", op.value("type", std::string()));
  if (required.empty() && (kind == "PROJECTION" || kind == "CTE_SCAN" ||
                           op.contains("sampling_passthrough_input")))
    for (const auto field : node->fields) {
      const auto &output = imported.at("fields").at(field);
      required.insert(sample_column(output));
      required.insert(field_key(output));
    }
  return required;
}

OwnedInputs
NativeSampler::Impl::bound_independent_inputs(const Json &op,
                                              const OwnedInputs &inputs) {
  if (inputs.empty())
    return inputs;
  const bool semantic = semantic_operator(op);
  const auto limit = semantic ? initial_size : relational_sample_rows;
  // Final source rows have already been drawn by SQL. Binary consumers pair
  // these coordinates directly rather than redrawing smaller side pools.
  if (direct_sampling)
    return inputs;
  // Binary semantic joins sample their pair domain in sample_join. Preparing
  // their source sides here would generate unused argument/projection work.
  if (semantic && inputs.size() > 1)
    return inputs;
  std::vector<std::size_t> targets;
  for (const auto &input : inputs)
    targets.push_back(row_count(*input));
  if (inputs.size() == 1)
    targets[0] = std::min(targets[0], limit);
  else {
    // Bound the Cartesian INPUT domain, not just each side independently:
    // two pools of 200 rows must never cause 40,000 relational candidates.
    // Grow the smaller sides first so a short side can use its full support.
    std::vector<std::size_t> order(inputs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto a, auto b) { return targets[a] < targets[b]; });
    std::size_t remaining = limit;
    for (std::size_t i = 0; i < order.size(); ++i) {
      auto &target = targets[order[i]];
      const auto sides = order.size() - i;
      auto balanced = sides == 1
                          ? remaining
                          : static_cast<std::size_t>(std::pow(
                                static_cast<double>(remaining), 1. / sides));
      balanced = std::max(std::size_t(1), balanced);
      auto fits = [&](std::size_t base) {
        std::size_t power = 1;
        for (std::size_t n = 0; n < sides; ++n) {
          if (base > remaining / power)
            return false;
          power *= base;
        }
        return true;
      };
      // Floating-point roots may round upward at integer boundaries.
      while (!fits(balanced))
        --balanced;
      target = std::min(target, balanced);
      if (target)
        remaining /= target;
    }
  }
  bool bounded = false;
  for (std::size_t side = 0; side < inputs.size(); ++side)
    bounded |= row_count(*inputs[side]) > targets[side];
  if (!bounded)
    return inputs;
  // A unary task may bind several source relations. Draw bounded source
  // tuples lazily; a factorized leaf is still subject to the semantic cap.
  auto domain_identity = [&](auto &&self, const Json &rows) -> Json {
    if (product(rows))
      return Json{
          {"product", Json::array({self(self, rows.at("product")[0]),
                                   self(self, rows.at("product")[1])})}};
    Json ids = Json::array();
    for (const auto &row : rows)
      ids.push_back(row.at("id"));
    return ids;
  };
  Json domains = Json::array();
  for (const auto &input : inputs)
    domains.push_back(domain_identity(domain_identity, *input));
  const auto seed = std::stoull(
      digest(Json::array({settings.value("seed", Json(42)), domains}))
          .substr(0, 16),
      nullptr, 16);
  OwnedInputs result;
  for (std::size_t side = 0; side < inputs.size(); ++side) {
    if (row_count(*inputs[side]) <= targets[side])
      result.push_back(inputs[side]);
    else {
      resident_detail::JoinOutputStream stream;
      result.push_back(std::make_shared<const Json>(
          stream.take(*inputs[side], targets[side], seed + side)));
    }
  }
  return result;
}

SharedSamples NativeSampler::Impl::independent_population(
    const StatePtr &state, const NodePtr &node,
    const std::set<std::string> &required, bool &complete,
    std::size_t source_limit) {
  if (!source_limit)
    source_limit = relational_sample_rows;
  // A sampling domain follows data dependencies, never predicate survivors.
  // Keep each child's original binding so binary semantic joins still receive
  // the same left/right fields even when their ancestors are skipped.
  if (!node->operator_id) {
    const auto key = node_relkey(*node->relations.begin());
    if (!required.empty() &&
        profile_.value("cardinality_only_sources", Json::object())
            .contains(key))
      throw std::invalid_argument(
          "Correlated cardinality-only input has no sampled field values");
    if (direct_sampling) {
      if (sampling_input_owner) {
        const auto local = operator_direct_sources.find(
            {*sampling_input_owner, key, source_limit});
        if (local != operator_direct_sources.end())
          return local->second;
      }
      const auto found = direct_sources.find({key, source_limit});
      if (found == direct_sources.end())
        throw std::invalid_argument("Missing direct base sample cohort");
      return found->second;
    }
    const auto identity = "base:" + key + ':' +
                          std::to_string(relations.at(key).version) + ':' +
                          std::to_string(source_limit);
    auto found = independent_populations.find(identity);
    if (found != independent_populations.end())
      return found->second;
    auto rows = std::make_shared<const Json>(relations.at(key).admitted);
    if (rows->size() > source_limit) {
      // Every semantic operator and generated-field producer uses the same
      // immutable base subcohort. A larger relational pool cannot change the
      // producer's rows when a different consumer requests another column.
      const auto seed = std::stoull(
          digest(Json::array({"independent_base_cohort",
                              settings.value("seed", Json(42)), key}))
              .substr(0, 16),
          nullptr, 16);
      resident_detail::JoinOutputStream stream;
      rows =
          std::make_shared<const Json>(stream.take(*rows, source_limit, seed));
    }
    return independent_populations.emplace(identity, std::move(rows))
        .first->second;
  }
  const auto &description = execution_op(state, node);
  const auto kind = description.value("kind", std::string());
  const auto nodes = state->nodes();
  std::set<std::string> available;
  std::vector<Json> outputs;
  for (const auto child : node->children) {
    const auto found =
        std::find_if(nodes.begin(), nodes.end(),
                     [&](const auto &n) { return n->node_id == child; });
    for (const auto field : (*found)->fields)
      available.insert(sample_column(imported.at("fields").at(field)));
  }
  bool needed = false;
  if (kind.find("PROJECTION") != std::string::npos || kind == "CTE_SCAN") {
    for (const auto field : node->fields) {
      const auto &output = imported.at("fields").at(field);
      outputs.push_back(output);
      if (!available.count(sample_column(output)))
        needed |= required.count(sample_column(output)) ||
                  required.count(field_key(output));
    }
    for (const auto &field :
         description.value("produced_fields", Json::array()))
      needed |= required.count(sample_column(field)) ||
                required.count(field_key(field));
    if (kind == "SEMANTIC_PROJECTION")
      needed |= required.count(output_column(description)) != 0;
  }
  const Json *stage = &description;
  // A generated semantic field owns its own cohort, including when reached
  // through a relational consumer with a larger allowance.
  if (needed && semantic_operator(description))
    source_limit = initial_size;
  std::set<std::string> dependencies = required;
  if (needed && !semantic_operator(description)) {
    auto sliced = projection_slice(description, outputs, available, required);
    dependencies = sliced.dependencies;
    // Stable ownership is required by the native selection signature cache.
    const auto key = digest(sliced.op);
    stage = &independent_projection_ops.try_emplace(key, std::move(sliced.op))
                 .first->second;
  } else if (needed) {
    for (const auto &field :
         description.value("required_columns", Json::array())) {
      dependencies.insert(sample_column(field));
      dependencies.insert(field_key(field));
    }
    for (const auto &field : input_keys(description))
      dependencies.insert(field);
  }
  OwnedInputs inputs;
  std::string identity = std::to_string(*node->operator_id) +
                         ":limit:" + std::to_string(source_limit) + ':';
  if (direct_sampling && sampling_input_owner)
    identity += ":owner:" + std::to_string(*sampling_input_owner) + ':';
  for (const auto &field : dependencies)
    identity += std::to_string(field.size()) + ':' + field;
  for (std::size_t side = 0; side < node->children.size(); ++side) {
    if (description.contains("sampling_passthrough_input") &&
        side != description.at("sampling_passthrough_input").get<std::size_t>())
      continue;
    const auto child = node->children[side];
    const auto found =
        std::find_if(nodes.begin(), nodes.end(),
                     [&](const auto &n) { return n->node_id == child; });
    inputs.push_back(independent_population(state, *found, dependencies,
                                            complete, source_limit));
  }
  identity += population_versions.key(SampleInputs(inputs));
  identity += needed ? ":projection:" + digest(*stage) : ":base";
  if (auto found = independent_populations.find(identity);
      found != independent_populations.end())
    return found->second;
  SharedSamples result;
  if (needed) {
    const bool semantic = semantic_operator(description);
    inputs = bound_independent_inputs(*stage, inputs);
    if (semantic) {
      inputs = prepare_inputs(*stage, inputs);
    }
    const auto emitted = selection(*stage, inputs, semantic);
    if (!emitted) {
      complete = false;
      return std::make_shared<const Json>(Json::array());
    }
    result = emitted.rows;
  } else {
    if (inputs.empty())
      throw std::logic_error("Independent sampling requires a base relation");
    std::vector<const Json *> factors;
    auto flatten = [&](auto &&self, const Json &rows) -> void {
      if (product(rows)) {
        self(self, rows.at("product")[0]);
        self(self, rows.at("product")[1]);
      } else
        factors.push_back(&rows);
    };
    for (const auto &input : inputs)
      flatten(flatten, *input);
    auto identity_for = [](const Json &rows) {
      Json identity = Json::array();
      for (const auto &entry : origins(rows))
        identity.push_back(entry.first);
      identity.push_back(columns(rows));
      return canonical(identity);
    };
    std::stable_sort(factors.begin(), factors.end(),
                     [&](const auto *a, const auto *b) {
                       return identity_for(*a) < identity_for(*b);
                     });
    std::vector<const Json *> relevant;
    for (const auto *factor : factors) {
      const auto fields = columns(*factor);
      bool binds = false;
      for (const auto &field : dependencies)
        binds |= fields.count(field) != 0;
      if (binds)
        relevant.push_back(factor);
    }
    if (!relevant.empty())
      factors = std::move(relevant);
    if (direct_sampling && factors.size() > 1) {
      const auto target = sampling_input_owner && operator_sample_sizes.count(
                                                      *sampling_input_owner)
                              ? operator_sample_sizes.at(*sampling_input_owner)
                              : source_limit;
      result = resident_detail::direct_product_samples(factors, target);
    } else if (factors.size() == 1)
      result = std::make_shared<const Json>(*factors.front());
    else {
      Json combined = *factors.front();
      for (std::size_t i = 1; i < factors.size(); ++i)
        combined =
            Json{{"product", Json::array({std::move(combined), *factors[i]})}};
      result = std::make_shared<const Json>(std::move(combined));
    }
  }
  // The cohort and projection labels are immutable; reuse across candidate
  // orders does not re-run an unrelated semantic filter or relational join.
  independent_populations.emplace(std::move(identity), result);
  return result;
}

StateData NativeSampler::Impl::replay(const StatePtr &state) {
  PhaseTimer timer(phase_metrics, "replay", trace_timings);
  row_weights.cohort(cohort_serial);
  population_moments.cohort(cohort_serial);
  if (!survival_cohort || survival_cohort->serial != cohort_serial) {
    PhaseTimer timer(phase_metrics, "survival_cohort", trace_timings, true);
    auto cohort = std::make_shared<resident_detail::AdmissionCohort>();
    cohort->serial = cohort_serial;
    for (const auto &entry : relations) {
      auto &selected = cohort->selected[entry.first];
      for (const auto &stratum : entry.second.strata)
        selected.push_back(stratum.selected);
    }
    survival_cohort = std::move(cohort);
  }
  StateData data;
  data.state = state;
  auto nodes = state->nodes();
  const StateData *previous = nullptr;
  if (auto found = states.find(state->handle_id); found != states.end())
    previous = &found->second;
  auto current = [&](const std::shared_ptr<const CoreObservation> &core) {
    if (!core->complete &&
        (!core->retry || core->retry->requests != ledger.used() ||
         core->retry->expired != deadline_reached()))
      return false;
    return (core->join_sample_key.empty() ||
            join_probe_limits.at(core->join_sample_key) ==
                core->join_probe_limit) &&
           (core->join_stream_key.empty() ||
            join_output_limits.at(core->join_stream_key) ==
                core->join_output_limit);
  };
  if (!previous && state->parent)
    if (auto found = states.find(state->parent->handle_id);
        found != states.end())
      previous = &found->second;
  auto publish = [&](const NodePtr &node,
                     const std::shared_ptr<const Observation> &observation) {
    PhaseTimer timer(phase_metrics, "publish_statistics", trace_timings, true);
    // Cohort evidence replaces intrinsic facts; branch precision/budget
    // annotations survive exactly as in the reference publisher.
    const auto nid = node->node_id;
    auto prior = previous;
    if (prior && prior->stats.count(nid)) {
      data.stats.emplace(nid, prior->stats.at(nid));
      if (prior->observations.at(nid) != observation)
        resident_detail::replace_estimate_fields(
            data.stats.at(nid).write(), observation->statistics.read());
    } else
      data.stats.emplace(nid, observation->statistics);
    data.observations[node->node_id] = observation;
  };
  for (auto &node : nodes) {
    std::string nid = std::to_string(node->node_id);
    auto token = state_token(node, data);
    const auto core_key = core_token(node, data);
    StateClaim physical(*this, physical_claims,
                        state_actor && node == state->added
                            ? std::vector<std::string>{core_key}
                            : std::vector<std::string>{});
    if (auto it = observations.find(token); it != observations.end())
      if (auto existing = it->second.lock()) {
        if (existing->complete && current(existing->core) &&
            estimates_current(node, *existing->core, data)) {
          if (node == state->added)
            state_turn();
          publish(node, existing);
          continue;
        }
      }
    auto observation = std::make_shared<Observation>();
    observation->serial = ++observation_serial;
    observation->token = token;
    if (auto it = cores.find(core_key); it != cores.end())
      if (auto core = it->second.lock(); core && current(core)) {
        if (node == state->added)
          state_turn();
        if (!estimates_current(node, *core, data)) {
          core = refresh_core(node, execution_op(state, node), data, *core,
                              observation->serial);
          cores[core_key] = core;
        }
        if (!core->complete) {
          counters["incomplete_core_hits"] =
              counters.value("incomplete_core_hits", std::size_t(0)) + 1;
        }
        if (node == state->added)
          state_turn();
        observation->core = core;
        observation->rows = core->rows;
        observation->boundary = core->boundary;
        observation->complete = core->complete;
        observation->statistics = core->scalar_statistics;
        if (node->operator_id) {
          const auto &op = execution_op(state, node);
          const auto &action = action_for(state, *node->operator_id);
          if (semantic_operator(op) && sample_depth_selectivity)
            observation->statistics.write().update(
                cache_inputs(op, action, core->inputs, core->input_complete,
                             *observation, &core->input_boundaries));
        }
        publish(node, observation);
        observations[token] = observation;
        continue;
      }
    auto core = std::make_shared<CoreObservation>();
    core->node = node;
    core->cohort = survival_cohort;
    core->serial = observation->serial;
    core->estimate_serial = observation->serial;
    core->evidence = evidence_node(node, data);
    observation->core = core;
    if (!node->operator_id) {
      auto key = node_relkey(*node->relations.begin());
      auto &rel = relations.at(key);
      observation->rows = std::make_shared<const Json>(rel.admitted);
      row_weights.register_population(observation->rows);
      population_moments.register_population(observation->rows);
      observation->boundary =
          std::make_shared<const Json>(boundary(*observation->rows));
      double population = rel.population;
      observation->statistics = SharedStatistics(
          Json{{"output_rows", population},
               {"input_rows", population},
               {"selectivity", 1.},
               {"output_width", (*observation->boundary)[0]},
               {"input_width", (*observation->boundary)[0]},
               {"output_field_statistics", (*observation->boundary)[1]},
               {"effective_samples", effective(*observation->rows)},
               {"cardinality_basis",
                profile_.value("cardinality_only_sources", Json::object())
                        .contains(key)
                    ? "correlated input domain upper bound; no sampled values"
                    : "DuckDB base cardinality; sampled field distribution"}});
    } else {
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
      state_turn();
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
          prior.update(state_backend_call(
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
    core->rows = observation->rows;
    core->boundary = observation->boundary;
    core->complete = observation->complete;
    Json intrinsic = std::move(observation->statistics.write());
    observation->statistics = SharedStatistics(slim(intrinsic));
    core->scalar_statistics = observation->statistics;
    intrinsic.erase("conditional_sketch_plan");
    for (auto *key :
         {"alpha", "right_alpha", "pre", "basis", "left_total_length"}) {
      intrinsic.erase(key);
      if (core->scalar_statistics.read().contains(key))
        core->scalar_statistics.write().erase(key);
    }
    core->statistics = std::make_shared<const Json>(std::move(intrinsic));
    observation->core = core;
    if (!core->complete && !selection_remaining())
      core->retry =
          CoreObservation::RetryContext{ledger.used(), deadline_reached()};
    if (core->complete || core->retry)
      cores[core_key] = core;
    publish(node, observation);
    if (observation->complete)
      observations[token] = observation;
  }
  return data;
}

void NativeSampler::Impl::observe(const StatePtr &state) {
  auto updated = replay(state);
  if (state->parent) {
    auto &latest = updated.stats.at(state->added->node_id).write();
    latest["sampling_strategy"] = "independent_base";
  }
  cache_edges(updated);
  states[state->handle_id] = std::move(updated);
}
}
