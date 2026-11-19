#include "../detail.hpp"
#include "../join/join_candidates.hpp"
#include "../survival/estimate_fields.hpp"
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

}
