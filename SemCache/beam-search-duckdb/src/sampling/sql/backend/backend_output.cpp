#include "backend_internal.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace semcache::sampling {

using namespace sql_detail;

Json NativeSQLBackend::Impl::reconstruct_output(OutputContext &ctx) {
  const auto &op = ctx.op;
  const auto &inputs = ctx.inputs;
  const auto &incoming = ctx.incoming;
  auto &scope = ctx.scope;
  auto &existence = ctx.existence;
  const auto &normalized = ctx.normalized;
  const auto &records = ctx.records;
  const auto &returned = ctx.returned;
  const auto &source_count_columns = ctx.source_count_columns;
  const bool factored = ctx.factored;
  const bool aggregate = ctx.aggregate;
  const bool source_only = ctx.source_only;
  const bool left_semi = ctx.left_semi;
  const bool paired = ctx.paired;
  const bool preserve = ctx.preserve;
  const bool sampled_factors = ctx.sampled_factors;
  const auto &kind = ctx.kind;
  const std::size_t output_limit = ctx.output_limit;
  const std::size_t output_population = ctx.output_population;
  resident_detail::PhaseTimer restoration(phase_metrics, "sql/restore_rows",
                                          trace_timings, true);
  SQLRowLineages lineages(sql_lineage_canonicalizations);
  std::vector<std::map<std::string, const Json *>> sources(incoming.size());
  if (!factored)
    for (std::size_t side = 0; side < incoming.size(); ++side)
      for (const auto &row : incoming[side])
        sources[side].emplace(stage_id(row), &row);
  std::map<std::string, double> pair_factors;
  if (paired)
    for (const auto &row : inputs[0])
      if (row.contains("_sample_weight_factor"))
        pair_factors.emplace(stage_id(row), sample_factor(row));
  bool sampled = false;
  for (const auto &rows : inputs)
    sampled |= sampled_population(rows);
  Json output = Json::array();
  for (const auto &record : records) {
    Json values;
    if (source_only) {
      values = Json::object();
      for (std::size_t col = 0; col < returned.size(); ++col)
        values[returned[col]] = record[col];
    } else {
      if (record[0].is_null())
        throw std::runtime_error(
            "DuckDB returned NULL for a required sample JSON");
      values = Json::parse(record[0].get<std::string>());
    }
    if (sampled_factors) {
      std::size_t leaf = 0;
      values["__sample_id"] = scope.identity(incoming[0], values, leaf);
    }
    Json origin = Json::array(), source_values = Json::object();
    double expansion = 1.;
    bool sampled_row = aggregate && sampled;
    Json origin_probabilities = Json::object();
    if (factored && !aggregate) {
      std::vector<const Json *> lineage_sources;
      lineage_sources.reserve(scope.metadata.size());
      for (std::size_t leaf = 0; leaf < scope.metadata.size(); ++leaf) {
        const auto &key = scope.metadata[leaf];
        auto id = values.value(key, Json());
        if (!id.is_string() ||
            !scope.leaves[leaf]->count(id.get<std::string>()))
          throw std::invalid_argument(
              "Factorized SQL lost a source identity");
        const auto &source = *scope.sources[leaf].at(id.get<std::string>());
        merge_sample_origins(origin_probabilities, source);
        if (!source_count_columns.empty()) {
          const auto copies = std::stoull(
              record.at(returned.size() + leaf).get<std::string>());
          const auto draws = std::min(output_limit, output_population);
          const auto included =
              source_inclusion(output_population, copies, draws);
          const auto previous =
              source.value("_sample_origin_probabilities", Json::object());
          auto &probabilities =
              origin_probabilities["_sample_origin_probabilities"];
          if (!probabilities.is_object())
            probabilities = Json::object();
          for (const auto &origin : source.at("lineage")) {
            const auto key = canonical(origin);
            const auto prior = previous.value(
                key, source.contains("_sample_weight_factor") ? Json(nullptr)
                                                              : Json(1.));
            probabilities[key] =
                prior.is_number() && std::isfinite(prior.get<double>())
                    ? Json(prior.get<double>() * included)
                    : Json(nullptr);
          }
        }
        if (source.contains("_sample_weight_factor")) {
          sampled_row = true;
          expansion *= sample_factor(source);
        }
        lineage_sources.push_back(&source);
        if (source_only) {
          source_values.update(
              scope.leaves[leaf]->at(id.get<std::string>()).at("values"));
          ++normalized_row_copies;
        }
        values.erase(key);
      }
      origin = lineages.many(lineage_sources);
      if (source_only) {
        source_values.update(values);
        values = std::move(source_values);
      }
    }
    if (!preserve)
      for (auto it = values.begin(); it != values.end();) {
        if (it.key().rfind("__semantic_input:", 0) == 0)
          it = values.erase(it);
        else
          ++it;
      }
    std::string identity;
    if (aggregate) {
      if (kind == "DISTINCT")
        values.erase("__sample_id");
      identity = digest(Json::array({op.at("operator_id"), values}));
      values.erase("__sample_id");
    } else if (incoming.size() == 1 || left_semi) {
      auto id = values.value("__sample_id", Json());
      if (!id.is_string() ||
          (!factored && !sources[0].count(id.get<std::string>())))
        throw std::invalid_argument("Unary SQL must preserve __sample_id");
      identity = id.get<std::string>();
      values.erase("__sample_id");
      if (!factored) {
        const auto &source = *sources[0].at(identity);
        if (source_only) {
          values = normalized[0]->at(identity).at("values");
          ++normalized_row_copies;
        }
        merge_sample_origins(origin_probabilities, source);
        origin = source.at("lineage");
        sampled_row = source.contains("_sample_weight_factor");
        expansion = sample_factor(source);
      }
    } else {
      auto left = values.value("__left_sample_id", Json()),
           right = values.value("__right_sample_id", Json());
      if (!left.is_string() || !right.is_string() ||
          (!factored && (!sources[0].count(left.get<std::string>()) ||
                         !sources[1].count(right.get<std::string>()))))
        throw std::invalid_argument(
            "Join SQL must preserve both sample identities");
      if (paired && left != right)
        throw std::invalid_argument(
            "Semantic stage invented a candidate pair");
      identity = paired ? left.get<std::string>()
                        : digest(Json::array({left, right}));
      values.erase("__left_sample_id");
      values.erase("__right_sample_id");
      if (!factored) {
        const auto &a = *sources[0].at(left.get<std::string>()),
                   &b = *sources[1].at(right.get<std::string>());
        if (source_only) {
          values = normalized[0]->at(left.get<std::string>()).at("values");
          values.update(
              normalized[1]->at(right.get<std::string>()).at("values"));
          normalized_row_copies += 2;
        }
        merge_sample_origins(origin_probabilities, a);
        merge_sample_origins(origin_probabilities, b);
        origin = lineages.pair(a, b);
        sampled_row = a.contains("_sample_weight_factor") ||
                      b.contains("_sample_weight_factor");
        expansion = sample_factor(a) * sample_factor(b);
      }
      if (paired && pair_factors.count(identity)) {
        sampled_row = true;
        expansion = pair_factors.at(identity);
      }
    }
    if (!preserve) {
      for (auto it = values.begin(); it != values.end();) {
        if (it.key().rfind("__semantic_input:", 0) == 0) {
          it = values.erase(it);
        } else {
          ++it;
        }
      }
    }
    // The ID-only path restores fields from SQL-normalized input banks.
    // Only expressions/custom stages introduce values requiring conversion.
    if (!source_only)
      for (auto it = values.begin(); it != values.end(); ++it)
        it.value() = value(it.key(), std::move(it.value()));
    Json row{{"id", identity},
             {"values", std::move(values)},
             {"lineage", std::move(origin)}};
    if (kind == "CTE_SCAN")
      row["id"] =
          digest(Json::array({"cte-scan", op.at("operator_id"), identity}));
    if (sampled_row)
      sample_factor(row, expansion);
    merge_sample_origins(row, origin_probabilities);
    if (output_limit) {
      row["_sample_population_rows"] = output_population;
      if (output_population > records.size()) {
        sample_factor(row, sample_factor(row) * double(output_population) /
                               double(records.size()));
        if (source_count_columns.empty()) {
          Json probabilities = Json::object();
          for (const auto &source : row.at("lineage"))
            probabilities[canonical(source)] = nullptr;
          row["_sample_origin_probabilities"] = std::move(probabilities);
        }
        row["_sample_origin_approximate"] = true;
        row["_sample_match_incomplete"] = true;
      }
    }
    output.push_back(std::move(row));
  }
  if (existence)
    existence->done();
  scope.done();
  return output;
}

Json NativeSQLBackend::estimate_relation(const Json &op,
                                         const SampleInputs &inputs) {
  auto kind = upper(op.value("kind", op.value("type", std::string())));
  auto remember = [&](const std::string &key, const Json &estimate) {
    if (impl->relational_estimates.size() >= 1024) {
      impl->relational_estimates.erase(impl->estimate_order.front());
      impl->estimate_order.pop_front();
    }
    impl->relational_estimates.emplace(key, estimate);
    impl->estimate_order.push_back(key);
  };
  std::function<Json(const Json &)> cardinality =
      [&](const Json &node) -> Json {
    if (node.contains("extra_info") &&
        node["extra_info"].contains("Estimated Cardinality"))
      return std::stod(
          node["extra_info"]["Estimated Cardinality"].get<std::string>());
    auto children = node.value("children", Json::array());
    return children.size() == 1 ? cardinality(children[0]) : Json();
  };
  std::function<bool(const Json &)> proven_empty = [&](const Json &node) {
    const auto name = upper(node.value("name", std::string()));
    if (name == "EMPTY_RESULT")
      return true;
    const auto children = node.value("children", Json::array());
    return children.size() == 1 &&
           (name == "PROJECTION" || name == "FILTER" || name == "ORDER_BY" ||
            name == "LIMIT" || name == "STREAMING_LIMIT" || name == "TOP_N") &&
           proven_empty(children[0]);
  };
  // Preserve real table statistics and equality joins. A filter EXPLAIN over
  // a temporary Cartesian sample has only a generic predicate prior, which
  // must not be extrapolated to the original tables' entire pair population.
  if (op.contains("population_estimate_sql") &&
      op.contains("population_input_sql")) {
    const auto &input_sql = op.at("population_input_sql");
    if (!input_sql.is_array() || input_sql.empty())
      throw std::invalid_argument("Population estimate requires input SQL");
    const auto sql = op.at("population_estimate_sql").get<std::string>();
    const auto key =
        digest(Json::array({"population-estimate-v1", sql, input_sql}));
    if (auto found = impl->relational_estimates.find(key);
        found != impl->relational_estimates.end()) {
      ++impl->relational_estimate_hits;
      return found->second;
    }
    bool rounded_zero = false;
    auto explain = [&](const std::string &statement) -> Json {
      const auto cache_key =
          digest(Json::array({"population-cardinality-v1", statement}));
      if (auto found = impl->relational_estimates.find(cache_key);
          found != impl->relational_estimates.end()) {
        ++impl->relational_estimate_hits;
        rounded_zero |= found->second.value("rounded_zero", false);
        return found->second.at("cardinality");
      }
      ++impl->relational_explains;
      const auto records = impl->query("EXPLAIN (FORMAT JSON) " + statement);
      const auto tree = Json::parse(records.at(0).back().get<std::string>());
      auto value =
          proven_empty(tree.at(0)) ? Json(0.) : cardinality(tree.at(0));
      // EXPLAIN prints integer estimates. A rounded zero is not proof that a
      // filtered/joined population is empty; retain a weak one-row estimate.
      const bool floor = value == 0 && !proven_empty(tree.at(0));
      if (floor)
        value = 1.;
      rounded_zero |= floor;
      remember(cache_key,
               Json{{"cardinality", value}, {"rounded_zero", floor}});
      return value;
    };
    Json sides = Json::array();
    double population_input = 1.;
    bool known = true;
    for (const auto &statement : input_sql) {
      const auto size = explain(statement.get<std::string>());
      sides.push_back(size);
      known &= size.is_number() && size.get<double>() >= 0. &&
               std::isfinite(size.get<double>());
      if (known)
        population_input *= size.get<double>();
    }
    known &= std::isfinite(population_input);
    const auto output = explain(sql);
    Json rate = known && population_input > 0. && output.is_number()
                    ? Json(output.get<double>() / population_input)
                    : Json();
    const bool contracting =
        kind == "FILTER" || kind == "DISTINCT" ||
        (kind == "AGGREGATE" && !op.value("scalar_aggregate", false));
    const bool capped =
        contracting && rate.is_number() && rate.get<double>() > 1.;
    if (capped)
      rate = 1.;
    Json estimate = {{"input_rows", known ? Json(population_input) : Json()},
                     {"output_rows", capped ? Json(population_input) : output},
                     {"selectivity", rate},
                     {"population_input_rows", sides},
                     {"population_output_rows", output},
                     {"population_rounded_zero", rounded_zero},
                     {"population_contraction_capped", capped},
                     {"uncertain", rounded_zero || capped},
                     {"cardinality_basis",
                      "DuckDB EXPLAIN on real-table relational context"}};
    remember(key, estimate);
    return estimate;
  }
  double incoming = 1;
  for (const auto &rows : inputs)
    incoming *= row_count(rows);
  if (!incoming && kind == "FILTER")
    return {
        {"input_rows", 0.},
        {"output_rows", nullptr},
        {"selectivity", nullptr},
        {"cardinality_basis", "empty conditional sample; selectivity unknown"}};
  auto sql_template =
      inputs.size() == 2 && kind == "FILTER"
          ? op.value("candidate_sql", std::string())
          : op.value("sql", op.value("sampling_sql", std::string()));
  auto key = digest(
      Json::array({sql_template, kind, op.value("input_columns", Json::array()),
                   impl->types, impl->payloads.fingerprints(inputs)}));
  auto cached = impl->relational_estimates.find(key);
  if (cached != impl->relational_estimates.end()) {
    ++impl->relational_estimate_hits;
    return cached->second;
  }
  Impl::Scope scope(*impl, false);
  bool factored =
      std::any_of(inputs.begin(), inputs.end(),
                  [](const Json &rows) { return rows.is_object(); });
  for (std::size_t side = 0; side < inputs.size(); ++side)
    scope.expressions.push_back(scope.compile(inputs[side], names_for(op, side),
                                              factored, inputs.owner(side)));
  auto sql = impl->stage_sql(op, scope.expressions);
  impl->query("COMMIT; BEGIN TRANSACTION");
  impl->retired.clear();
  impl->committed_transient.clear();
  for (const auto &name : scope.owned)
    if (!impl->table_keys.count(name))
      impl->committed_transient.insert(name);
  for (auto &entry : impl->tables)
    entry.second.committed = true;
  ++impl->relational_explains;
  auto records = impl->query("EXPLAIN (FORMAT JSON) " + sql);
  auto tree = Json::parse(records.at(0).back().get<std::string>());
  auto output = cardinality(tree.at(0));
  // Explicit nulls invalidate a generic logical prior when this bound
  // conditional stage has no usable estimate (notably an empty sample).
  Json estimate{
      {"input_rows", incoming},
      {"output_rows", nullptr},
      {"selectivity", nullptr},
      {"cardinality_basis", "DuckDB EXPLAIN of bound relational stage"}};
  if (!output.is_null()) {
    estimate["output_rows"] = output;
    estimate["selectivity"] =
        incoming ? Json(output.get<double>() / incoming) : Json();
  }
  scope.done();
  remember(key, estimate);
  return estimate;
}

} // namespace semcache::sampling
