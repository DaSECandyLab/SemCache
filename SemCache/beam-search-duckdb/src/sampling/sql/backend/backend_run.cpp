#include "backend_internal.hpp"
#include <algorithm>
#include <stdexcept>

namespace semcache::sampling {

using namespace sql_detail;

Json NativeSQLBackend::Impl::run(const Json &op, const SampleInputs &inputs,
                                 const Json &action, bool preserve) {
  // A semantic sample stage targets one model lane, so extra SQL workers
  // cannot overlap its requests. Relational stages use at most one worker
  // per available input DataChunk, bounded by the configured CPU cap.
  std::size_t chunks = 1;
  if (action.is_null()) {
    const auto chunk_size = api.duckdb_vector_size();
    for (const auto &input : inputs) {
      const auto rows = row_count(input);
      chunks = std::max(chunks, rows / chunk_size +
                                    std::size_t(rows % chunk_size != 0));
    }
  }
  const auto requested = std::min(cpu_limit, chunks);
  if (stage_workers != requested) {
    query("SET threads=" + std::to_string(requested));
    stage_workers = requested;
  }
  stage_peak = std::max(stage_peak, requested);

  if (inputs.size() != 1 && inputs.size() != 2)
    throw std::invalid_argument("SQL stage requires one or two boundaries");
  auto kind = upper(op.value("kind", op.value("type", std::string())));
  if (op.value("join_type", std::string()) == "LEFT" &&
      (!op.value("calibration_discard_output", false) ||
       !action.is_null() ||
       op.value("calibration_purpose", std::string()) !=
           "operator_time_calibration"))
    throw std::invalid_argument(
        "LEFT JOIN sample materialization is unsupported; only cardinality "
        "estimates and output-discarding time calibration are supported");
  const bool left_semi =
      inputs.size() == 2 && op.value("join_type", std::string()) == "SEMI";
  bool factored =
      std::any_of(inputs.begin(), inputs.end(),
                  [](const Json &rows) { return rows.is_object(); });
  bool aggregate =
      kind.find("AGGREGATE") != std::string::npos || kind == "DISTINCT";
  const auto output_limit = op.value("sampling_output_limit", std::size_t(0));
  if (output_limit && (kind != "FILTER" || !action.is_null()))
    throw std::invalid_argument(
        "Bounded relational output requires an ordinary filter");
  bool paired = inputs.size() == 1 &&
                kind.find("JOIN") != std::string::npos && !action.is_null();
  auto incoming = inputs;
  std::vector<Json> paired_inputs;
  auto adjusted = op;
  if (paired) {
    if (!op.contains("pair_columns") || op["pair_columns"].size() != 2)
      throw std::invalid_argument(
          "Candidate-pair execution requires disjoint field partitions");
    paired_inputs = {Json::array(), Json::array()};
    std::set<std::string> fields;
    for (const auto &side : op["pair_columns"])
      for (const auto &key : side)
        if (!fields.insert(key.get<std::string>()).second)
          throw std::invalid_argument("Candidate-pair columns overlap");
    // Empty candidate cohorts carry no row schema and need no SQL/model
    // work. Do not mistake absent row values for missing bound arguments.
    if (inputs[0].empty())
      return Json::array();
    // Preparing compound semantic arguments adds temporary values after the
    // relational candidate pair has fixed its base-field partition. Keep
    // those declared arguments on their bound side without accepting any
    // unrelated fields or changing the selected pair identities.
    const auto present = columns(inputs[0]);
    const std::set<std::string> present_fields(present.begin(),
                                               present.end());
    for (const auto &field : op.value("required_columns", Json::array()))
      if (!present_fields.count(resident_detail::sample_column(field)))
        throw std::invalid_argument(
            "Candidate input lost a required source field");
    // Bound projection slices omit unused outputs, whereas a direct base
    // sample may retain its complete schema. Both retain the candidate's
    // original left/right ownership; only absent, unneeded fields may drop.
    fields.clear();
    for (auto &side : adjusted["pair_columns"]) {
      Json retained = Json::array();
      for (const auto &key : side)
        if (present_fields.count(key.get<std::string>())) {
          retained.push_back(key);
          fields.insert(key.get<std::string>());
        }
      side = std::move(retained);
    }
    for (const auto &field : op.value("input_field_schema", Json::array())) {
      const auto key = field.value("value_key", std::string());
      if (key.rfind("__semantic_input:", 0) != 0 || fields.count(key) ||
          std::find(present.begin(), present.end(), key) == present.end())
        continue;
      bool right = false;
      for (const auto &name : op.value("right_input_fields", Json::array()))
        right |= name == field.at("name");
      adjusted["pair_columns"][right ? 1 : 0].push_back(key);
      fields.insert(key);
    }
    for (const auto &row : inputs[0]) {
      const auto &source = row.at("values");
      if (source.size() != fields.size() ||
          std::any_of(fields.begin(), fields.end(),
                      [&](const auto &key) { return !source.contains(key); }))
        throw std::invalid_argument(
            "Candidate input changed its field partition");
      for (std::size_t side = 0; side < 2; ++side) {
        auto values = Json::object();
        for (const auto &key : adjusted["pair_columns"][side])
          values[key.get<std::string>()] =
              row.at("values").at(key.get<std::string>());
        Json part{{"id", row.at("id")},
                  {"values", std::move(values)},
                  {"lineage", row.at("lineage")}};
        merge_sample_origins(part, row);
        paired_inputs[side].push_back(std::move(part));
      }
    }
    auto sql = op.value("sql", op.value("sampling_sql", std::string()));
    const auto trusted = sql.find(" FROM $LEFT l JOIN $RIGHT r ON ");
    if (trusted == std::string::npos)
      throw std::invalid_argument(
          "Candidate pair SQL requires trusted left/right join template");
    auto on = sql.find(" ON ", trusted);
    sql.insert(on + 4, "l.__sample_id=r.__sample_id AND ");
    adjusted["sql"] = sql;
    incoming = SampleInputs(paired_inputs);
    adjusted["input_columns"] = adjusted["pair_columns"];
  }
  Scope scope(*this, !aggregate);
  // The right side of a semi join is an existence test, not an output
  // ancestor. Keep its factor metadata out of restored left-row lineage.
  std::unique_ptr<Scope> existence;
  for (std::size_t side = 0; side < incoming.size(); ++side) {
    if (left_semi && side == 1) {
      existence = std::make_unique<Scope>(*this, false);
      scope.expressions.push_back(
          existence->compile(incoming[side], names_for(adjusted, side), false,
                             incoming.owner(side)));
    } else {
      scope.expressions.push_back(
          scope.compile(incoming[side], names_for(adjusted, side), factored,
                        incoming.owner(side)));
    }
  }
  auto sql = stage_sql(adjusted, scope.expressions, scope.metadata);
  const auto effective_sql =
      incoming.size() == 2 && kind == "FILTER"
          ? adjusted.value("candidate_sql", std::string())
          : adjusted.value("sql",
                           adjusted.value("sampling_sql", std::string()));
  const std::string unary_template = "SELECT * FROM $INPUT";
  const std::string binary_template =
      "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,"
      "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l JOIN "
      "$RIGHT r ON ";
  auto tail = effective_sql.rfind(unary_template, 0) == 0
                  ? effective_sql.substr(unary_template.size())
                  : std::string("invalid");
  bool source_only =
      incoming.size() == 1 &&
      (kind == "FILTER" || kind == "ORDER_BY" || kind == "LIMIT") &&
      (tail.empty() || tail.rfind(" WHERE ", 0) == 0 ||
       tail.rfind(" ORDER BY ", 0) == 0 || tail.rfind(" LIMIT ", 0) == 0);
  if (incoming.size() == 2 &&
      (kind == "FILTER" || kind == "JOIN" || kind == "COMPARISON_JOIN" ||
       kind == "CROSS_PRODUCT") &&
      effective_sql.rfind(binary_template, 0) == 0) {
    std::set<std::string> names;
    source_only = true;
    for (std::size_t side = 0; side < incoming.size(); ++side) {
      auto fields = names_for(adjusted, side);
      auto present = columns(incoming[side]);
      fields.insert(fields.end(), present.begin(), present.end());
      std::set<std::string> side_names(fields.begin(), fields.end());
      for (const auto &field : side_names)
        source_only &= names.insert(upper(field)).second;
    }
  }
  // A compound/custom statement can promote types or replace rows even
  // when its first SELECT uses the trusted full-field spelling.
  const auto upper_sql = upper(effective_sql);
  for (const auto &keyword : {"UNION", "INTERSECT", "EXCEPT", ";"})
    if (upper_sql.find(keyword) != std::string::npos)
      source_only = false;
  std::vector<std::shared_ptr<const std::map<std::string, Json>>> normalized;
  bool row_local = !output_limit && source_only && !factored &&
                   incoming.size() == 1 && kind == "FILTER" &&
                   action.is_null() &&
                   adjusted.value("sampling_row_local", false) &&
                   adjusted.value("sampling_row_local_sql", std::string()) ==
                       effective_sql &&
                   adjusted.contains("sampling_required_fields") &&
                   adjusted["sampling_required_fields"].is_array();
  if (row_local) {
    Json schema = Json::array();
    auto fields = names_for(adjusted, 0);
    auto present = columns(incoming[0]);
    fields.insert(fields.end(), present.begin(), present.end());
    std::set<std::string> complete(fields.begin(), fields.end());
    for (const auto &field : complete)
      schema.push_back(
          Json::array({field, types.value(field, std::string("VARCHAR"))}));
    for (const auto &field : adjusted["sampling_required_fields"])
      row_local &=
          field.is_string() && complete.count(field.get<std::string>());
    if (row_local) {
      timed_input_units = 0;
      const auto binding = digest(Json::array(
          {effective_sql, schema, adjusted["sampling_required_fields"]}));
      std::vector<std::string> keys;
      std::map<std::string, std::string> missing_ids;
      std::map<std::string, bool> decisions;
      std::set<std::string> pending;
      Json misses = Json::array();
      for (const auto &source : incoming[0]) {
        Json required = Json::array();
        for (const auto &field : adjusted["sampling_required_fields"])
          required.push_back(
              source.at("values").value(field.get<std::string>(), Json()));
        // Identical raw values under the same bound schema always produce
        // the same predicate result. Different raw representations may miss
        // this cache but cannot incorrectly share a decision. SQL performs
        // all casts; no full typed payload bank is needed to look up keys.
        auto key = digest(Json::array({binding, required}));
        keys.push_back(key);
        auto found = predicate_results.find(key);
        if (found != predicate_results.end()) {
          ++predicate_cache_hits;
          decisions.emplace(key, found->second);
        } else if (pending.insert(key).second) {
          misses.push_back(source);
          missing_ids.emplace(stage_id(source), key);
          decisions.emplace(key, false);
        }
      }
      if (!misses.empty()) {
        auto name =
            table(misses,
                  std::vector<std::string>(complete.begin(), complete.end()),
                  scope.owned);
        auto miss_sql = stage_sql(adjusted, {name});
        const auto stage_started = std::chrono::steady_clock::now();
        auto records = query("SELECT __sample_id FROM (" + miss_sql + ") t");
        stage_seconds += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - stage_started)
                             .count();
        timed_input_units = misses.size();
        predicate_evaluated_rows += misses.size();
        for (const auto &record : records)
          decisions.at(missing_ids.at(record[0].get<std::string>())) = true;
      }
      std::set<std::string> selected_ids;
      for (std::size_t i = 0; i < incoming[0].size(); ++i)
        if (decisions.at(keys[i]))
          selected_ids.insert(stage_id(incoming[0][i]));
      normalized.push_back(normalized_rows(
          scope.expressions[0], selected_ids.size() == incoming[0].size()
                                    ? nullptr
                                    : &selected_ids));
      Json output = Json::array();
      std::size_t index = 0;
      for (const auto &source : incoming[0]) {
        if (decisions.at(keys[index++])) {
          auto row = normalized[0]->at(stage_id(source));
          ++normalized_row_copies;
          row["lineage"] = source.at("lineage");
          copy_sample_metadata(row, source);
          if (!preserve) {
            for (auto it = row["values"].begin();
                 it != row["values"].end();) {
              if (it.key().rfind("__semantic_input:", 0) == 0) {
                it = row["values"].erase(it);
              } else {
                ++it;
              }
            }
          }
          output.push_back(std::move(row));
        }
      }
      scope.done();
      // Publish only after both execution and restoration have succeeded.
      for (const auto &key : pending) {
        if (predicate_results.size() >= 8192) {
          predicate_results.erase(predicate_order.front());
          predicate_order.pop_front();
        }
        predicate_results.emplace(key, decisions.at(key));
        predicate_order.push_back(key);
      }
      ++id_only_stages;
      return output;
    }
  }
  std::vector<std::string> returned =
      incoming.size() == 1
          ? std::vector<std::string>{"__sample_id"}
          : std::vector<std::string>{"__left_sample_id", "__right_sample_id"};
  returned.insert(returned.end(), scope.metadata.begin(),
                  scope.metadata.end());
  const bool sampled_factors =
      output_limit && factored && source_only && incoming.size() == 1;
  if (sampled_factors)
    returned = scope.metadata;
  std::string selected = "semantic_sample_json(t)";
  if (source_only) {
    selected.clear();
    for (const auto &key : returned) {
      if (!selected.empty())
        selected += ',';
      selected += identifier(key);
    }
  }
  if (source_only)
    ++id_only_stages;
  std::size_t output_population = 0;
  std::vector<std::string> source_count_columns;
  if (output_limit) {
    const auto count = query("SELECT count(*) FROM (" + sql + ") t");
    output_population = std::stoull(count.at(0).at(0).get<std::string>());
    if (sampled_factors && output_population > output_limit) {
      // Count copies before bounding the output. A source that occurs in c
      // of N matching rows has inclusion 1-C(N-c,m)/C(N,m), not m/N.
      // Only integer counts cross the SQL boundary; payloads remain bounded.
      std::string windows;
      for (std::size_t i = 0; i < scope.metadata.size(); ++i) {
        auto name = "__source_count_" + std::to_string(i);
        source_count_columns.push_back(name);
        windows += ",count(*) OVER (PARTITION BY " +
                   identifier(scope.metadata[i]) + ") AS " + identifier(name);
        selected += ',' + identifier(name);
      }
      sql = "SELECT *" + windows + " FROM (" + sql + ") t";
    }
    const auto seed = op.value("sampling_output_seed", std::uint64_t(17));
    // Bound after predicates, before JSON restoration. Stable hash order
    // yields nested identity prefixes when a consumer asks for more rows.
    std::string order;
    for (const auto &field : sampled_factors
                                 ? scope.metadata
                                 : std::vector<std::string>{"__sample_id"}) {
      if (!order.empty())
        order += ',';
      order += identifier(field);
    }
    sql = "SELECT * FROM (" + sql + ") t ORDER BY hash(" + order + ',' +
          std::to_string(seed) + ")," + order + " LIMIT " +
          std::to_string(output_limit);
  }
  bool configured = false;
  Json records;
  try {
    if (!action.is_null()) {
      auto setting = Json{{"model_id", action.at("model_id")},
                          {"layout", action.at("layout")},
                          {"candidate_pairs", paired}};
      if (action.contains("llm_request_limit"))
        setting["llm_request_limit"] = action.at("llm_request_limit");
      query("SET semantic_sampling_action=" + quote(setting.dump()));
      configured = true;
    }
    const auto stage_started = std::chrono::steady_clock::now();
    records = query("SELECT " + selected + " FROM (" + sql + ") t");
    stage_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - stage_started)
                         .count();
    if (configured) {
      query("SET semantic_sampling_action=''");
      configured = false;
    }
  } catch (...) {
    if (configured) {
      try {
        query("SET semantic_sampling_action=''");
      } catch (...) {
      }
    }
    throw;
  }
  if (op.value("calibration_discard_output", false)) {
    if (!action.is_null() ||
        op.value("calibration_purpose", std::string()) !=
            "operator_time_calibration" ||
        op.value("join_type", std::string()) != "LEFT")
      throw std::invalid_argument(
          "Discarded SQL output is restricted to LEFT JOIN time calibration");
    // The SQL and result fetch above have completed and stage_seconds is
    // recorded. No caller of this calibration path consumes survivor rows.
    scope.done();
    return Json::array();
  }
  if (source_only && !factored)
    for (std::size_t side = 0; side < incoming.size(); ++side) {
      std::set<std::string> ids;
      for (const auto &record : records)
        ids.insert(record.at(side).get<std::string>());
      normalized.push_back(normalized_rows(
          scope.expressions[side],
          ids.size() == incoming[side].size() ? nullptr : &ids));
    }
  OutputContext ctx{op, inputs, incoming, scope, existence, normalized, records,
                    returned, source_count_columns, factored, aggregate,
                    source_only, left_semi, paired, preserve, sampled_factors,
                    kind, output_limit, output_population};
  return reconstruct_output(ctx);
}

} // namespace semcache::sampling
