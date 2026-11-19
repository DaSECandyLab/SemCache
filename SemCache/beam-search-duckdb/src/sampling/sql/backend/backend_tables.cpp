#include "backend_internal.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace semcache::sampling {

using namespace sql_detail;

void NativeSQLBackend::Impl::append(const std::string &table, const Json &rows,
                                    const std::vector<std::string> &names) {
  resident_detail::PhaseTimer timer(phase_metrics, "sql/append",
                                    trace_timings, true);
  duckdb_appender appender = nullptr;
  auto check = [&](duckdb_state state) {
    if (state) {
      const char *error =
          appender ? api.duckdb_appender_error(appender) : nullptr;
      throw std::runtime_error(error ? error : "DuckDB appender failed");
    }
  };
  struct Guard {
    API &api;
    duckdb_appender &appender;
    ~Guard() {
      if (appender)
        api.duckdb_appender_destroy(&appender);
    }
  } guard{api, appender};
  check(api.duckdb_appender_create_ext(connection, "temp", "main",
                                       table.c_str(), &appender));
  std::vector<duckdb_type> kinds;
  for (idx_t col = 0; col < api.duckdb_appender_column_count(appender);
       ++col) {
    auto kind = api.duckdb_appender_column_type(appender, col);
    kinds.push_back(api.duckdb_get_type_id(kind));
    api.duckdb_destroy_logical_type(&kind);
  }
  for (const auto &row : rows) {
    for (std::size_t col = 0; col < names.size(); ++col) {
      Json item = names[col] == "__sample_id"
                      ? Json(stage_id(row))
                      : row.at("values").value(names[col], Json());
      if (item.is_null()) {
        check(api.duckdb_append_null(appender));
        continue;
      }
      if (item.is_boolean()) {
        check(api.duckdb_append_bool(appender, item.get<bool>()));
        continue;
      }
      if (item.is_number_float() && !std::isfinite(item.get<double>()))
        throw std::invalid_argument("Non-finite sample value");
      if (kinds[col] == DUCKDB_TYPE_BOOLEAN &&
          (item.is_number() || integer_marker(item))) {
        const bool nonzero = integer_marker(item) ? marker_text(item) != "0"
                                                  : item.get<double>() != 0;
        check(api.duckdb_append_bool(appender, nonzero));
        continue;
      }
      if (item.is_number_integer() &&
          (!item.is_number_unsigned() ||
           item.get<uint64_t>() <= uint64_t(INT64_MAX))) {
        check(api.duckdb_append_int64(appender, item.get<int64_t>()));
        continue;
      }
      std::string text = integer_marker(item) ? marker_text(item)
                         : item.is_string()   ? item.get<std::string>()
                                              : item.dump();
      if (item.is_number_float()) {
        const auto number =
            item.get<double>() == 0 ? 0.0 : item.get<double>();
        text = Json(number).dump();
        const auto kind = kinds[col];
        const bool decimal_cast =
            kind == DUCKDB_TYPE_VARCHAR || kind == DUCKDB_TYPE_DECIMAL ||
            kind == DUCKDB_TYPE_HUGEINT || kind == DUCKDB_TYPE_UHUGEINT ||
            (kind >= DUCKDB_TYPE_TINYINT && kind <= DUCKDB_TYPE_UBIGINT);
        if (!decimal_cast || text.find_first_of("eE") != std::string::npos) {
          check(api.duckdb_append_double(appender, number));
          continue;
        }
      }
      check(api.duckdb_append_varchar_length(appender, text.data(),
                                             text.size()));
    }
    check(api.duckdb_appender_end_row(appender));
  }
  check(api.duckdb_appender_close(appender));
}

std::string NativeSQLBackend::Impl::table(const Json &rows, const std::vector<std::string> &requested,
                                          std::vector<std::string> &owned,
                                          const SharedSamples &population, std::size_t retained_rows) {
  resident_detail::PhaseTimer timer(phase_metrics, "sql/table", trace_timings,
                                    true);
  if (!rows.is_array())
    throw std::invalid_argument(
        "Materialized sample input must be a row array");
  std::vector<std::string> names{"__sample_id"};
  auto add = [&](const std::string &key) {
    if (std::find(names.begin(), names.end(), key) == names.end())
      names.push_back(key);
  };
  for (const auto &name : requested)
    add(name);
  for (const auto &name : columns(rows))
    add(name);
  Json schema = Json::array();
  for (const auto &name : names)
    schema.push_back(name == "__sample_id"
                         ? "VARCHAR"
                         : types.value(name, std::string("VARCHAR")));
  const auto key = digest(
      Json::array({payloads.fingerprint(rows, population, retained_rows),
                   names, schema, sampling_ids_key}));
  auto found = tables.find(key);
  if (found != tables.end()) {
    ++sql_table_hits;
    ++found->second.borrowers;
    found->second.used = ++use_clock;
    owned.push_back(found->second.name);
    return found->second.name;
  }
  ++sql_table_misses;
  const auto name = "native_input_" + std::to_string(++serial);
  owned.push_back(name);
  std::string definitions;
  for (std::size_t col = 0; col < names.size(); ++col) {
    if (!definitions.empty())
      definitions += ',';
    definitions +=
        identifier(names[col]) + " " + schema[col].get<std::string>();
  }
  query("CREATE TEMP TABLE " + name + " (" + definitions + ")");
  if (!rows.empty()) {
    bool scalar = true;
    std::set<std::string> ids;
    for (const auto &row : rows) {
      if (!row.at("id").is_string() ||
          !ids.insert(row.at("id").get<std::string>()).second)
        throw std::invalid_argument(
            "Sample identities must be unique non-NULL strings");
      for (const auto &item : row.at("values"))
        scalar &= !item.is_structured() || integer_marker(item);
    }
    if (scalar)
      append(name, rows, names);
    else {
      std::string values;
      for (const auto &row : rows) {
        if (!values.empty())
          values += ',';
        values += '(';
        for (std::size_t col = 0; col < names.size(); ++col) {
          if (col)
            values += ',';
          values += literal(names[col] == "__sample_id"
                                ? Json(stage_id(row))
                                : row.at("values").value(names[col], Json()));
        }
        values += ')';
      }
      query("INSERT INTO " + name + " VALUES " + values);
    }
  }
  bool eligible = rows.size() <= 8192;
  while (eligible && !tables.empty() &&
         (tables.size() >= 32 || cached_rows + rows.size() > 8192)) {
    auto victim = tables.end();
    for (auto it = tables.begin(); it != tables.end(); ++it)
      if (!it->second.borrowers &&
          (victim == tables.end() || it->second.used < victim->second.used))
        victim = it;
    if (victim == tables.end()) {
      eligible = false;
      break;
    }
    query("DROP TABLE " + victim->second.name);
    if (victim->second.committed)
      retired.insert(victim->second.name);
    cached_rows -= victim->second.rows;
    table_keys.erase(victim->second.name);
    tables.erase(victim);
  }
  if (eligible) {
    tables.emplace(key, Table{name, rows.size(), {}, 1, false, ++use_clock});
    table_keys.emplace(name, key);
    cached_rows += rows.size();
  }
  return name;
}

void NativeSQLBackend::Impl::release(const std::vector<std::string> &owned) {
  for (const auto &name : owned) {
    auto found = table_keys.find(name);
    if (found != table_keys.end())
      --tables.at(found->second).borrowers;
    else {
      query("DROP TABLE IF EXISTS " + name);
      if (committed_transient.erase(name))
        retired.insert(name);
    }
  }
}

void NativeSQLBackend::Impl::rollback(const std::vector<std::string> &owned) noexcept {
  std::set<std::string> names(owned.begin(), owned.end());
  for (const auto &entry : tables)
    names.insert(entry.second.name);
  names.insert(retired.begin(), retired.end());
  names.insert(committed_transient.begin(), committed_transient.end());
  tables.clear();
  table_keys.clear();
  relational_estimates.clear();
  estimate_order.clear();
  retired.clear();
  committed_transient.clear();
  cached_rows = 0;
  try {
    if (transaction) {
      query("ROLLBACK");
      transaction = false;
    }
    for (const auto &name : names)
      query("DROP TABLE IF EXISTS " + name);
    query("BEGIN TRANSACTION");
    transaction = true;
  } catch (...) {
    close();
  }
}

std::shared_ptr<const std::map<std::string, Json>>
NativeSQLBackend::Impl::normalized_rows(const std::string &name,
                                        const std::set<std::string> *selected_ids) {
  resident_detail::PhaseTimer timer(phase_metrics, "sql/normalized_inputs",
                                    trace_timings, true);
  Table *cached_table = nullptr;
  auto table_key = table_keys.find(name);
  if (table_key != table_keys.end())
    cached_table = &tables.at(table_key->second);
  auto normalized = cached_table ? cached_table->normalized : nullptr;
  const bool covered =
      normalized &&
      (selected_ids ? std::all_of(selected_ids->begin(), selected_ids->end(),
                                  [&](const std::string &id) {
                                    return normalized->count(id);
                                  })
                    : normalized->size() == cached_table->rows);
  if (!covered) {
    auto bank =
        normalized
            ? std::make_shared<std::map<std::string, Json>>(*normalized)
            : std::make_shared<std::map<std::string, Json>>();
    if (selected_ids && selected_ids->empty())
      return bank;
    std::string filter;
    if (selected_ids) {
      for (const auto &id : *selected_ids) {
        if (bank->count(id))
          continue;
        if (!filter.empty())
          filter += ',';
        filter += quote(id);
      }
      filter = " WHERE __sample_id IN (" + filter + ')';
    }
    ++normalized_leaf_reads;
    for (const auto &record : query("SELECT semantic_sample_json(t) FROM " +
                                    name + " t" + filter)) {
      auto values = Json::parse(record[0].get<std::string>());
      auto identity = values.at("__sample_id").get<std::string>();
      if (bank->count(identity))
        continue;
      values.erase("__sample_id");
      for (auto it = values.begin(); it != values.end(); ++it)
        it.value() = value(it.key(), std::move(it.value()));
      bank->emplace(identity,
                    Json{{"id", identity}, {"values", std::move(values)}});
    }
    normalized = bank;
    // Preserve immutable readers. A partial bank is reusable only for IDs
    // it actually contains; a full request also checks the table row count.
    if (cached_table)
      cached_table->normalized = normalized;
  }
  return normalized;
}

NativeSQLBackend::Impl::Scope::Scope(Impl &backend, bool restore_sources)
    : backend(backend), restore_sources(restore_sources) {}

NativeSQLBackend::Impl::Scope::~Scope() {
  if (!finished)
    backend.rollback(owned);
}

std::string NativeSQLBackend::Impl::Scope::identity(const Json &rows, const Json &selected,
                                                    std::size_t &leaf) const {
  if (rows.is_array())
    return selected.at(metadata.at(leaf++)).get<std::string>();
  const auto &parts = rows.at("product");
  const auto left = identity(parts[0], selected, leaf);
  const auto right = identity(parts[1], selected, leaf);
  return digest(Json::array({left, right}));
}

std::string NativeSQLBackend::Impl::Scope::compile(const Json &rows, const std::vector<std::string> &names,
                                                   bool track, const SharedSamples &population,
                                                   std::size_t retained_rows) {
  if (!retained_rows && population)
    retained_rows = PopulationFingerprints::materialized_rows(*population);
  if (rows.is_object()) {
    const auto &parts = rows.at("product");
    if (parts.size() != 2)
      throw std::invalid_argument("Invalid binary sample product");
    const auto present = columns(rows);
    std::vector<std::string> left_names, right_names;
    // An empty factor has no row values from which to recover its schema.
    // Carry missing declared columns into one empty branch. The product
    // remains empty, including for scalar aggregates over this boundary.
    if (!names.empty()) {
      auto *empty = row_count(parts[0]) == 0   ? &left_names
                    : row_count(parts[1]) == 0 ? &right_names
                                               : nullptr;
      if (empty)
        for (const auto &name : names)
          if (std::find(present.begin(), present.end(), name) ==
              present.end())
            empty->push_back(name);
    }
    auto left = compile(parts[0], left_names, true,
                        population ? SharedSamples(population, &parts[0])
                                   : SharedSamples{},
                        retained_rows),
         right = compile(parts[1], right_names, true,
                         population ? SharedSamples(population, &parts[1])
                                    : SharedSamples{},
                         retained_rows);
    return "(SELECT "
           "sha256(semantic_sample_json(list_value(l.__sample_id,r.__"
           "sample_id))) __sample_id,"
           "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM " +
           left + " l CROSS JOIN " + right + " r)";
  }
  auto name = backend.table(rows, names, owned, population, retained_rows);
  if (!track)
    return name;
  auto key = "__factor_leaf_" + std::to_string(leaves.size());
  metadata.push_back(key);
  // Keep the same compiled SQL columns for aggregates, while avoiding a
  // normalized payload bank and source index that no output row consumes.
  leaves.push_back(restore_sources ? backend.normalized_rows(name)
                                   : nullptr);
  sources.emplace_back();
  if (restore_sources)
    for (const auto &source : rows)
      sources.back().emplace(backend.stage_id(source), &source);
  return "(SELECT *,__sample_id AS " + identifier(key) + " FROM " + name +
         ")";
}

void NativeSQLBackend::Impl::Scope::done() {
  backend.release(owned);
  finished = true;
}

std::string NativeSQLBackend::Impl::stage_sql(const Json &op, const std::vector<std::string> &expressions,
                                              const std::vector<std::string> &metadata) {
  auto kind = upper(op.value("kind", op.value("type", std::string())));
  auto sql = expressions.size() == 2 && kind == "FILTER"
                 ? op.value("candidate_sql", std::string())
                 : op.value("sql", op.value("sampling_sql", std::string()));
  if (sql.empty())
    throw std::invalid_argument(
        "Relational sampling requires trusted bound stage SQL");
  sql = prepared_join_arguments(std::move(sql), op, expressions.size() == 2);
  if (expressions.size() == 1) {
    if (sql.find("$INPUT") == std::string::npos)
      throw std::invalid_argument("Unary SQL requires $INPUT");
    replace_all(sql, "$INPUT", expressions[0]);
    if (!metadata.empty() && kind.find("AGGREGATE") == std::string::npos &&
        sql.rfind("SELECT *", 0) != 0) {
      const auto prefix = sql.rfind("SELECT DISTINCT ", 0) == 0
                              ? std::string("SELECT DISTINCT ")
                              : std::string("SELECT ");
      if (sql.rfind(prefix, 0) != 0)
        throw std::invalid_argument(
            "Factorized projection requires trusted SELECT template");
      std::string fields;
      for (const auto &key : metadata) {
        if (!fields.empty())
          fields += ',';
        fields += identifier(key);
      }
      sql.insert(prefix.size(), fields + ',');
    }
  } else {
    if (sql.find("$LEFT") == std::string::npos ||
        sql.find("$RIGHT") == std::string::npos)
      throw std::invalid_argument("Binary SQL requires $LEFT and $RIGHT");
    replace_all(sql, "$LEFT", expressions[0]);
    replace_all(sql, "$RIGHT", expressions[1]);
  }
  return sql;
}

void NativeSQLBackend::Impl::collect_events() {
  resident_detail::PhaseTimer timer(phase_metrics, "sql/collect_events",
                                    trace_timings, true);
  std::ifstream file(telemetry, std::ios::binary);
  if (!file)
    return;
  file.seekg(0, std::ios::end);
  auto end = file.tellg();
  if (end < telemetry_offset)
    telemetry_offset = 0;
  file.seekg(telemetry_offset);
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty())
      events.push_back(Json::parse(line));
  }
  telemetry_offset = end;
}

} // namespace semcache::sampling
