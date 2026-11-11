#include <cstdlib>
#include "backend.hpp"
#include "../common/exact_integer.hpp"
#include "../common/row_metadata.hpp"
#include "../statistics/statistics.hpp"
#include "../statistics/relational_estimate.hpp"
#include "../resident/timing.hpp"
#include "../resident/populations.hpp"
#include "duckdb.h"
#include "payloads.hpp"
#include "prepared_arguments.hpp"
#include "row_lineage.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace semcache {
namespace sampling {
namespace {
std::string upper(std::string text) {
  for (auto &c : text)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return text;
}
bool integer_marker(const Json &value) { return exact_integer(value); }
std::string marker_text(const Json &value) {
  auto text = integer_text(value);
  auto start =
      !text.empty() && text[0] == '-' ? std::size_t(1) : std::size_t(0);
  if (start == text.size() ||
      !std::all_of(text.begin() + start, text.end(),
                   [](unsigned char c) { return c >= '0' && c <= '9'; }))
    throw std::invalid_argument("Invalid exact sample integer");
  return text;
}
std::string identifier(const std::string &value) {
  if (value.find('\0') != std::string::npos)
    throw std::invalid_argument("SQL identifier cannot contain NUL");
  std::string result = "\"";
  for (auto c : value) {
    result += c;
    if (c == '"')
      result += c;
  }
  return result + "\"";
}
std::string quote(const std::string &value) {
  if (value.find('\0') != std::string::npos)
    throw std::invalid_argument("SQL statement cannot contain NUL");
  std::string result = "'";
  for (auto c : value) {
    result += c;
    if (c == '\'')
      result += c;
  }
  return result + "'";
}
std::string literal(const Json &value) {
  if (value.is_null())
    return "NULL";
  if (value.is_boolean())
    return value.get<bool>() ? "true" : "false";
  if (integer_marker(value))
    return marker_text(value);
  if (value.is_number()) {
    if (value.is_number_float() && !std::isfinite(value.get<double>()))
      throw std::invalid_argument("Non-finite sample value");
    return value.dump();
  }
  if (value.is_string()) {
    static const char hex[] = "0123456789abcdef";
    std::string encoded;
    for (unsigned char c : value.get_ref<const std::string &>()) {
      encoded += hex[c >> 4];
      encoded += hex[c & 15];
    }
    return "decode(from_hex('" + encoded + "'))";
  }
  std::string result = value.is_array() ? "[" : "{";
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (result.size() > 1)
      result += ',';
    if (value.is_object())
      result += identifier(it.key()) + ':';
    result += literal(it.value());
  }
  return result + (value.is_array() ? "]" : "}");
}
void replace_all(std::string &text, const std::string &from,
                 const std::string &to) {
  for (std::size_t at = 0; (at = text.find(from, at)) != std::string::npos;
       at += to.size())
    text.replace(at, from.size(), to);
}
std::size_t row_count(const Json &rows) {
  if (rows.is_array())
    return rows.size();
  if (!rows.is_object() || !rows.contains("product") ||
      rows.at("product").size() != 2)
    throw std::invalid_argument("Boundary must be rows or a binary product");
  auto left = row_count(rows.at("product")[0]),
       right = row_count(rows.at("product")[1]);
  if (right && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("Cartesian sample population is too large");
  return left * right;
}
std::vector<std::string> columns(const Json &rows) {
  std::vector<std::string> result;
  auto add = [&](const std::string &key) {
    if (std::find(result.begin(), result.end(), key) == result.end())
      result.push_back(key);
  };
  if (rows.is_array()) {
    for (const auto &row : rows)
      for (auto it = row.at("values").begin(); it != row.at("values").end();
           ++it)
        add(it.key());
  } else {
    const auto &parts = rows.at("product");
    for (const auto &key : columns(parts[0]))
      add(key);
    for (const auto &key : columns(parts[1])) {
      if (std::find(result.begin(), result.end(), key) != result.end())
        throw std::invalid_argument(
            "Factorized products require disjoint field names");
      add(key);
    }
  }
  return result;
}
Json merged_lineage(const Json &left, const Json &right,
                    std::size_t &canonicalizations) {
  std::map<std::string, Json> unique;
  for (const auto *side : {&left, &right})
    for (const auto &entry : *side) {
      ++canonicalizations;
      unique.emplace(canonical(entry), entry);
    }
  Json result = Json::array();
  for (const auto &entry : unique)
    result.push_back(entry.second);
  return result;
}
std::vector<std::string> names_for(const Json &op, std::size_t side) {
  if (!op.contains("input_columns") || op["input_columns"].empty())
    return {};
  const auto &names = op["input_columns"];
  if (names[0].is_array()) {
    if (side >= names.size())
      return {};
    return names[side].get<std::vector<std::string>>();
  }
  // Binary stages can share a flat input schema.
  return names.get<std::vector<std::string>>();
}
}

struct NativeSQLBackend::Impl {
  struct API {
    void *library = nullptr;
#define SQL_API(name) decltype(&::name) name = nullptr
    SQL_API(duckdb_vector_size);
    SQL_API(duckdb_create_config);
    SQL_API(duckdb_set_config);
    SQL_API(duckdb_destroy_config);
    SQL_API(duckdb_open_ext);
    SQL_API(duckdb_connect);
    SQL_API(duckdb_disconnect);
    SQL_API(duckdb_close);
    SQL_API(duckdb_query);
    SQL_API(duckdb_destroy_result);
    SQL_API(duckdb_result_error);
    SQL_API(duckdb_column_count);
    SQL_API(duckdb_column_type);
    SQL_API(duckdb_row_count);
    SQL_API(duckdb_value_is_null);
    SQL_API(duckdb_value_string);
    SQL_API(duckdb_free);
    SQL_API(duckdb_result_chunk_count);
    SQL_API(duckdb_result_get_chunk);
    SQL_API(duckdb_destroy_data_chunk);
    SQL_API(duckdb_data_chunk_get_size);
    SQL_API(duckdb_data_chunk_get_vector);
    SQL_API(duckdb_vector_get_validity);
    SQL_API(duckdb_vector_get_data);
    SQL_API(duckdb_string_t_length);
    SQL_API(duckdb_string_t_data);
    SQL_API(duckdb_appender_create_ext);
    SQL_API(duckdb_appender_column_count);
    SQL_API(duckdb_appender_column_type);
    SQL_API(duckdb_get_type_id);
    SQL_API(duckdb_destroy_logical_type);
    SQL_API(duckdb_appender_error);
    SQL_API(duckdb_appender_end_row);
    SQL_API(duckdb_appender_close);
    SQL_API(duckdb_appender_destroy);
    SQL_API(duckdb_append_null);
    SQL_API(duckdb_append_bool);
    SQL_API(duckdb_append_int64);
    SQL_API(duckdb_append_double);
    SQL_API(duckdb_append_varchar_length);
#undef SQL_API
    explicit API(const std::string &path) {
      library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!library)
        throw std::runtime_error(
            std::string("Cannot load exact DuckDB library: ") + dlerror());
      try {
#define LOAD_API(name)                                                         \
  name = reinterpret_cast<decltype(name)>(dlsym(library, #name));              \
  if (!name)                                                                   \
  throw std::runtime_error("Pinned DuckDB C API missing " #name)
        LOAD_API(duckdb_vector_size);
        LOAD_API(duckdb_create_config);
        LOAD_API(duckdb_set_config);
        LOAD_API(duckdb_destroy_config);
        LOAD_API(duckdb_open_ext);
        LOAD_API(duckdb_connect);
        LOAD_API(duckdb_disconnect);
        LOAD_API(duckdb_close);
        LOAD_API(duckdb_query);
        LOAD_API(duckdb_destroy_result);
        LOAD_API(duckdb_result_error);
        LOAD_API(duckdb_column_count);
        LOAD_API(duckdb_column_type);
        LOAD_API(duckdb_row_count);
        LOAD_API(duckdb_value_is_null);
        LOAD_API(duckdb_value_string);
        LOAD_API(duckdb_free);
        LOAD_API(duckdb_result_chunk_count);
        LOAD_API(duckdb_result_get_chunk);
        LOAD_API(duckdb_destroy_data_chunk);
        LOAD_API(duckdb_data_chunk_get_size);
        LOAD_API(duckdb_data_chunk_get_vector);
        LOAD_API(duckdb_vector_get_validity);
        LOAD_API(duckdb_vector_get_data);
        LOAD_API(duckdb_string_t_length);
        LOAD_API(duckdb_string_t_data);
        LOAD_API(duckdb_appender_create_ext);
        LOAD_API(duckdb_appender_column_count);
        LOAD_API(duckdb_appender_column_type);
        LOAD_API(duckdb_get_type_id);
        LOAD_API(duckdb_destroy_logical_type);
        LOAD_API(duckdb_appender_error);
        LOAD_API(duckdb_appender_end_row);
        LOAD_API(duckdb_appender_close);
        LOAD_API(duckdb_appender_destroy);
        LOAD_API(duckdb_append_null);
        LOAD_API(duckdb_append_bool);
        LOAD_API(duckdb_append_int64);
        LOAD_API(duckdb_append_double);
        LOAD_API(duckdb_append_varchar_length);
#undef LOAD_API
      } catch (...) {
        dlclose(library);
        library = nullptr;
        throw;
      }
    }
    ~API() {
      if (library)
        dlclose(library);
    }
  } api;
  duckdb_database database = nullptr;
  duckdb_connection connection = nullptr;
  Json config;
  double stage_seconds = 0.;
  std::optional<std::size_t> timed_input_units;
  Json types, events = Json::array(), observations = Json::array();
  Json selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  // Bounded stages borrow the original row payloads. Only SQL identities are
  // replaced by fixed-width ordinals, immediately at table insertion.
  const std::map<std::string, std::string> *sampling_ids = nullptr;
  std::string sampling_ids_key;
  std::string stage_id(const Json &row) const {
    const auto &id = row.at("id").get_ref<const std::string &>();
    return sampling_ids ? sampling_ids->at(id) : id;
  }
  enum class CellType { Text, Boolean, Floating, Integer, Nested };
  std::unordered_map<std::string, CellType> cell_types;

  static CellType classify_type(const std::string &name) {
    const auto type = upper(name);
    if (type == "BOOLEAN")
      return CellType::Boolean;
    if (type == "FLOAT" || type == "REAL" || type == "DOUBLE")
      return CellType::Floating;
    static const std::set<std::string> integers{
        "TINYINT",  "SMALLINT",  "INTEGER",  "BIGINT",  "HUGEINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"};
    if (integers.count(type))
      return CellType::Integer;
    if (type.rfind("STRUCT", 0) == 0 || type.rfind("LIST", 0) == 0 ||
        type.rfind("MAP", 0) == 0 ||
        (type.size() >= 2 && type.compare(type.size() - 2, 2, "[]") == 0))
      return CellType::Nested;
    return CellType::Text;
  }

  void field_type(const std::string &key, const std::string &type) {
    types[key] = type;
    cell_types.insert_or_assign(key, classify_type(type));
  }
  SQLPayloads payloads;
  std::filesystem::path temporary, telemetry;
  std::streamoff telemetry_offset = 0;
  std::size_t serial = 0, cpu_limit = 1, stage_workers = 0, stage_peak = 1;
  bool scheduler_unrestricted = false;
  bool transaction = false;
  bool trace_timings = false;
  std::map<std::string, resident_detail::PhaseMetric> phase_metrics;
  mutable std::size_t sql_value_normalizations = 0;
  std::size_t sql_lineage_canonicalizations = 0;
  std::unordered_map<std::string, Json> relational_estimates;
  std::deque<std::string> estimate_order;
  std::size_t relational_explains = 0, relational_estimate_hits = 0,
              normalized_leaf_reads = 0, id_only_stages = 0,
              predicate_evaluated_rows = 0, predicate_cache_hits = 0,
              normalized_row_copies = 0, sql_table_hits = 0,
              sql_table_misses = 0;
  std::unordered_map<std::string, bool> predicate_results;
  std::deque<std::string> predicate_order;

  struct Table {
    std::string name;
    std::size_t rows;
    std::shared_ptr<const std::map<std::string, Json>> normalized;
    std::size_t borrowers = 0;
    bool committed = false;
    std::uint64_t used = 0;
  };
  std::map<std::string, Table> tables;
  std::map<std::string, std::string> table_keys;
  std::set<std::string> retired, committed_transient;
  std::size_t cached_rows = 0;
  std::uint64_t use_clock = 0;

  explicit Impl(const Json &config)
      : api((std::filesystem::absolute(config.at("build").get<std::string>()) /
             "src/libduckdb.so")
                .string()),
        config(config), types(config.value("field_types", Json::object())),
        trace_timings(config.value("trace_timings", false)),
        // NativeSQLBackend is the Beam sampling backend. It defaults to the
        // independent-request policy; normal query execution uses the
        // semantic extension setting, whose default remains ordered lanes.
        scheduler_unrestricted(config.value("semantic_unrestricted", true)) {
    const auto parallelism = config.value("parallelism", Json(1));
    if (!parallelism.is_number_integer() || parallelism <= 0 ||
        parallelism > std::numeric_limits<std::int64_t>::max())
      throw std::invalid_argument("Parallelism must be a positive integer");
    const auto workers = std::to_string(parallelism.get<std::int64_t>());
    const auto cpu = config.value("cpu_workers", parallelism);
    if (!cpu.is_number_integer() || cpu <= 0 ||
        (config.contains("cpu_workers") && cpu > 64))
      throw std::invalid_argument("CPU worker count must be between 1 and 64");
    cpu_limit = static_cast<std::size_t>(
        std::min(parallelism.get<std::int64_t>(), cpu.get<std::int64_t>()));
    const auto sql_workers = std::to_string(cpu_limit);
    for (auto it = types.begin(); it != types.end(); ++it)
      cell_types.emplace(it.key(),
                         classify_type(it.value().get<std::string>()));
    char name[] = "/tmp/sem-native-sql-XXXXXX";
    auto folder = mkdtemp(name);
    if (!folder)
      throw std::runtime_error("Cannot create native stage directory");
    temporary = folder;
    telemetry =
        config.contains("telemetry") && config["telemetry"].is_string()
            ? std::filesystem::absolute(config["telemetry"].get<std::string>())
            : temporary / "telemetry.jsonl";
    if (std::filesystem::exists(telemetry))
      telemetry_offset = std::filesystem::file_size(telemetry);
    duckdb_config settings = nullptr;
    char *error = nullptr;
    try {
      if (api.duckdb_create_config(&settings))
        throw std::runtime_error("Cannot create DuckDB configuration");
      for (const auto &setting :
           std::vector<std::pair<std::string, std::string>>{
               {"allow_unsigned_extensions", "true"}, {"threads", sql_workers}})
        if (api.duckdb_set_config(settings, setting.first.c_str(),
                                  setting.second.c_str()))
          throw std::runtime_error("Cannot configure pinned DuckDB");
      std::string path;
      if (config.contains("database") && !config["database"].is_null()) {
        path = std::filesystem::absolute(config["database"].get<std::string>())
                   .string();
        if (api.duckdb_set_config(settings, "access_mode", "READ_ONLY"))
          throw std::runtime_error("Cannot open read-only DuckDB");
      }
      if (api.duckdb_open_ext(path.empty() ? nullptr : path.c_str(), &database,
                              settings, &error))
        throw std::runtime_error(error ? error : "Cannot open DuckDB database");
      if (api.duckdb_connect(database, &connection))
        throw std::runtime_error("Cannot connect to DuckDB");
      api.duckdb_destroy_config(&settings);
      if (error) {
        api.duckdb_free(error);
        error = nullptr;
      }
      auto build =
          std::filesystem::absolute(config.at("build").get<std::string>());
      query("LOAD " +
            quote((build / "extension/semantic/semantic.duckdb_extension")
                      .string()));
      query("SET semantic_models=" + quote(config.at("models").dump()));
      query("SET semantic_parallelism=" + workers);
      query(std::string("SET semantic_unrestricted=") +
            (scheduler_unrestricted ? "true" : "false"));
      if (const auto cache = std::getenv("SEMCACHE_RESPONSE_CACHE"))
        query("SET semantic_response_cache=" + quote(cache));
      query("SET semantic_runtime=''");
      query("SET semantic_telemetry=" + quote(telemetry.string()));
      if (config.contains("setup") &&
          !config["setup"].get<std::string>().empty())
        query(config["setup"].get<std::string>());
      query("BEGIN TRANSACTION");
      transaction = true;
    } catch (...) {
      if (settings)
        api.duckdb_destroy_config(&settings);
      if (error)
        api.duckdb_free(error);
      close(true);
      throw;
    }
  }
  ~Impl() { close(true); }
  void close(bool remove_directory = false) noexcept {
    payloads.clear();
    tables.clear();
    table_keys.clear();
    relational_estimates.clear();
    estimate_order.clear();
    retired.clear();
    committed_transient.clear();
    cached_rows = 0;
    if (connection)
      api.duckdb_disconnect(&connection);
    if (database)
      api.duckdb_close(&database);
    transaction = false;
    if (remove_directory && !temporary.empty()) {
      std::error_code error;
      std::filesystem::remove_all(temporary, error);
    }
  }
  Json query(const std::string &sql) {
    resident_detail::PhaseTimer timer(phase_metrics, "sql/query", trace_timings,
                                      true);
    if (!connection)
      throw std::runtime_error("Native SQL backend is closed");
    if (sql.find('\0') != std::string::npos)
      throw std::invalid_argument("SQL cannot contain NUL");
    duckdb_result result{};
    struct Guard {
      API &api;
      duckdb_result &result;
      ~Guard() { api.duckdb_destroy_result(&result); }
    } guard{api, result};
    if (api.duckdb_query(connection, sql.c_str(), &result)) {
      auto message = api.duckdb_result_error(&result);
      throw std::runtime_error(message ? message : "DuckDB stage failed");
    }
    const auto count = api.duckdb_column_count(&result);
    bool strings = count > 0;
    for (idx_t col = 0; col < count; ++col) {
      strings &= api.duckdb_column_type(&result, col) == DUCKDB_TYPE_VARCHAR;
    }
    Json rows = Json::array();
    auto add = [&](std::vector<Json> &&values) {
      Json row = Json::array();
      for (idx_t col = 0; col < count; ++col) {
        row.push_back(std::move(values[col]));
      }
      rows.push_back(std::move(row));
    };
    if (strings) {
      for (idx_t chunk_index = 0;
           chunk_index < api.duckdb_result_chunk_count(result); ++chunk_index) {
        auto chunk = api.duckdb_result_get_chunk(result, chunk_index);
        if (!chunk)
          throw std::runtime_error("DuckDB lost a materialized chunk");
        struct ChunkGuard {
          API &api;
          duckdb_data_chunk chunk;
          ~ChunkGuard() { api.duckdb_destroy_data_chunk(&chunk); }
        } cleanup{api, chunk};
        const auto size = api.duckdb_data_chunk_get_size(chunk);
        std::vector<std::vector<Json>> values(size, std::vector<Json>(count));
        for (idx_t col = 0; col < count; ++col) {
          auto vector = api.duckdb_data_chunk_get_vector(chunk, col);
          auto validity = api.duckdb_vector_get_validity(vector);
          auto data = static_cast<duckdb_string_t *>(
              api.duckdb_vector_get_data(vector));
          for (idx_t row = 0; row < size; ++row) {
            if (validity && !(validity[row >> 6] & (uint64_t(1) << (row & 63))))
              continue;
            values[row][col] =
                std::string(api.duckdb_string_t_data(&data[row]),
                            api.duckdb_string_t_length(data[row]));
          }
        }
        for (auto &row : values)
          add(std::move(row));
      }
    } else {
      for (idx_t row = 0; row < api.duckdb_row_count(&result); ++row) {
        std::vector<Json> values(count);
        for (idx_t col = 0; col < count; ++col) {
          if (api.duckdb_value_is_null(&result, col, row))
            continue;
          auto value = api.duckdb_value_string(&result, col, row);
          try {
            values[col] = std::string(value.data, value.size);
          } catch (...) {
            api.duckdb_free(value.data);
            throw;
          }
          api.duckdb_free(value.data);
        }
        add(std::move(values));
      }
    }
    return rows;
  }
  Json value(const std::string &key, Json item) const {
    ++sql_value_normalizations;
    if (item.is_null())
      return nullptr;
    if (item.is_string()) {
      const auto found = cell_types.find(key);
      const auto type =
          found == cell_types.end() ? CellType::Text : found->second;
      const auto &text = item.get_ref<const std::string &>();
      if (type == CellType::Boolean)
        return text == "true";
      if (type == CellType::Floating)
        return std::stod(text);
      if (type == CellType::Integer) {
        auto integer = Json::parse(text);
        // SQL HUGEINT retains its exact decimal representation when it
        // exceeds the JSON integer domain, rather than rounding it.
        return integer.is_number_integer() ? integer : make_integer(text);
      }
      if (type == CellType::Nested)
        return Json::parse(text);
    }
    return item;
  }
  void append(const std::string &table, const Json &rows,
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
  std::string table(const Json &rows, const std::vector<std::string> &requested,
                    std::vector<std::string> &owned,
                    const SharedSamples &population = {},
                    std::size_t retained_rows = 0) {
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
  void release(const std::vector<std::string> &owned) {
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
  void rollback(const std::vector<std::string> &owned) noexcept {
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
  // The cached bank contains SQL-normalized payloads only. Every borrower
  // supplies its current lineage, weights and origin probabilities.
  std::shared_ptr<const std::map<std::string, Json>>
  normalized_rows(const std::string &name,
                  const std::set<std::string> *selected_ids = nullptr) {
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
  struct Scope {
    Impl &backend;
    std::vector<std::string> owned, expressions, metadata;
    std::vector<std::shared_ptr<const std::map<std::string, Json>>> leaves;
    std::vector<std::map<std::string, const Json *>> sources;
    bool finished = false;
    bool restore_sources;
    explicit Scope(Impl &backend, bool restore_sources = true)
        : backend(backend), restore_sources(restore_sources) {}
    ~Scope() {
      if (!finished)
        backend.rollback(owned);
    }
    std::string identity(const Json &rows, const Json &selected,
                         std::size_t &leaf) const {
      if (rows.is_array())
        return selected.at(metadata.at(leaf++)).get<std::string>();
      const auto &parts = rows.at("product");
      const auto left = identity(parts[0], selected, leaf);
      const auto right = identity(parts[1], selected, leaf);
      return digest(Json::array({left, right}));
    }
    std::string compile(const Json &rows, const std::vector<std::string> &names,
                        bool track, const SharedSamples &population = {},
                        std::size_t retained_rows = 0) {
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
    void done() {
      backend.release(owned);
      finished = true;
    }
  };
  std::string stage_sql(const Json &op,
                        const std::vector<std::string> &expressions,
                        const std::vector<std::string> &metadata = {}) {
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
  void collect_events() {
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
  Json run(const Json &op, const SampleInputs &inputs, const Json &action,
           bool preserve = false) {
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
};

NativeSQLBackend::NativeSQLBackend(const Json &config)
    : impl(std::make_unique<Impl>(config)) {}
NativeSQLBackend::~NativeSQLBackend() = default;
Json NativeSQLBackend::load(const std::string &relation,
                            const std::string &sql) {
  impl->relational_estimates.clear();
  impl->estimate_order.clear();
  Json output = Json::array();
  for (const auto &record :
       impl->query("SELECT semantic_sample_json(t) FROM (" + sql + ") t")) {
    auto values = Json::parse(record[0].get<std::string>());
    auto id = values.value("__sample_id", Json());
    if (id.is_null())
      throw std::invalid_argument(
          "Base SQL must expose a stable non-NULL __sample_id");
    auto identity = id.is_string() ? id.get<std::string>() : id.dump();
    values.erase("__sample_id");
    for (auto it = values.begin(); it != values.end(); ++it)
      it.value() = impl->value(it.key(), std::move(it.value()));
    output.push_back(
        Json{{"id", identity},
             {"values", std::move(values)},
             {"lineage", Json::array({Json::array({relation, id})})}});
  }
  return output;
}
std::shared_ptr<NativeSQLBackend> NativeSQLBackend::fork_semantic() const {
  Json config = impl->config;
  // Stage SQL can retain macros, types or subqueries from the source catalog
  // and session setup. Preserve that environment; only transport telemetry
  // and connection-local parallelism differ. Workers are reused across layers.
  config.erase("telemetry");
  config["parallelism"] = 1;
  auto result = std::make_shared<NativeSQLBackend>(config);
  return result;
}
std::shared_ptr<NativeSQLBackend>
NativeSQLBackend::fork_calibration(const Json &settings) const {
  Json config = impl->config;
  config.erase("telemetry");
  config["parallelism"] = settings.value("parallelism", std::size_t(1));
  config["cpu_workers"] = settings.value("parallelism", std::size_t(1));
  config["semantic_unrestricted"] = false;
  config["setup"] =
      config.value("setup", std::string()) +
      "; SET semantic_response_cache='{}'; SET semantic_request_reuse=false; "
      "SET semantic_max_concurrent_requests_per_model=" +
      std::to_string(
          settings.value("max_concurrent_requests_per_model", std::size_t(1))) +
      ";";
  return std::make_shared<NativeSQLBackend>(config);
}
std::string
NativeSQLBackend::scheduling_key(const std::string &model) const {
  const auto &entry = impl->config.at("models").at(model);
  // The semantic extension validated this registry when the backend opened.
  // Match ModelRegistry's normalization, including explicit default HTTPS
  // ports and a trailing slash on the optional path prefix.
  static const std::regex pattern(
      "^(http|https)://([A-Za-z0-9][A-Za-z0-9.-]*)(:([0-9]{1,5}))?(/[A-Za-z0-9/_-]*)?$");
  std::smatch match;
  const auto url = entry.at("base_url").get<std::string>();
  if (!std::regex_match(url, match, pattern))
    throw std::invalid_argument("Invalid calibration model endpoint");
  auto prefix = match[5].str();
  if (!prefix.empty() && prefix.back() == '/')
    prefix.pop_back();
  return Json::array({match[1].str() == "https", match[2].str(),
                      match[4].matched ? std::stoi(match[4].str()) : 443,
                      prefix, entry.at("model")})
      .dump();
}
std::size_t NativeSQLBackend::sample_workers_peak() const {
  return impl->stage_peak;
}
Json NativeSQLBackend::execute(const Json &op, const SampleInputs &inputs,
                               const Json &action) {
  const auto started = std::chrono::steady_clock::now();
  const auto first = impl->events.size();
  impl->stage_seconds = 0.;
  impl->timed_input_units.reset();
  impl->selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  auto observe = [&](bool failed) {
    impl->collect_events();
    impl->selection_progress = {
        {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
    Json sizes = Json::array(), events = Json::array();
    for (const auto &rows : inputs)
      sizes.push_back(row_count(rows));
    for (std::size_t i = first; i < impl->events.size(); ++i) {
      const auto &event = impl->events[i];
      events.push_back(event);
      // A retransmission repeats a logical request but is a billed HTTP call,
      // so it counts against the request budget like a first attempt.
      const auto name = event.value("event", std::string());
      if (name == "sampling_llm_submit" || name == "sampling_llm_retransmit")
        impl->selection_progress["requests"] =
            impl->selection_progress["requests"].get<std::size_t>() + 1;
      if (event.value("event", std::string()) == "sampling_stage_progress") {
        impl->selection_progress["completed_inputs"] =
            event.at("completed_inputs");
        impl->selection_progress["budget_exhausted"] =
            event.at("budget_exhausted");
      }
    }
    Json schema = Json::array();
    for (const auto &field : op.value("input_field_schema", Json::array()))
      schema.push_back(
          Json::array({field.at("name"), field.value("type", Json())}));
    Json logical_task = op.contains("task")
                            ? Json(digest(Json::array(
                                  {op.at("kind"), op.at("task"), schema,
                                   op.value("output_schema", Json::array())})))
                            : Json();
    impl->observations.push_back(
        Json{{"operator_id", op.value("operator_id", Json())},
             {"kind", op.value("kind", op.value("type", std::string()))},
             {"task_key", logical_task},
             {"model_id", action.is_null() ? Json() : action.at("model_id")},
             {"inputs", sizes},
             {"seconds", std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count()},
             {"execution_seconds", impl->stage_seconds},
             {"purpose", op.value("calibration_purpose", std::string())},
             {"output_discarded", op.value("calibration_discard_output", false)},
             {"events", events},
             {"failed", failed}});
    if (impl->timed_input_units)
      impl->observations.back()["timed_input_units"] = *impl->timed_input_units;
  };
  try {
    std::map<std::string, std::string> ordinal_ids, original_ids;
    struct IdentityGuard {
      Impl &backend;
      ~IdentityGuard() {
        backend.sampling_ids = nullptr;
        backend.sampling_ids_key.clear();
      }
    } identities{*impl};
    // Accuracy evaluates one ordinary Join pair through two singleton sides.
    // Its only pair already has a known order, so retain that original path
    // and its row-level prompt instead of applying selected-pair ordinals.
    const auto kind = upper(op.value("kind", op.value("type", std::string())));
    const bool singleton_join =
        (kind == "SEMANTIC_JOIN" || kind == "SEMANTIC_NESTED_LOOP_JOIN") &&
        inputs.size() == 2 && inputs[0].is_array() && inputs[1].is_array() &&
        inputs[0].size() == 1 && inputs[1].size() == 1;
    if (!action.is_null() && action.contains("llm_request_limit") &&
        !singleton_join) {
      if (inputs.size() != 1 || !inputs[0].is_array())
        throw std::invalid_argument(
            "Bounded semantic stages require flat ordered inputs");
      std::size_t ordinal = 0;
      Json ordered_ids = Json::array();
      for (const auto &row : inputs[0]) {
        if (!row.at("id").is_string())
          throw std::invalid_argument(
              "Sample identities must be unique non-NULL strings");
        const auto &original = row.at("id").get_ref<const std::string &>();
        auto id = std::to_string(ordinal++);
        id.insert(0, 20 - id.size(), '0');
        if (!ordinal_ids.emplace(original, id).second)
          throw std::invalid_argument(
              "Sample identities must be unique non-NULL strings");
        original_ids.emplace(id, original);
        ordered_ids.push_back(original);
      }
      impl->sampling_ids = &ordinal_ids;
      impl->sampling_ids_key = digest(ordered_ids);
    }
    auto output = impl->run(op, inputs, action,
                            op.value("_preserve_sample_inputs", false));
    if (!original_ids.empty())
      for (auto &row : output)
        row["id"] = original_ids.at(row.at("id").get<std::string>());
    observe(false);
    return output;
  } catch (...) {
    try {
      observe(true);
    } catch (...) {
    }
    throw;
  }
}
std::vector<Json> NativeSQLBackend::prepare_inputs(const Json &op,
                                                   const SampleInputs &inputs) {
  std::vector<Json> output;
  for (std::size_t side = 0; side < inputs.size(); ++side) {
    std::string expressions;
    for (const auto &field : op.value("input_field_schema", Json::array())) {
      auto key = field.value("value_key", std::string());
      if (key.rfind("__semantic_input:", 0) != 0)
        continue;
      bool right = false;
      for (const auto &name : op.value("right_input_fields", Json::array()))
        right |= name == field.at("name");
      if (inputs.size() == 2 && right != (side == 1))
        continue;
      if (!expressions.empty())
        expressions += ',';
      expressions += field.at("sample_expression").get<std::string>() + " AS " +
                     identifier(key);
      impl->field_type(key, field.value("type", std::string("VARCHAR")));
    }
    if (expressions.empty() || !row_count(inputs[side])) {
      output.push_back(inputs[side]);
      continue;
    }
    Json stage{{"operator_id", op.value("operator_id", Json(0))},
               {"kind", "PROJECTION"},
               {"sql", "SELECT *," + expressions + " FROM $INPUT"},
               {"_preserve_sample_inputs", true}};
    output.push_back(impl->run(stage, {&inputs[side]}, nullptr, true));
  }
  return output;
}
Json NativeSQLBackend::join_pairs(const Json &op, const Json &pairs,
                                  const Json &action) {
  impl->selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  auto kind = upper(op.value("kind", std::string()));
  if (action.is_null()) {
    // Evaluate only the already drawn relational pairs. Equal private IDs
    // constrain the trusted binary SQL without forming a Cartesian cohort.
    if (!pairs.is_array())
      throw std::invalid_argument("Selected relational pairs must be an array");
    std::vector<Json> inputs{Json::array(), Json::array()};
    for (const auto &pair : pairs) {
      if (!pair.is_array() || pair.size() != 2)
        throw std::invalid_argument("Candidate pair must contain two rows");
      const auto id = digest(Json::array({pair[0].at("id"), pair[1].at("id")}));
      for (std::size_t side = 0; side < 2; ++side) {
        auto row = pair[side];
        row["id"] = id;
        inputs[side].push_back(std::move(row));
      }
    }
    if (pairs.empty())
      return Json::array();
    auto stage = op;
    auto sql = kind == "FILTER" ? op.value("candidate_sql", std::string())
                                : op.value("sql", op.value("sampling_sql", std::string()));
    const auto left = sql.find(" FROM $LEFT l ");
    const auto right = left == std::string::npos ? std::string::npos
                                                : sql.find(" $RIGHT r ON ", left);
    if (right == std::string::npos)
      throw std::invalid_argument("Direct relational pairs require trusted binary SQL");
    const auto predicate = right + std::string(" $RIGHT r ON ").size();
    sql = sql.substr(0, predicate) + "l.__sample_id=r.__sample_id AND (" +
          sql.substr(predicate) + ")";
    stage[kind == "FILTER" ? "candidate_sql" : "sql"] = std::move(sql);
    return execute(stage, SampleInputs(inputs), nullptr);
  }
  if ((kind != "SEMANTIC_JOIN" && kind != "SEMANTIC_NESTED_LOOP_JOIN") ||
      action.is_null())
    throw std::invalid_argument(
        "Pair batching requires a semantic Join and model");
  if (!pairs.is_array())
    throw std::invalid_argument("Selected pairs must be an array");
  if (pairs.empty())
    return Json::array();
  std::array<Json, 2> sides{{Json::array(), Json::array()}};
  for (const auto &pair : pairs) {
    if (!pair.is_array() || pair.size() != 2)
      throw std::invalid_argument("Candidate pair must contain two rows");
    for (std::size_t side = 0; side < 2; ++side)
      sides[side].push_back(pair[side]);
  }
  auto left = columns(sides[0]), right = columns(sides[1]);
  if (left.empty() || right.empty())
    throw std::invalid_argument("Candidate pairs need both field partitions");
  for (const auto &key : right)
    if (std::find(left.begin(), left.end(), key) != left.end())
      throw std::invalid_argument(
          "Candidate-pair inputs require disjoint field partitions");
  Json rows = Json::array();
  std::set<std::string> identities;
  for (const auto &pair : pairs) {
    auto identity = digest(Json::array({pair[0].at("id"), pair[1].at("id")}));
    if (!identities.insert(identity).second)
      throw std::invalid_argument("Candidate-pair identities must be unique");
    Json values = Json::object();
    for (const auto &key : left)
      values[key] = pair[0].at("values").value(key, Json());
    for (const auto &key : right)
      values[key] = pair[1].at("values").value(key, Json());
    Json row{
        {"id", identity},
        {"values", std::move(values)},
        {"lineage", merged_lineage(pair[0].at("lineage"), pair[1].at("lineage"),
                                   impl->sql_lineage_canonicalizations)}};
    if (pair[0].contains("_sample_weight_factor") ||
        pair[1].contains("_sample_weight_factor"))
      sample_factor(row, sample_factor(pair[0]) * sample_factor(pair[1]));
    merge_sample_origins(row, pair[0]);
    merge_sample_origins(row, pair[1]);
    rows.push_back(std::move(row));
  }
  auto stage = op;
  stage["pair_columns"] = Json::array({left, right});
  auto result = execute(stage, {&rows}, action);
  if (!impl->observations.empty()) {
    auto &observation = impl->observations.back();
    std::array<std::set<std::string>, 2> unique;
    for (const auto &pair : pairs)
      for (std::size_t side = 0; side < 2; ++side)
        unique[side].insert(pair[side].at("id").get<std::string>());
    observation["candidate_pair_count"] = pairs.size();
    observation["source_input_rows"] =
        Json::array({unique[0].size(), unique[1].size()});
  }
  return result;
}
Json NativeSQLBackend::sample_population(const Json &op) {
  const auto &spec = op.at("population_sample");
  const auto started = std::chrono::steady_clock::now();
  const auto records = impl->query(spec.at("probe_sql").get<std::string>());
  if (records.size() != 1 || records.at(0).size() != 4)
    throw std::runtime_error("Population sample requires four aggregate counts");
  std::array<double, 4> counts;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const auto &value = records.at(0).at(i);
    counts[i] = value.is_string() ? std::stod(value.get<std::string>())
                                : value.get<double>();
    if (!std::isfinite(counts[i]) || counts[i] < 0)
      throw std::runtime_error("Invalid relational sample count");
  }
  const auto left = counts[0], right = counts[1], n = counts[2], passed = counts[3];
  if (n > left || (left > 0 && n == 0))
    throw std::runtime_error("Relational population sample is incomplete");
  const bool join = spec.at("kind") == "join", census = n == left;
  const double denominator = n * (join ? right : 1.);
  const bool floored = denominator > 0 && passed == 0 && !census;
  const double point = denominator ? passed / denominator : 0.;
  const double rate = floored ? 1. / denominator : point;
  Json sides = Json::array({left});
  if (join)
    sides.push_back(right);
  Json sampled = {{"input_rows", left * (join ? right : 1.)},
          {"output_rows", left * (join ? right : 1.) * rate},
          {"selectivity", rate},
          {"sampled_selectivity", point},
          {"sampled_selectivity_complete", true},
          {"population_input_rows", sides},
          {"population_output_rows", left * (join ? right : 1.) * rate},
          {"population_sample_rows", n},
          {"population_sample_output_rows", passed},
          {"population_sample_census", census},
          {"population_sample_zero_floor", floored},
          {"population_cardinality_sampling", true},
          {"conditioned_relational_filters", spec.at("conditioned_filters")},
          {"effective_samples", n},
          {"uncertain", !census},
          {"reliability", {{"selectivity", census ? 1. : n / (n + 1.)}}},
          {"population_sampling_seconds", std::chrono::duration<double>(
               std::chrono::steady_clock::now() - started).count()},
          {"cardinality_basis", join
              ? "one-sided reservoir join sample against full peer with ready relational filters"
              : "reservoir sample of base-field relational predicate"}};
  Json estimated = Json::object();
  try {
    if (op.contains("population_estimate_sql") && op.contains("population_input_sql"))
      estimated = estimate_relation(op, SampleInputs{});
  } catch (const std::exception &error) {
    sampled["duckdb_estimate_error"] = error.what();
  }
  return mean_relational_estimate(sampled, estimated);
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
Json NativeSQLBackend::selection_progress() const {
  return impl->selection_progress;
}
const Json &NativeSQLBackend::events() const { return impl->events; }
Json NativeSQLBackend::drain_events() {
  auto result = std::move(impl->events);
  impl->events = Json::array();
  return result;
}
const Json &NativeSQLBackend::observations() const {
  return impl->observations;
}
Json NativeSQLBackend::drain_observations() {
  auto result = std::move(impl->observations);
  impl->observations = Json::array();
  return result;
}
Json NativeSQLBackend::diagnostics() const {
  return {
      {"relational_explains", impl->relational_explains},
      {"relational_estimate_hits", impl->relational_estimate_hits},
      {"normalized_leaf_reads", impl->normalized_leaf_reads},
      {"normalized_row_copies", impl->normalized_row_copies},
      {"sql_value_normalizations", impl->sql_value_normalizations},
      {"sql_lineage_canonicalizations", impl->sql_lineage_canonicalizations},
      {"sql_table_hits", impl->sql_table_hits},
      {"sql_table_misses", impl->sql_table_misses},
      {"sql_payload_measurements", impl->payloads.measurements()},
      {"sql_payload_hits", impl->payloads.hits()},
      {"id_only_stages", impl->id_only_stages},
      {"predicate_evaluated_rows", impl->predicate_evaluated_rows},
      {"predicate_cache_hits", impl->predicate_cache_hits}};
}
Json NativeSQLBackend::timings() const {
  Json result = Json::object();
  for (const auto &entry : impl->phase_metrics)
    result[entry.first] = {{"seconds", entry.second.seconds},
                           {"calls", entry.second.calls},
                           {"spans_ns", entry.second.spans}};
  return result;
}
void NativeSQLBackend::close() {
  try {
    impl->collect_events();
  } catch (...) {
  }
  impl->close(true);
}
}
}
