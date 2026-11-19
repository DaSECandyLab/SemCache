#include "backend_internal.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>

namespace semcache::sampling {
namespace sql_detail {

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

} // namespace sql_detail

using namespace sql_detail;

NativeSQLBackend::Impl::API::API(const std::string &path) {
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

NativeSQLBackend::Impl::API::~API() {
    if (library)
      dlclose(library);
  }

NativeSQLBackend::Impl::Impl(const Json &config)
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

NativeSQLBackend::Impl::~Impl() { close(true); }

std::string NativeSQLBackend::Impl::stage_id(const Json &row) const {
  const auto &id = row.at("id").get_ref<const std::string &>();
  return sampling_ids ? sampling_ids->at(id) : id;
}

NativeSQLBackend::Impl::CellType NativeSQLBackend::Impl::classify_type(const std::string &name) {
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

void NativeSQLBackend::Impl::field_type(const std::string &key, const std::string &type) {
  types[key] = type;
  cell_types.insert_or_assign(key, classify_type(type));
}

void NativeSQLBackend::Impl::close(bool remove_directory) noexcept {
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

Json NativeSQLBackend::Impl::query(const std::string &sql) {
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

Json NativeSQLBackend::Impl::value(const std::string &key, Json item) const {
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

} // namespace semcache::sampling
