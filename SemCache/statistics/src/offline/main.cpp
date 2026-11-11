#include "duckdb.h"
#include "nlohmann/json.hpp"
#include "semcache/prefix_sketch.h"
#include "configuration.hpp"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
using Json = nlohmann::json;
namespace fs = std::filesystem;
std::string Quote(const std::string &s, char q) {
  std::string out(1, q);
  for (char c : s) {
    out += c;
    if (c == q)
      out += c;
  }
  return out + q;
}
struct Result {
  duckdb_result result{};
  Result(duckdb_connection c, const std::string &sql) {
    if (duckdb_query(c, sql.c_str(), &result)) {
      std::string error = duckdb_result_error(&result);
      duckdb_destroy_result(&result);
      throw std::runtime_error(error);
    }
  }
  ~Result() { duckdb_destroy_result(&result); }
  Result(const Result &) = delete;
  std::string Text(size_t c, size_t r) {
    auto value = duckdb_value_string(&result, c, r);
    std::string s(value.data ? value.data : "", value.size);
    duckdb_free(value.data);
    return s;
  }
};
struct Database {
  duckdb_database db = nullptr;
  duckdb_connection conn = nullptr;
  Database(const std::string &path, size_t threads) {
    duckdb_config config;
    duckdb_create_config(&config);
    duckdb_set_config(config, "access_mode", "READ_ONLY");
    duckdb_set_config(config, "allow_unsigned_extensions", "true");
    duckdb_set_config(config, "threads", std::to_string(threads).c_str());
    char *error = nullptr;
    auto status = duckdb_open_ext(path.c_str(), &db, config, &error);
    duckdb_destroy_config(&config);
    if (status) {
      std::string msg(error ? error : "Cannot open database");
      duckdb_free(error);
      throw std::runtime_error(msg);
    }
    if (duckdb_connect(db, &conn)) {
      duckdb_close(&db);
      throw std::runtime_error("Cannot connect");
    }
  }
  ~Database() {
    duckdb_disconnect(&conn);
    duckdb_close(&db);
  }
  std::string Digest(const std::string &s) {
    Result r(conn, "SELECT sha256(" + Quote(s, '\'') + ")");
    return r.Text(0, 0);
  }
};
Json Stamp(const fs::path &p) {
  struct stat s {};
  if (stat(p.c_str(), &s)) {
    if (errno == ENOENT)
      return nullptr;
    throw std::runtime_error("Cannot stat " + p.string());
  }
  return Json::array(
      {s.st_dev, s.st_ino, s.st_size,
       int64_t(s.st_mtim.tv_sec) * 1000000000 + s.st_mtim.tv_nsec});
}
Json Snapshot(const fs::path &p) {
  return {{"database", p.string()},
          {"file", Stamp(p)},
          {"wal", Stamp(p.string() + ".wal")}};
}
struct TableSource {
  std::unique_ptr<Result> result;
  std::vector<duckdb_data_chunk> chunks;
  std::vector<size_t> starts;
  std::vector<std::vector<duckdb_string_t *>> columns;
  std::vector<std::string> scratch;
  size_t count = 0;
  std::string error;
  TableSource(duckdb_connection conn, const std::string &schema,
              const std::string &table, const std::vector<std::string> &names) {
    std::string sql = "SELECT ";
    for (size_t i = 0; i < names.size(); ++i) {
      if (i)
        sql += ',';
      sql += "semantic_sample_json(struct_pack(v := " + Quote(names[i], '"') +
             "))";
    }
    sql += " FROM " + Quote(schema, '"') + "." + Quote(table, '"');
    result = std::make_unique<Result>(conn, sql);
    const auto n = duckdb_result_chunk_count(result->result);
    chunks.reserve(n);
    starts.reserve(n);
    columns.resize(names.size());
    for (size_t k = 0; k < n; ++k) {
      auto chunk = duckdb_result_get_chunk(result->result, k);
      chunks.push_back(chunk);
      starts.push_back(count);
      count += duckdb_data_chunk_get_size(chunk);
      for (size_t f = 0; f < names.size(); ++f)
        columns[f].push_back(static_cast<duckdb_string_t *>(
            duckdb_vector_get_data(duckdb_data_chunk_get_vector(chunk, f))));
    }
  }
  ~TableSource() {
    for (auto &chunk : chunks)
      duckdb_destroy_data_chunk(&chunk);
  }
  std::string_view Value(size_t f, size_t row) const {
    if (row >= count)
      throw std::out_of_range("Offline row index");
    auto k = std::upper_bound(starts.begin(), starts.end(), row) -
             starts.begin() - 1;
    auto &value = columns.at(f).at(k)[row - starts[k]];
    const char *data = duckdb_string_t_data(&value);
    size_t size = duckdb_string_t_length(value);
    if (size < 6 || std::string_view(data, 5) != "{\"v\":" ||
        data[size - 1] != '}')
      throw std::runtime_error("Unexpected native field serialization");
    return {data + 5, size - 6};
  }
  static int Fetch(size_t f, const size_t *ids, size_t n, const char **strings,
                   size_t *sizes, void *ctx) noexcept {
    auto &s = *static_cast<TableSource *>(ctx);
    try {
      s.scratch.resize(n);
      for (size_t i = 0; i < n; ++i) {
        auto value = s.Value(f, ids[i]);
        if (value.find_first_of("<>") != std::string_view::npos) {
          auto &out = s.scratch[i];
          out.clear();
          for (char c : value) {
            if (c == '<')
              out += "\\u003c";
            else if (c == '>')
              out += "\\u003e";
            else
              out += c;
          }
          strings[i] = out.data();
          sizes[i] = out.size();
        } else {
          strings[i] = value.data();
          sizes[i] = value.size();
        }
      }
      return 0;
    } catch (const std::exception &e) {
      s.error = e.what();
      return 1;
    }
  }
};
void AtomicWrite(const fs::path &path, const Json &value) {
  fs::create_directories(path.parent_path());
  std::string temporary = (path.parent_path() / ".prefix-XXXXXX").string();
  int fd = mkstemp(temporary.data());
  if (fd < 0)
    throw std::runtime_error("Cannot create offline artifact");
  try {
    auto bytes = value.dump();
    size_t done = 0;
    while (done < bytes.size()) {
      auto n = write(fd, bytes.data() + done, bytes.size() - done);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        throw std::runtime_error("Cannot write offline artifact");
      done += n;
    }
    if (fsync(fd))
      throw std::runtime_error("Cannot flush offline artifact");
    close(fd);
    fd = -1;
    fs::rename(temporary, path);
  } catch (...) {
    if (fd >= 0)
      close(fd);
    unlink(temporary.c_str());
    throw;
  }
}
std::set<std::string> BuildDatabase(const fs::path &root, fs::path database,
                                    fs::path output, const Json &settings,
                                    const Json &adaptive, int parallelism,
                                    const std::vector<std::string> &tables) {
  database = fs::canonical(database);
  output = fs::absolute(output);
  auto snapshot = Snapshot(database);
  Database db(database.string(), parallelism);
  const auto build =
      root / settings.at("runtime").at("build").get<std::string>();
  Result load(
      db.conn,
      "LOAD " +
          Quote(
              (build / "extension/semantic/semantic.duckdb_extension").string(),
              '\''));
  Result begin(db.conn, "BEGIN TRANSACTION");
  Result catalog_result(db.conn, "SELECT current_database()");
  const auto catalog = catalog_result.Text(0, 0);
  Result available(
      db.conn,
      "SELECT table_schema,table_name FROM information_schema.tables WHERE "
      "table_type='BASE TABLE' AND table_schema NOT IN "
      "('information_schema','pg_catalog') ORDER BY 1,2");
  std::vector<std::pair<std::string, std::string>> selected;
  for (size_t r = 0; r < duckdb_row_count(&available.result); ++r) {
    auto schema = available.Text(0, r), table = available.Text(1, r);
    if (tables.empty() ||
        std::find(tables.begin(), tables.end(), table) != tables.end() ||
        std::find(tables.begin(), tables.end(), schema + "." + table) !=
            tables.end())
      selected.emplace_back(schema, table);
  }
  auto directory = output / db.Digest(database.string()).substr(0, 20);
  for (const auto &[schema, table] : selected) {
    const auto start = std::chrono::steady_clock::now();
    Result cols(db.conn, "SELECT column_name FROM information_schema.columns "
                         "WHERE table_schema=" +
                             Quote(schema, '\'') + " AND table_name=" +
                             Quote(table, '\'') + " ORDER BY ordinal_position");
    std::vector<std::string> names;
    for (size_t r = 0; r < duckdb_row_count(&cols.result); ++r)
      names.push_back(cols.Text(0, r));
    if (std::find(names.begin(), names.end(), "rowid") == names.end())
      names.push_back("rowid");
    std::sort(names.begin(), names.end());
    TableSource source(db.conn, schema, table, names);
    std::vector<size_t> populations(names.size(), source.count);
    auto cfg = adaptive.dump(), fields = Json(names).dump();
    void *handle =
        sem_table_build(cfg.c_str(), fields.c_str(), populations.data(),
                        names.size(), TableSource::Fetch, &source);
    if (!handle)
      throw std::runtime_error(std::string(sem_prefix_error()) + " " +
                               source.error);
    std::unique_ptr<void, decltype(&sem_adaptive_destroy)> guard(
        handle, sem_adaptive_destroy);
    Json value = Json::parse(sem_adaptive_json(handle));
    for (auto &summary : value["summaries"])
      summary["metadata"]["serializer"] = "prompt-v4-first-field-value-v1";
    if (Snapshot(database) != snapshot)
      throw std::runtime_error(
          "Source database changed during offline construction");
    value.update(Json{{"version", 2},
                      {"snapshot", snapshot},
                      {"catalog", catalog},
                      {"schema", schema},
                      {"table", table}});
    auto key = "[" + Json(schema).dump() + ", " + Json(table).dump() + "]";
    auto path = directory / (db.Digest(key) + ".json");
    AtomicWrite(path, value);
    std::cout << Json({{"schema", schema},
                       {"table", table},
                       {"rows", source.count},
                       {"fields", names.size()},
                       {"sampled_rows", value["cooccurrence"]["rows"].size()},
                       {"parallelism", parallelism},
                       {"seconds", std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count()},
                       {"path", path.string()}})
                     .dump()
              << std::endl;
  }
  Result commit(db.conn, "COMMIT");
  std::set<std::string> matched;
  for (const auto &[schema, table] : selected) {
    matched.insert(table);
    matched.insert(schema + "." + table);
  }
  return matched;
}

int main(int argc, char **argv) {
  try {
    fs::path root = SEM_ROOT_PATH;
    fs::path config = root / "config/system/config.json", database, output;
    std::vector<std::string> tables;
    Json adaptive;
    int parallelism = 0;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (++i >= argc)
          throw std::invalid_argument("Missing value for " + arg);
        return argv[i];
      };
      if (arg == "--config")
        config = next();
      else if (arg == "--database")
        database = next();
      else if (arg == "--output")
        output = next();
      else if (arg == "--parallelism")
        parallelism = std::stoi(next());
      else if (arg == "--adaptive-config")
        adaptive = Json::parse(next());
      else if (arg == "--tables") {
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          tables.push_back(argv[++i]);
      } else if (arg == "--help") {
        std::cout
            << "semcache_statistics_build [--config PATH] [--database PATH] "
               "[--output DIR] [--parallelism 1..64] [--tables NAME ...]\n";
        return 0;
      } else
        throw std::invalid_argument("Unknown option: " + arg);
    }
    Json settings = semcache::ReadConfiguration(config);
    if (output.empty())
      output = root / settings.at("sampling")
                          .at("offline_statistics_dir")
                          .get<std::string>();
    if (adaptive.is_null())
      adaptive = settings.at("sampling").at("adaptive_sketch");
    if (!parallelism)
      parallelism =
          settings.at("statistics").at("build_parallelism").get<int>();
    if (parallelism < 1 || parallelism > 64)
      throw std::invalid_argument("parallelism must be in [1,64]");
    adaptive["build_parallelism"] = parallelism;
    std::vector<fs::path> databases;
    if (!database.empty()) {
      databases.push_back(database);
    } else {
      for (const auto &entry :
           settings.at("runtime").at("source_databases").items())
        databases.push_back(root / entry.value().get<std::string>());
    }
    std::set<std::string> matched;
    for (const auto &path : databases) {
      auto built = BuildDatabase(root, path, output, settings, adaptive,
                                 parallelism, tables);
      matched.insert(built.begin(), built.end());
    }
    for (const auto &table : tables)
      if (!matched.count(table))
        throw std::invalid_argument(
            "Requested offline statistics table does not exist: " + table);

  } catch (const std::exception &e) {
    std::cerr << "Offline statistics: " << e.what() << std::endl;
    return 1;
  }
}
