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
#include <map>
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
void AppendEscaped(std::string &out, std::string_view value) {
  for (char c : value) {
    if (c == '<')
      out += "\\u003c";
    else if (c == '>')
      out += "\\u003e";
    else
      out += c;
  }
}
struct JointSource {
  const TableSource &base;
  std::vector<std::vector<size_t>> orders;
  std::vector<std::string> scratch;
  std::string error;
  static int Fetch(size_t f, const size_t *ids, size_t n, const char **strings,
                   size_t *sizes, void *ctx) noexcept {
    auto &s = *static_cast<JointSource *>(ctx);
    try {
      const auto &order = s.orders.at(f);
      s.scratch.resize(n);
      for (size_t i = 0; i < n; ++i) {
        auto &out = s.scratch[i];
        out.clear();
        AppendEscaped(out, s.base.Value(order[0], ids[i]));
        for (size_t k = 1; k < order.size(); ++k) {
          out += "\n{" + std::to_string(k) + "}: ";
          AppendEscaped(out, s.base.Value(order[k], ids[i]));
        }
        strings[i] = out.data();
        sizes[i] = out.size();
      }
      return 0;
    } catch (const std::exception &e) {
      s.error = e.what();
      return 1;
    }
  }
};
std::vector<std::vector<std::string>>
SelectJointOrders(const Json &spec, size_t default_budget,
                  const std::vector<std::string> &available) {
  std::set<std::string> known(available.begin(), available.end());
  size_t budget = default_budget;
  if (spec.is_object() && spec.contains("b_joint"))
    budget = spec.at("b_joint").get<size_t>();
  if (!budget)
    return {};
  const Json &raw = spec.is_object() ? spec.value("orders", Json::array()) : spec;
  if (!raw.is_array())
    throw std::invalid_argument("Joint order specification must be an array");
  struct Ranked {
    double count;
    std::vector<std::string> columns;
  };
  std::map<size_t, std::vector<Ranked>> buckets;
  for (size_t idx = 0; idx < raw.size(); ++idx) {
    const auto &item = raw[idx];
    const Json &cols = item.is_object() ? item.at("columns") : item;
    double weight = item.is_object() ? item.value("count", 1.0)
                                     : double(raw.size() - idx);
    if (!cols.is_array() || cols.size() < 2 || weight <= 0.)
      continue;
    std::vector<std::string> order = cols.get<std::vector<std::string>>();
    std::set<std::string> uniq(order.begin(), order.end());
    if (uniq.size() != order.size())
      continue;
    bool ok = true;
    for (const auto &c : order)
      ok &= known.count(c) != 0;
    if (ok)
      buckets[order.size()].push_back({weight, std::move(order)});
  }
  for (auto &[_, list] : buckets)
    std::sort(list.begin(), list.end(), [](const Ranked &a, const Ranked &b) {
      if (a.count != b.count)
        return a.count > b.count;
      return a.columns < b.columns;
    });
  std::vector<std::vector<std::string>> selected;
  for (size_t pos = 0; selected.size() < budget; ++pos) {
    bool progressed = false;
    for (auto &[_, list] : buckets) {
      if (pos >= list.size())
        continue;
      selected.push_back(list[pos].columns);
      progressed = true;
      if (selected.size() == budget)
        break;
    }
    if (!progressed)
      break;
  }
  return selected;
}
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
                                    const std::vector<std::string> &tables,
                                    const Json &joint_specs) {
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
    const Json &table_joint_spec =
        joint_specs.contains(schema + "." + table)
            ? joint_specs.at(schema + "." + table)
        : joint_specs.contains(table) ? joint_specs.at(table)
                                      : Json();
    const size_t b_joint =
        settings.at("statistics").value("b_joint", size_t(8));
    auto selected_joint =
        table_joint_spec.is_null()
            ? std::vector<std::vector<std::string>>{}
            : SelectJointOrders(table_joint_spec, b_joint, names);
    if (!selected_joint.empty()) {
      JointSource joint_source{source, {}, {}, {}};
      std::vector<std::string> joint_keys;
      joint_source.orders.reserve(selected_joint.size());
      joint_keys.reserve(selected_joint.size());
      for (const auto &order : selected_joint) {
        std::vector<size_t> cols;
        for (const auto &c : order)
          cols.push_back(
              std::find(names.begin(), names.end(), c) - names.begin());
        joint_source.orders.push_back(std::move(cols));
        joint_keys.push_back(Json(order).dump());
      }
      Json joint_cfg = adaptive;
      joint_cfg["cooccurrence_sample_rows"] = 0;
      std::vector<size_t> joint_pop(joint_keys.size(), source.count);
      auto jcfg = joint_cfg.dump(), jfields = Json(joint_keys).dump();
      void *jhandle = sem_table_build(jcfg.c_str(), jfields.c_str(),
                                      joint_pop.data(), joint_keys.size(),
                                      JointSource::Fetch, &joint_source);
      if (!jhandle)
        throw std::runtime_error(std::string(sem_prefix_error()) + " " +
                                 joint_source.error);
      std::unique_ptr<void, decltype(&sem_adaptive_destroy)> jguard(
          jhandle, sem_adaptive_destroy);
      Json jvalue = Json::parse(sem_adaptive_json(jhandle));
      for (auto &summary : jvalue["summaries"])
        summary["metadata"]["serializer"] = "prompt-v4-first-field-value-v1";
      value["joint_summaries"] = std::move(jvalue["summaries"]);
    }
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
    fs::path config = root / "config/system/config.json", database, output,
             joint_orders_path;
    std::vector<std::string> tables;
    Json adaptive, joint_specs = Json::object();
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
      else if (arg == "--joint-orders")
        joint_orders_path = next();
      else if (arg == "--tables") {
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          tables.push_back(argv[++i]);
      } else if (arg == "--help") {
        std::cout
            << "semcache_statistics_build [--config PATH] [--database PATH] "
               "[--output DIR] [--parallelism 1..64] [--joint-orders PATH] "
               "[--tables NAME ...]\n";
        return 0;
      } else
        throw std::invalid_argument("Unknown option: " + arg);
    }
    if (!joint_orders_path.empty()) {
      std::ifstream stream(joint_orders_path);
      if (!stream)
        throw std::runtime_error("Cannot read joint orders: " +
                                 joint_orders_path.string());
      joint_specs = Json::parse(stream);
      if (!joint_specs.is_object())
        throw std::invalid_argument("Joint orders JSON must be an object");
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
                                 parallelism, tables, joint_specs);
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
