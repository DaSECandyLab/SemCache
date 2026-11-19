#pragma once

#include "../backend.hpp"
#include "../payloads.hpp"
#include "../prepared_arguments.hpp"
#include "../row_lineage.hpp"
#include "../../common/exact_integer.hpp"
#include "../../common/row_metadata.hpp"
#include "../../statistics/statistics.hpp"
#include "../../statistics/relational_estimate.hpp"
#include "../../resident/support/timing.hpp"
#include "../../resident/support/populations.hpp"
#include "duckdb.h"
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace semcache::sampling {
namespace sql_detail {

std::string upper(std::string text);
bool integer_marker(const Json &value);
std::string marker_text(const Json &value);
std::string identifier(const std::string &value);
std::string quote(const std::string &value);
std::string literal(const Json &value);
void replace_all(std::string &text, const std::string &from, const std::string &to);
std::size_t row_count(const Json &rows);
std::vector<std::string> columns(const Json &rows);
Json merged_lineage(const Json &left, const Json &right, std::size_t &canonicalizations);
std::vector<std::string> names_for(const Json &op, std::size_t side);

} // namespace sql_detail

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
    explicit API(const std::string &path);
    ~API();
  } api;
  duckdb_database database = nullptr;
  duckdb_connection connection = nullptr;
  Json config;
  double stage_seconds = 0.;
  std::optional<std::size_t> timed_input_units;
  Json types, events = Json::array(), observations = Json::array();
  Json selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  const std::map<std::string, std::string> *sampling_ids = nullptr;
  std::string sampling_ids_key;
  enum class CellType { Text, Boolean, Floating, Integer, Nested };
  std::unordered_map<std::string, CellType> cell_types;
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

  struct Scope {
    Impl &backend;
    std::vector<std::string> owned, expressions, metadata;
    std::vector<std::shared_ptr<const std::map<std::string, Json>>> leaves;
    std::vector<std::map<std::string, const Json *>> sources;
    bool finished = false;
    bool restore_sources;
    explicit Scope(Impl &backend, bool restore_sources = true);
    ~Scope();
    std::string identity(const Json &rows, const Json &selected, std::size_t &leaf) const;
    std::string compile(const Json &rows, const std::vector<std::string> &names,
                        bool track, const SharedSamples &population = {},
                        std::size_t retained_rows = 0);
    void done();
  };

  struct OutputContext {
    const Json &op;
    const SampleInputs &inputs;
    const SampleInputs &incoming;
    Scope &scope;
    std::unique_ptr<Scope> &existence;
    const std::vector<std::shared_ptr<const std::map<std::string, Json>>> &normalized;
    const Json &records;
    const std::vector<std::string> &returned;
    const std::vector<std::string> &source_count_columns;
    bool factored;
    bool aggregate;
    bool source_only;
    bool left_semi;
    bool paired;
    bool preserve;
    bool sampled_factors;
    const std::string &kind;
    std::size_t output_limit;
    std::size_t output_population;
  };

  explicit Impl(const Json &config);
  ~Impl();
  std::string stage_id(const Json &row) const;
  static CellType classify_type(const std::string &name);
  void field_type(const std::string &key, const std::string &type);
  void close(bool remove_directory = false) noexcept;
  Json query(const std::string &sql);
  Json value(const std::string &key, Json item) const;
  void append(const std::string &table, const Json &rows, const std::vector<std::string> &names);
  std::string table(const Json &rows, const std::vector<std::string> &requested,
                    std::vector<std::string> &owned, const SharedSamples &population = {},
                    std::size_t retained_rows = 0);
  void release(const std::vector<std::string> &owned);
  void rollback(const std::vector<std::string> &owned) noexcept;
  std::shared_ptr<const std::map<std::string, Json>>
  normalized_rows(const std::string &name, const std::set<std::string> *selected_ids = nullptr);
  std::string stage_sql(const Json &op, const std::vector<std::string> &expressions,
                        const std::vector<std::string> &metadata = {});
  void collect_events();
  Json run(const Json &op, const SampleInputs &inputs, const Json &action, bool preserve = false);
  Json reconstruct_output(OutputContext &ctx);
};

} // namespace semcache::sampling
