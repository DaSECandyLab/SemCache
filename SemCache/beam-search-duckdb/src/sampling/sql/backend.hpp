#pragma once

#include "../common/inputs.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace semcache {
namespace sampling {
using Json = nlohmann::ordered_json;

// Owns the pinned DuckDB connection, private stage tables and actual request
// evidence. No operation in this interface calls the Python interpreter.
// Boundaries are row arrays or {"product":[leftBoundary,rightBoundary]}.
// Rows are {"id":string,"values":object,"lineage":[[relation,id],...]}.
class NativeSQLBackend {
public:
  explicit NativeSQLBackend(const Json &config);
  ~NativeSQLBackend();
  NativeSQLBackend(const NativeSQLBackend &) = delete;
  NativeSQLBackend &operator=(const NativeSQLBackend &) = delete;

  // Independent staging connection with the same database/setup contract.
  // Keep macros/types available to the original semantic stage SQL.
  std::shared_ptr<NativeSQLBackend> fork_semantic() const;
  std::size_t sample_workers_peak() const;
  std::shared_ptr<NativeSQLBackend>
  fork_calibration(const Json &settings) const;
  // Physical endpoint/model identity, excluding credentials and request options.
  std::string scheduling_key(const std::string &model) const;

  Json load(const std::string &relation, const std::string &sql);
  Json execute(const Json &op, const SampleInputs &inputs,
               const Json &action = nullptr);
  std::vector<Json> prepare_inputs(const Json &op, const SampleInputs &inputs);
  Json join_pairs(const Json &op, const Json &pairs, const Json &action);
  Json estimate_relation(const Json &op, const SampleInputs &inputs);
  // Model-free population probe on this connection; no semantic HTTP calls.
  Json sample_population(const Json &op);

  Json selection_progress() const;
  const Json &events() const;
  Json drain_events();
  const Json &observations() const;
  Json drain_observations();
  Json diagnostics() const;
  Json timings() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
}
