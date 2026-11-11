#pragma once
#include "../common/shared_statistics.hpp"
#include "../statistics/depth_index.hpp"
#include "join/join_probe_evidence.hpp"
#include "join/join_stream.hpp"
#include "sampler.hpp"
#include "survival/source_evidence.hpp"
#include "filter/decision_mask.hpp"
#include "filter/filter_pool.hpp"
#include "evidence/context_prior.hpp"
#include <array>
#include <cstdint>

namespace semcache::sampling::resident_detail {
struct Stratum {
  double population = 0;
  Json rows = Json::array();
  std::size_t selected = 0;
};
struct FieldDepthIndex {
  std::uint64_t version = 0;
  std::vector<DepthCohort> cohorts;
  std::vector<DepthPopulation> populations;
};
struct Relation {
  Json id;
  double population = 0;
  std::vector<Stratum> strata;
  Json admitted = Json::array();
  std::map<std::string, std::size_t> row_strata;
  std::map<std::pair<std::string, std::string>, std::size_t> row_lengths;
  std::map<std::string, std::shared_ptr<const Json>> summaries;
  // Borrowed from the immutable Impl::imported offline artifact snapshot.
  const Json *offline_cooccurrence = nullptr;
  std::map<std::string, std::string> offline_columns;
  std::map<std::pair<std::size_t, std::string>, std::map<std::size_t, double>>
      length_counts;
  std::uint64_t version = 1;
};
struct AdmissionCohort {
  std::uint64_t serial = 0;
  std::map<std::string, std::vector<std::size_t>> selected;
};
struct FilterBinding {
  std::string relation, task;
  std::set<std::string> fields;
};
struct CacheInputView {
  Json statistics;
  std::shared_ptr<const Json> summary, right_summary, conditional_plan;
  std::string relation, field;
};
struct CoreObservation {
  // An unpaid suffix may be shared only while the input/probe versions and
  // request context are unchanged. A new paid answer must retry it rather
  // than treating unknown labels as negative evidence.
  struct RetryContext {
    std::size_t requests = 0;
    bool expired = false;
  };
  std::optional<RetryContext> retry;
  NodePtr node;
  std::uint64_t serial = 0;
  std::uint64_t estimate_serial = 0;
  std::vector<std::uint64_t> input_estimate_versions;
  std::string prior_task;
  std::size_t prior_samples = 0;
  std::string condition_pool;
  std::uint64_t condition_version = 0;
  std::shared_ptr<const EvidenceNode> evidence;
  std::uint64_t shared_prior_version = 0;
  bool shared_prior_subscribed = false;
  std::uint64_t depth_prior_version = 0;
  bool depth_prior_subscribed = false;
  std::string depth_exposure;
  // Even exhaustive admitted pairs cannot prove a source has no counterpart
  // when the opposite base population is only partially admitted.
  std::set<std::string> uncertain_source_matches;
  std::shared_ptr<const Json> rows, boundary, statistics;
  std::shared_ptr<const Json> statistical_rows;
  std::shared_ptr<const Json> expression_statistics;
  SharedStatistics scalar_statistics;
  OwnedInputs inputs;
  OwnedInputs input_boundaries;
  // Input/cohort evidence is shared by implementation variants. Static
  // prompt prefixes and history-dependent reuse remain specific to each view.
  mutable std::map<std::vector<std::string>, CacheInputView> cache_input_views;
  std::shared_ptr<const Json> probe_units;
  std::shared_ptr<const JoinProbeSnapshot> join_evidence;
  std::shared_ptr<const DecisionMask> decisions;
  std::shared_ptr<const AdmissionCohort> cohort;
  Json estimated_survivals = Json::object();
  bool sampled = false, evidence_sampled = false;
  // Kept at the boundary, not only on rows: empty finite Join outputs cannot
  // prove that no source row matches an unprobed counterpart.
  bool input_match_incomplete = false, match_incomplete = false;
  bool full_join_input = false;
  std::string join_sample_key;
  std::size_t join_probe_limit = 0;
  std::string join_stream_key{};
  std::size_t join_output_limit = 0;
  bool complete = true, input_complete = true;
};
struct Selection {
  SharedSamples rows, probes, units;
  Json diagnostics = Json::object();
  OwnedInputs prepared_inputs;
  std::string join_sample_key;
  std::size_t join_probe_limit = 0;
  std::string join_stream_key{};
  std::size_t join_output_limit = 0;
  SharedSamples statistical_rows{};
  std::shared_ptr<const JoinProbeSnapshot> join_evidence{};
  std::shared_ptr<const DecisionMask> decisions{};
  explicit operator bool() const { return bool(rows); }
};
struct Observation {
  std::uint64_t serial = 0;
  std::string token;
  SharedStatistics statistics;
  std::shared_ptr<const Json> rows, boundary;
  std::shared_ptr<const CoreObservation> core;
  std::shared_ptr<const Json> summary, right_summary, conditional_plan;
  std::string cache_relation, cache_field;
  bool complete = true;
};
struct StateData {
  StatePtr state;
  std::map<Id, std::shared_ptr<const Observation>> observations;
  std::map<Id, SharedStatistics> stats;
};
struct Group {
  std::string key;
  std::vector<Json> unit;
  std::vector<std::array<std::size_t, 2>> members;
  double weight = 0;
};
}
