#pragma once
#include "../common/exact_integer.hpp"
#include "../common/operators.hpp"
#include "../common/parallel.hpp"
#include "../common/stage_pipeline.hpp"
#include "../common/ordered_tasks.hpp"
#include "../statistics/population_moments.hpp"
#include "../statistics/sample_weights.hpp"
#include "../statistics/value_metrics.hpp"
#include "numeric.hpp"
#include "populations.hpp"
#include "random.hpp"
#include "survival/sampled_rates.hpp"
#include "survival/source_evidence.hpp"
#include "timing.hpp"
#include "types.hpp"
#include <tuple>
#include <unordered_map>

namespace semcache::sampling {
using resident_detail::columns;
using resident_detail::combine;
using resident_detail::CoreObservation;
using resident_detail::Counts;
using resident_detail::field_key;
using resident_detail::FieldDepthIndex;
using resident_detail::find_value;
using resident_detail::finite_number;
using resident_detail::FSum;
using resident_detail::Group;
using resident_detail::independent_leaf_identities;
using resident_detail::input_keys;
using resident_detail::join_operator;
using resident_detail::leaf_at;
using resident_detail::leaf_for;
using resident_detail::node_relkey;
using resident_detail::Observation;
using resident_detail::Origins;
using resident_detail::origins;
using resident_detail::output_column;
using resident_detail::PhaseMetric;
using resident_detail::PhaseTimer;
using resident_detail::product;
using resident_detail::Random;
using resident_detail::Relation;
using resident_detail::relkey;
using resident_detail::replace_leaf;
using resident_detail::representative;
using resident_detail::restore;
using resident_detail::row_at;
using resident_detail::row_count;
using resident_detail::sample_column;
using resident_detail::Selection;
using resident_detail::StateData;
using resident_detail::Stratum;
using resident_detail::Sum;
using resident_detail::task_key;

struct NativeSampler::Impl {
  struct Moment {
    std::size_t n = 0;
    double mean = 0, m2 = 0;
  };
  std::map<std::tuple<std::string, std::string, std::string>, Moment> moments;
  mutable std::unique_ptr<ParallelRanges> cpu_workers;
  Json imported, profile_, history_, settings, sketch_config;
  std::shared_ptr<NativeSQLBackend> backend;
  StagePipeline<Json> sql_pipeline;
  struct StateWorker {
    std::shared_ptr<NativeSQLBackend> backend;
    StagePipeline<Json> pipeline;
  };
  std::unique_ptr<OrderedTasks> state_workers;
  std::vector<std::unique_ptr<StateWorker>> state_slots;
  struct StateActor {
    Impl &owner;
    OrderedTasks::Task &task;
    StateWorker &worker;
    std::shared_ptr<NativeSQLBackend> saved;
    StateActor(Impl &, OrderedTasks::Task &, StateWorker &);
    ~StateActor();
    void leave();
    void enter();
  };
  static thread_local StateActor *state_actor;
  void state_turn();
  Json state_backend_call(const std::function<Json()> &,
                          const std::function<void()> &prepare = {});
  void run_state_services(std::size_t count,
                          const std::function<void(std::size_t)> &);
  std::size_t pending_selection = 0;
  std::map<std::string, std::size_t> physical_claims, selection_claims;
  std::map<std::pair<const void *, std::string>, std::set<std::size_t>>
      claim_waiters;
  struct StateClaim {
    Impl &owner;
    std::map<std::string, std::size_t> &claims;
    std::vector<std::string> keys;
    std::optional<std::size_t> ticket;
    StateClaim(Impl &, std::map<std::string, std::size_t> &,
               std::vector<std::string>);
    void release();
    ~StateClaim();
  };
  struct QualityWorker {
    std::shared_ptr<NativeSQLBackend> backend;
    StagePipeline<Json> pipeline;
    explicit QualityWorker(std::shared_ptr<NativeSQLBackend> b)
        : backend(std::move(b)) {}
  };
  std::vector<std::unique_ptr<QualityWorker>> quality_workers;
  std::vector<std::shared_ptr<NativeSQLBackend>> selection_workers;
  std::map<Id, std::size_t> selection_model_cursor;
  std::map<Id, Json> ops;
  std::map<std::string, Relation> relations;
  std::map<std::pair<std::string, std::size_t>, SharedSamples> direct_sources;
  bool direct_sampling = false;
  std::optional<Id> sampling_input_owner;
  std::map<std::tuple<Id, std::string, std::size_t>, SharedSamples>
      operator_direct_sources;
  std::map<Id, std::size_t> operator_sample_sizes;
  std::unordered_map<std::uint64_t, StateData> states;
  mutable std::unordered_map<std::uint64_t, Json> state_actions;
  mutable std::unordered_map<std::uint64_t, Json> execution_ops;
  mutable std::unordered_map<const Json *, std::string> execution_signatures;
  PopulationVersions population_versions;
  std::map<std::string, std::uint64_t> join_probe_seeds;
  std::map<std::string, std::weak_ptr<const Observation>> observations;
  std::map<std::string, std::weak_ptr<const CoreObservation>> cores;
  mutable std::map<std::string, std::shared_ptr<const Json>> conditional_builds;
  mutable std::map<std::string, std::shared_ptr<const Json>> depth_cache;
  mutable std::size_t cache_input_view_hits = 0;
  using SurvivalKey =
      std::tuple<std::uint64_t, std::uint64_t, std::string, std::string>;
  mutable std::map<SurvivalKey, std::shared_ptr<const Json>> survival_fields;
  mutable std::map<SurvivalKey, std::shared_ptr<const Json>> selection_fields;
  mutable std::map<SurvivalKey, std::shared_ptr<const Json>>
      selection_source_fields;
  mutable std::map<SurvivalKey, std::shared_ptr<const Json>>
      selection_fusion_fields;
  std::shared_ptr<const resident_detail::AdmissionCohort> survival_cohort;
  mutable resident_detail::SampledRateCache sampled_rate_cache;
  mutable std::map<std::pair<std::string, std::string>, FieldDepthIndex>
      field_indexes;
  std::map<std::string, Selection> selection_cache;
  std::map<std::string, std::map<std::string, bool>> selection_priors;
  std::string selectivity_estimator;
  bool share_state_statistics = true;
  bool sample_depth_selectivity = true;
  bool ensure_prefix_depth_rates = true;
  std::map<std::string, SharedSamples> independent_populations;
  std::map<std::string, Json> independent_projection_ops;
  std::map<Id, std::uint64_t> evidence_tasks;
  std::map<std::string, std::uint64_t> evidence_contracts;
  std::map<resident_detail::EvidenceFamily, resident_detail::ContextPrior>
      shared_priors;
  std::map<resident_detail::EvidenceFamily,
           resident_detail::EvidenceNode::DepthPriors>
      shared_depth_priors;
  std::unordered_map<std::uint64_t,
                     std::shared_ptr<const resident_detail::EvidenceNode>>
      evidence_nodes;
  std::map<std::string, std::unique_ptr<resident_detail::FilterPool>>
      filter_pools;
  std::map<Id, std::optional<resident_detail::FilterBinding>> filter_bindings;
  std::map<std::string, std::deque<std::string>> join_selection_keys;
  std::map<std::string, std::shared_ptr<resident_detail::JoinProbeEvidence>>
      join_probe_cursors;
  std::set<std::string> join_probe_fallbacks;
  std::map<std::string, std::size_t> diversity_limits;
  std::map<std::string, std::size_t> join_probe_limits;
  std::map<std::string, std::size_t> join_output_limits;
  std::map<std::string, resident_detail::JoinOutputStream> join_streams;
  std::shared_ptr<PopulationFingerprints> populations =
      std::make_shared<PopulationFingerprints>();
  RequestLedger ledger;
  std::map<Id, std::size_t> selection_requests_by_operator;
  std::map<Id, std::pair<std::string, Json>> relational_fallback_cache;
  Json relational_fallback_prior(const NodePtr &node, const Json &op,
                                 const StateData &data);
  std::set<Id> directly_sampled_operators;
  std::vector<Id> accuracy_operator_ids;
  std::size_t accuracy_batch_limit;
  std::map<Id, std::size_t> accuracy_batches_used, accuracy_request_used;
  std::map<Id, std::set<std::string>> accuracy_sampled_inputs;
  mutable RowWeights row_weights;
  mutable PopulationMoments population_moments;
  mutable TextMetrics text_metrics;
  ReuseCache reuse_cache;
  std::map<std::string, std::shared_ptr<const Json>> history_summaries;
  // History is immutable between harvests. Index it once by model so cache
  // edge refreshes do not rescan unrelated models and incompatible prefixes
  // for every resident state.
  std::map<std::string, std::vector<const Json *>> history_by_model;
  std::map<std::string, std::map<Json, std::vector<const Json *>>>
      history_by_prefix;
  std::set<std::string> history_incomplete_models;
  mutable std::map<std::tuple<std::string, std::string, std::string>,
                   std::shared_ptr<const Json>>
      prefix_cooccurrences;
  std::set<std::pair<std::string, std::string>> quality_seen;
  std::set<Id> product_calibrations;
  std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> loads;
  std::uint64_t observation_serial = 0, history_version = 1, history_serial = 0,
                cohort_serial = 1;
  std::size_t initial_size, relational_sample_rows, base_sample_size,
      batch_size, budget, accuracy_size, max_strata, min_depth_samples,
      join_pair_budget, input_samples;
  Json counters = {{"selectivity", 0},
                   {"accuracy", 0},
                   {"judge", 0},
                   {"model_requests", 0},
                   {"sql_stages", 0}};
  Json archived_events = Json::array(), archived_observations = Json::array();
  double backend_seconds = 0;
  bool trace_timings;
  std::optional<std::chrono::steady_clock::time_point> search_deadline;
  bool deadline_reached() const {
    return search_deadline &&
           std::chrono::steady_clock::now() >= *search_deadline;
  }
  mutable std::map<std::string, PhaseMetric> phase_metrics;

  Impl(const Json &config, std::shared_ptr<NativeSQLBackend> b);
  Json slim(const Json &statistics) const;
  std::size_t positive(const char *key, std::size_t fallback);
  void prepare(Relation &rel, const Json &rows, const Json &seed);
  bool admit(Relation &rel, std::size_t count,
             const std::map<std::size_t, double> &priorities);
  double raw_weight(const Json &row, const Counts *counts) const;
  double weight(const Json &row, const Counts *counts = nullptr) const;
  PopulationMoments::Result weighted_moments(const Json &rows) const;
  double mass(const Json &rows, const Counts *counts = nullptr) const;
  double effective(const Json &rows, const Counts *counts = nullptr) const;
  Json boundary(const Json &rows) const;
  const FieldDepthIndex &field_index(const std::string &key,
                                     const std::string &name) const;
  Json field_rates(const std::string &key, const std::string &name,
                   const std::set<std::string> &reached,
                   const std::optional<std::set<std::string>> &eligible,
                   bool complete, bool census_allowed = true,
                   const std::vector<std::size_t> *selected = nullptr) const;
  Json depths(const NodePtr &node, const SampleInputs &rows,
              const std::optional<SampleInputs> &inputs, bool complete,
              bool census_allowed = true, bool match_incomplete = false) const;
  Json source_depths(const NodePtr &node,
                     const resident_detail::SourceEvidence &reached,
                     const resident_detail::SourceEvidence *eligible,
                     bool complete, bool census_allowed,
                     bool match_incomplete) const;
  Json
  sampled_field_rates(const std::string &key, const std::string &name,
                      const resident_detail::SourceEvidence &reached,
                      const resident_detail::SourceEvidence &eligible,
                      bool conditional, bool complete,
                      bool match_incomplete = false,
                      const std::vector<std::size_t> *selected = nullptr) const;
  std::vector<std::string> models(const Json &op) const;
  std::vector<Group> groups(const Json &op, const SampleInputs &inputs,
                            const std::string &model, const Counts *counts,
                            std::optional<std::vector<int>> &leaf,
                            const Json *units = nullptr,
                            bool defer_units = false) const;
  bool harvest(NativeSQLBackend *source = nullptr);
  void calibrate(const std::string &bank, const std::string &owner,
                 const std::string &key, const std::vector<double> &values,
                 const std::string &metric, const std::string &basis);
  void update_history_distances();
  void rebuild_history_index();
  void evict_history(const std::string &model, double now);
  Json execute(const Json &op, const SampleInputs &inputs, const Json &action,
               const Json *pairs = nullptr,
               const std::string &purpose = "selection",
               const std::function<void()> &prepare = {});
  bool resolve_groups(const Json &op, const SampleInputs &inputs,
                      const std::string &model,
                      const std::vector<Group> &groups,
                      const std::function<void()> &prepare = {});
  bool resolve_parallel_groups(const Json &op, const SampleInputs &inputs,
                               const std::vector<Group> &groups,
                               const std::function<void()> &prepare);
  Selection selection(const Json &op, const SampleInputs &inputs, bool semantic,
                      bool full_join_input = false);
  const resident_detail::FilterBinding *filter_binding(const Json &op);
  resident_detail::FilterPool &filter_pool(const std::string &relation);
  std::optional<std::size_t>
  filter_coordinate(const resident_detail::FilterBinding &binding,
                    const Json &row);
  std::optional<Selection> filter_selection(const Json &op,
                                            const SampleInputs &inputs);
  double selection_effective(const Json &op, const SampleInputs &inputs,
                             double ordinary_effective);
  void record_filter(const NodePtr &node, const Json &op,
                     const SampleInputs &inputs, const Selection &selection);
  std::optional<resident_detail::ConditionEstimate>
  conditional_selectivity(const NodePtr &node, const Json &op,
                          const StateData &data, CoreObservation &core);
  Selection join_batch(const Json &op, Selection selection,
                       std::size_t input_count, bool full_join_input = false);
  Selection sample_join(const Json &op, const SampleInputs &inputs,
                        const std::string &model, const std::string &key);
  std::optional<Selection> incremental_join(const Json &op,
                                            const SampleInputs &inputs,
                                            const std::string &model,
                                            const std::string &key,
                                            std::uint64_t seed);
  Json join_diversity(const Json &op, const Json &rows);
  const Json &execution_op(const StatePtr &state, const NodePtr &node) const;
  OwnedInputs prepare_inputs(const Json &op, const OwnedInputs &inputs);
  Json cache_inputs(const Json &op, const Json &action,
                    const SampleInputs &inputs, bool complete,
                    Observation &observation,
                    const OwnedInputs *input_boundaries = nullptr,
                    const Json *previous_survivals = nullptr) const;
  Json build_cache_inputs(const Json &op, const Json &action,
                          const SampleInputs &inputs, bool complete,
                          Observation &observation,
                          const OwnedInputs *input_boundaries,
                          const Json *previous_survivals) const;
  const Json &action_for(const StatePtr &state, Id oid) const;
  const Json &profile_statistic(const std::string &bank,
                                const std::string &model,
                                const std::string &key,
                                const Json *fallback = nullptr) const;
  std::string state_token(const NodePtr &node, const StateData &data) const;
  std::string core_token(const NodePtr &node, const StateData &data) const;
  StateData replay(const StatePtr &state);
  std::set<std::string> independent_dependencies(const Json &op,
                                                const NodePtr &node) const;
  OwnedInputs bound_independent_inputs(const Json &op,
                                       const OwnedInputs &inputs);
  SharedSamples independent_population(const StatePtr &state,
                                       const NodePtr &node,
                                       const std::set<std::string> &required,
                                       bool &complete,
                                       std::size_t source_limit = 0);
  std::shared_ptr<const resident_detail::EvidenceNode>
  evidence_node(const NodePtr &node, const StateData &data);
  std::optional<resident_detail::ContextPrior::Estimate>
  shared_selectivity(const Json &info, CoreObservation &core);
  void publish_depth_selectivity(const Json &info, CoreObservation &core);
  Json depth_selectivity(const CoreObservation &core, const Json &info,
                         const std::string &relation,
                         const std::string &field) const;
  Json depth_selectivity_fusion(const CoreObservation &core,
                                const std::string &relation,
                                const std::string &field) const;
  Json blend_depths(const CoreObservation &core, const Json &info,
                    const std::string &relation, const std::string &field,
                    const std::map<std::string, bool> &depth_support,
                    const Json &evidence) const;
  Json expression_depths(const Json &op, const CoreObservation &core) const;
  Json depth_selectivity_sources(const CoreObservation &core,
                                 const std::string &relation,
                                 const std::string &field) const;
  void cache_edges(StateData &data);
  CrossDependence cache_dependence(const Observation &source,
                                   const Observation &target) const;
  void calibrate_time(const StatePtr &state);
  void record_selection_time(const Json &op, const SampleInputs &inputs,
                             const Json &action, const Json *pairs,
                             const Json &observation, const Json &progress);
  void observe(const StatePtr &state);
  std::size_t selection_remaining() const {
    return ledger.remaining() - std::min(ledger.remaining(), pending_selection);
  }
  void estimate_survivals(const NodePtr &node, const Json &op,
                          const StateData &data, Json &info,
                          CoreObservation &core);
  Json input_survival(const CoreObservation &core, const StateData &data,
                      const std::string &relation,
                      const std::string &field) const;
  Json output_survival(const CoreObservation &core, const StateData &data,
                       const std::string &relation,
                       const std::string &field) const;
  bool estimates_current(const NodePtr &node, const CoreObservation &core,
                         const StateData &data) const;
  std::shared_ptr<const CoreObservation>
  refresh_core(const NodePtr &node, const Json &op, const StateData &data,
               const CoreObservation &core, std::uint64_t estimate_serial);
  bool refresh_estimates(StateData &data);
  void update_quality(Json &estimate, bool match, double weight,
                      const std::string &teacher);
  void accuracy_batch(const std::vector<StatePtr> &states);
  using QualitySink = std::function<void(const Json &, const Group &,
                                         const std::vector<std::string> &,
                                         const std::string &, const Json &)>;
  void prepare_accuracy(const StatePtr &state, const QualitySink &compare);
  std::vector<Group> quality_groups(const Json &op, const CoreObservation &core,
                                    const std::string &model,
                                    const Counts &counts);
  void finish_accuracy(const StatePtr &state);
  void initialize_accuracy_budget(const Json &operators);
};
}
