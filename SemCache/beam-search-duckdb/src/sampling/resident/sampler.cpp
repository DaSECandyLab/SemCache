#include "detail.hpp"
#include "../profile/environment.hpp"
#include "../common/prefix_selectivity.hpp"
#include <future>

namespace semcache::sampling {
namespace {
template <class Map, class Predicate>
void erase_matching(Map &values, Predicate remove) {
  for (auto it = values.begin(); it != values.end();)
    if (remove(*it))
      it = values.erase(it);
    else
      ++it;
}
}
NativeSampler::NativeSampler(const Json &config,
                             std::shared_ptr<NativeSQLBackend> backend)
    : impl(std::make_unique<Impl>(config, std::move(backend))) {}
NativeSampler::~NativeSampler() = default;
std::shared_ptr<SearchEnvironment>
NativeSampler::prepare_statistics(const StateProblem &problem,
                                  const ActionPriorities &priorities,
                                  const std::optional<std::string> &fixed_model,
                                  bool collect_accuracy, double max_seconds) {
  // Fixed-model heuristic search ranks scalar work estimates only.
  // Prefix coverage belongs to value-model search, even for direct callers.
  if (fixed_model) {
    impl->ensure_prefix_depth_rates = false;
    impl->settings["ensure_prefix_depth_rates"] = false;
  }
  auto state = problem.initial();
  // This is a data-collection traversal, not a candidate or an injected plan.
  // Every target still draws its own base cohort and necessary producer chain.
  while (!problem.complete(state)) {
    const auto choices =
        problem.actions(state, false, true, &priorities, fixed_model);
    if (choices.empty())
      throw std::invalid_argument(
          "Cannot collect operator statistics without a legal construction");
    state = problem.transition(state, choices.front());
  }
  begin_search(max_seconds);
  // Relational scalar probes have their own read-only connection. They do
  // not own the semantic SQL connection or the HTTP request budget. Join at
  // the freeze boundary, after independent semantic/accuracy observations.
  const auto source = impl->backend;
  Json population_ops = Json::array();
  for (const auto &[_, op] : impl->ops)
    if (op.contains("population_sample"))
      population_ops.push_back(op);
  auto population_job = std::async(std::launch::async,
      [source, operators = std::move(population_ops)] {
        Json result = {{"operators", Json::object()}, {"errors", Json::object()},
                       {"started_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count()}};
        if (source && !operators.empty()) {
          const auto worker = source->fork_semantic();
          for (const auto &op : operators) {
            const auto nid = std::to_string(op.at("node_id").get<Id>());
            try {
              result["operators"][nid] = worker->sample_population(op);
            } catch (const std::exception &error) {
              // Unsupported/failed probes retain their existing explicit
              // prior. Never manufacture a measured zero from a SQL error.
              result["errors"][nid] = error.what();
            }
          }
        }
        result["completed_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return result;
      });
  Json population;
  try {
    observe(state);
    if (impl->direct_sampling &&
        impl->settings.value("resample_until_output", true))
      resample_empty(state);
    if (fixed_model)
      impl->profile_["fixed_model_id"] = *fixed_model;
    else
      impl->profile_.erase("fixed_model_id");
    if (collect_accuracy && impl->settings.value("sample_accuracy", true))
      impl->accuracy_batch({state});
    impl->calibrate_time(state);
    refresh({state});
    population = population_job.get();
    impl->counters["relational_population_sampling"] = population;
  } catch (...) {
    end_search();
    throw;
  }
  end_search();
  Json evidence = impl->profile_;
  evidence.update(profile());
  evidence.erase("conditional_inputs");
  evidence.erase("prefix_statistics");
  // EXPLAIN estimates collected during preparation are operator facts too.
  // Keep the local rate/cost priors, not the canonical traversal's overlays.
  for (const auto &[id, observation] :
       impl->states.at(state->handle_id).observations) {
    const auto &node = observation->core->node;
    if (node->operator_id &&
        !semantic_operator(impl->ops.at(*node->operator_id))) {
      auto &prior = evidence["node_statistics"][std::to_string(id)];
      if (!prior.is_object())
        prior = Json::object();
      prior.update(*observation->core->statistics);
      const auto sampled = population.at("operators").find(std::to_string(id));
      if (sampled != population.at("operators").end()) {
        // Publish the fresh 50/50 sample/EXPLAIN estimate. Do not retain an
        // older fusion target over the newly collected source observations.
        for (const auto *key : {"estimated_selectivity", "selectivity_fusion",
                                "survival_estimated", "survival_estimation_basis"})
          prior.erase(key);
        const auto reliability = prior.value("reliability", Json::object());
        prior.update(*sampled);
        Json combined = reliability;
        combined.update(sampled->at("reliability"));
        prior["reliability"] = std::move(combined);
      }
      const auto &op = impl->ops.at(*node->operator_id);
      const auto kind = op.value("kind", op.value("type", std::string()));
      if (prefix_population_boundary(kind)) {
        // A scalar aggregate still emits one result on empty input. Keep that
        // cardinality without requiring a fictitious input survival rate (1/0).
        prior.erase("depth_selectivity_statistics");
        prior["prefix_selectivity_statistics"] = Json::object();
        prior["prefix_selectivity_status"] = "not_applicable_aggregate_input";
        continue;
      }
      if (impl->ensure_prefix_depth_rates) {
        const auto &core = *observation->core;
        const bool preserving = prefix_preserving_operator(kind);
        prior["prefix_selectivity_statistics"] = Json::object();
        prior["input_field_summaries"] = Json::object();
        const auto raw =
            prior.value("depth_selectivity_statistics", Json::object());
        std::set<std::string> required;
        for (const auto &descriptor :
             op.value("required_columns", Json::array()))
          required.insert(sample_column(descriptor));
        for (const auto &field : required) {
          Json summary = nullptr;
          std::string key = "generated:" + field;
          Json origin = nullptr;
          Json scalar =
              preserving ? Json(1.) : prior.value("selectivity", Json(nullptr));
          bool empty = false;
          for (const auto &raw_relation : node->relations) {
            const auto relation = node_relkey(raw_relation);
            const auto &source = impl->relations.at(relation);
            const auto found = source.summaries.find(field);
            if (found == source.summaries.end())
              continue;
            summary = *found->second;
            key = relation + ":" + field;
            origin = Json::array({relation, field});
            empty = source.population == 0;
            if (join_operator(op, node->children.size()))
              for (std::size_t side = 0; side < core.inputs.size(); ++side)
                if (origins(*core.inputs[side]).count(relation))
                  scalar = prior.value(
                      side ? "right_selectivity" : "left_selectivity", scalar);
            break;
          }
          if (summary.is_null()) {
            Json values = Json::array();
            for (const auto &input : core.inputs) {
              const auto leaf = leaf_for(*input, {field});
              if (!leaf)
                continue;
              const auto &rows = leaf_at(*input, *leaf);
              for (std::size_t i = 0; i < row_count(rows); ++i)
                if (const auto value = find_value(rows, i, field))
                  values.push_back(serialize_value(*value));
              break;
            }
            // Independent predicate populations deliberately skip aggregate
            // and window execution. Their produced bindings can still be an
            // ordinary projection's real inputs: reuse values already emitted
            // by that direct child during preparation, without executing it
            // again or substituting another source column.
            bool empty_producer = false;
            if (values.empty())
              for (const auto child : node->children) {
                const auto &producer =
                    impl->states.at(state->handle_id).observations.at(child);
                const auto &producer_node = producer->core->node;
                if (!producer_node->operator_id || !producer->complete)
                  continue;
                const auto &description =
                    impl->ops.at(*producer_node->operator_id);
                const auto producer_kind = description.value(
                    "kind", description.value("type", std::string()));
                const bool produces_values =
                    producer_kind.find("AGGREGATE") != std::string::npos ||
                    producer_kind.find("WINDOW") != std::string::npos;
                if (!produces_values)
                  continue;
                bool declared = false;
                for (const auto fid : producer_node->fields) {
                  const auto &descriptor = impl->imported.at("fields").at(fid);
                  declared |= sample_column(descriptor) == field ||
                              field_key(descriptor) == field;
                }
                const auto leaf = leaf_for(*producer->rows, {field});
                if (leaf) {
                  const auto &rows = leaf_at(*producer->rows, *leaf);
                  for (std::size_t i = 0; i < row_count(rows); ++i)
                    if (const auto value = find_value(rows, i, field))
                      values.push_back(serialize_value(*value));
                  if (!values.empty())
                    break;
                }
                empty_producer |= declared && !row_count(*producer->rows);
              }
            const bool empty_inputs =
                core.input_complete && !core.inputs.empty() &&
                std::any_of(
                    core.inputs.begin(), core.inputs.end(),
                    [](const auto &input) { return row_count(*input) == 0; });
            if (!values.empty() || empty_inputs || empty_producer) {
              const auto built =
                  build_summaries(Json{{field, values}}, impl->sketch_config);
              summary = built.at("summaries").at(field);
            }
          }
          if (summary.is_null())
            throw std::runtime_error(
                "Missing prefix input summary for relational operator " +
                std::to_string(*node->operator_id) + ", field " + field);
          auto dense = complete_prefix_selectivity(
              prefix_depth_support(summary),
              raw.is_object() ? raw.value(key, Json::object()) : Json::object(),
              scalar, preserving, empty);
          for (auto depth = dense.begin(); depth != dense.end(); ++depth)
            if (!finite_number(depth.value().at("rate")))
              throw std::runtime_error("No finite prefix selectivity anchor "
                                       "for relational operator " +
                                       std::to_string(*node->operator_id) +
                                       ", field " + field + ", depth " +
                                       depth.key());
          prior["input_field_summaries"][field] = {
              {"summary", std::move(summary)}, {"survival_field", key}};
          if (!origin.is_null())
            prior["input_field_summaries"][field]["origin"] = std::move(origin);
          prior["prefix_selectivity_statistics"][key] = std::move(dense);
        }
      }
    }
  }
  Json frozen = {{"query", impl->imported.at("query")},
                 {"fields", impl->imported.at("fields")},
                 {"operators", impl->imported.at("operators")},
                 {"profile", std::move(evidence)},
                 {"history", history()},
                 {"sampling", impl->settings},
                 {"prepared_statistics", true},
                 {"sampling_calls", calls()},
                 {"sampling_timings", timings()}};
  return std::make_shared<ProfileEnvironment>(frozen);
}
Json NativeSampler::calibrate_time(const StatePtr &state) {
  impl->calibrate_time(state);
  return impl->profile_.at("operator_time_calibration");
}
void NativeSampler::begin_search(double seconds) {
  const auto now = std::chrono::steady_clock::now();
  const auto maximum = std::chrono::steady_clock::time_point::max();
  impl->search_deadline =
      seconds >= std::chrono::duration<double>(maximum - now).count()
          ? maximum
          : now +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(seconds));
}
void NativeSampler::end_search() { impl->search_deadline.reset(); }
void NativeSampler::observe(const StatePtr &state) { impl->observe(state); }
void NativeSampler::refresh(const std::vector<StatePtr> &states) {
  const bool history_changed = impl->harvest();
  // A later state's observation may publish priors needed by an earlier state.
  for (auto &state : states) {
    if (!impl->states.count(state->handle_id)) {
      impl->observe(state);
    }
  }
  std::vector<bool> changed(states.size());
  for (std::size_t i = 0; i < states.size(); ++i)
    changed[i] = impl->refresh_estimates(impl->states.at(states[i]->handle_id));
  for (std::size_t i = 0; i < states.size(); ++i)
    if (history_changed || changed[i])
      impl->cache_edges(impl->states.at(states[i]->handle_id));
}
void NativeSampler::accuracy(const StatePtr &state) {
  if (impl->settings.value("sample_accuracy", true))
    impl->accuracy_batch({state});
}
void NativeSampler::accuracy_batch(const std::vector<StatePtr> &states) {
  if (impl->settings.value("sample_accuracy", true))
    impl->accuracy_batch(states);
}
void NativeSampler::retain(const std::vector<StatePtr> &live) {
  std::set<std::uint64_t> keep;
  std::set<std::uint64_t> keep_nodes;
  for (auto &state : live) {
    for (auto current = state;
         current && keep.insert(current->handle_id).second;
         current = current->parent) {
      // A retained ancestor's full parent chain is already in the set.
    }
    for (const auto &node : state->nodes())
      keep_nodes.insert(node->handle_id);
  }
  auto unused_state = [&](const auto &entry) {
    return !keep.count(entry.first);
  };
  auto unused_node = [&](const auto &entry) {
    return !keep_nodes.count(entry.first);
  };
  auto expired = [](const auto &entry) { return entry.second.expired(); };
  erase_matching(impl->states, unused_state);
  erase_matching(impl->state_actions, unused_state);
  erase_matching(impl->execution_ops, [&](const auto &entry) {
    if (!unused_node(entry))
      return false;
    impl->execution_signatures.erase(&entry.second);
    return true;
  });
  erase_matching(impl->observations, expired);
  erase_matching(impl->cores, expired);
  erase_matching(impl->evidence_nodes, unused_node);
}
}
