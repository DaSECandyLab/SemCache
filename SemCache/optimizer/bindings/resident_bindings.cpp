#include "bridge.hpp"
#include "native_features.hpp"
#include "resident_driver.hpp"
#include "grouped_pruning.hpp"
#include "join_filter_cost.hpp"
#include "runtime_driver.hpp"
#include "sampling/profile/environment.hpp"
#include "sampling/resident/sampler.hpp"
#include "sampling/statistics/relational_estimate.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
using semcache::sampling::from_python;
using semcache::sampling::Json;
using semcache::sampling::NativeSampler;
using semcache::sampling::NativeSQLBackend;
using semcache::sampling::ProfileEnvironment;
using semcache::sampling::SearchEnvironment;
using semcache::sampling::to_python;

py::object frozen_scalar(const Json &value, const py::object &dictionary,
                         const py::object &array) {
  if (value.is_object()) {
    py::dict result;
    for (auto field = value.begin(); field != value.end(); ++field)
      result[py::str(field.key())] = frozen_scalar(field.value(), dictionary, array);
    return dictionary(result);
  }
  if (value.is_array()) {
    py::list result;
    for (const auto &item : value)
      result.append(frozen_scalar(item, dictionary, array));
    return array(result);
  }
  return to_python(value);
}

py::dict raw_batch(const std::vector<semcache::StatePtr> &states,
                   const SearchEnvironment &sampler,
                   bool compact_requests = false,
                   bool compact_features = false) {
  py::list rows;
  // Ancestor records are immutable and shared by sibling candidates. Convert
  // each scalar block once per layer, retaining its owner against address reuse.
  std::unordered_map<const Json *,
                     std::pair<std::shared_ptr<const Json>, py::object>> blocks;
  auto evidence = py::module_::import("planning.observations.evidence");
  auto dictionary = evidence.attr("_EvidenceDict"), array = evidence.attr("_EvidenceList");
  const auto feature_source = compact_features
                                  ? dynamic_cast<const ProfileEnvironment *>(&sampler)
                                  : nullptr;
  for (const auto &state : states) {
    py::dict row;
    row["state"] = state;
    if (compact_requests) {
      // Request-count ranking uses no cache, quality, width or field sketches.
      // Keep the Python arithmetic identical while crossing only its inputs.
      py::dict statistics;
      for (const auto &block : sampler.statistic_blocks(state)) {
        py::dict cardinality;
        for (const auto *key : {"input_rows", "output_rows", "selectivity"}) {
          const auto field = block.second->find(key);
          if (field != block.second->end())
            cardinality[py::str(key)] = to_python(*field);
        }
        statistics[py::str(std::to_string(block.first))] =
            std::move(cardinality);
      }
      row["statistics"] = std::move(statistics);
    } else {
      py::dict statistics;
      const auto fields = feature_source ? feature_source->feature_blocks(state)
                                         : sampler.statistic_blocks(state);
      for (const auto &block : fields) {
        auto found = blocks.find(block.second.get());
        if (found == blocks.end())
          found = blocks.emplace(
              block.second.get(),
              std::make_pair(block.second,
                             frozen_scalar(*block.second, dictionary, array)))
                      .first;
        statistics[py::str(std::to_string(block.first))] = found->second.second;
      }
      row["statistics"] = std::move(statistics);
    }
    rows.append(row);
  }
  py::dict result;
  result["states"] = rows;
  if (compact_requests) {
    py::dict context, profile;
    profile["operator_time_calibration"] =
        to_python(sampler.time_calibration());
    context["profile"] = profile;
    result["context"] = context;
  } else
    result["context"] =
        states.empty() ? py::dict() : to_python(sampler.cache(states.front()));
  return result;
}
semcache::Scores scores_from_python(py::handle value, std::size_t expected) {
  auto rows = py::reinterpret_borrow<py::sequence>(value);
  if (py::len(rows) != expected)
    throw py::value_error("PlanValueModel returned the wrong number of scores");
  semcache::Scores result;
  for (auto item : rows) {
    auto row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 2)
      throw py::value_error("Ranking heads must return two finite scores");
    result.push_back({py::cast<double>(row[0]), py::cast<double>(row[1])});
  }
  semcache::validate_scores(result);
  return result;
}
semcache::ResidentPrediction prediction_from_python(py::handle value,
                                                    std::size_t expected) {
  semcache::ResidentPrediction result;
  if (PyDict_Check(value.ptr())) {
    auto row = py::reinterpret_borrow<py::dict>(value);
    result.scores = scores_from_python(row["scores"], expected);
    result.times = row["times"].cast<std::vector<std::optional<double>>>();
    result.accuracies =
        row["accuracies"].cast<std::vector<std::optional<double>>>();
    result.time_reasons =
        row["time_reasons"].cast<std::vector<std::optional<std::string>>>();
    if (row.contains("guard_index"))
      result.guard_index =
          row["guard_index"].cast<std::optional<std::size_t>>();
  } else {
    result.scores = scores_from_python(value, expected);
    for (const auto &score : result.scores) {
      result.times.push_back(-score[0]);
      result.accuracies.push_back(score[1]);
      result.time_reasons.push_back(std::nullopt);
    }
  }
  if (result.times.size() != expected || result.accuracies.size() != expected ||
      result.time_reasons.size() != expected)
    throw py::value_error("PlanValueModel returned the wrong number of values");
  return result;
}
py::dict export_selection(const semcache::CompletionSelection &native) {
  py::dict row;
  row["selected_indices"] = native.selected;
  row["statuses"] = native.statuses;
  row["constraint_satisfied"] = native.constraint_satisfied;
  row["accurate_index"] = native.accurate_index;
  return row;
}
struct StatisticsExport {
  std::unordered_map<const Json *,
                     std::pair<std::shared_ptr<const Json>, py::object>>
      immutable;
  py::object freeze = py::module_::import("planning.observations.evidence")
                          .attr("freeze_evidence");
  py::dict fields(const semcache::StatePtr &state,
                  const SearchEnvironment &sampler) {
    py::dict result;
    static const std::set<std::string> shared{"summary",
                                              "right_summary",
                                              "conditional_sketch_plan",
                                              "adaptive",
                                              "input_field_summaries",
                                              "prefix_selectivity_statistics",
                                              "input_field_statistics",
                                              "output_field_statistics",
                                              "depth_survival_statistics",
                                              "depth_selectivity_statistics"};
    for (const auto &node : sampler.statistic_fields(state)) {
      py::dict row;
      for (const auto &field : node.second) {
        if (!shared.count(field.first)) {
          row[py::str(field.first)] = to_python(*field.second);
          continue;
        }
        auto found = immutable.find(field.second.get());
        if (found == immutable.end()) {
          py::object converted = freeze(to_python(*field.second));
          found = immutable
                      .emplace(field.second.get(),
                               std::make_pair(field.second, converted))
                      .first;
        }
        row[py::str(field.first)] = found->second.second;
      }
      result[py::str(std::to_string(node.first))] = std::move(row);
    }
    return result;
  }
};
py::dict export_state(const semcache::StatePtr &state,
                      const SearchEnvironment &sampler,
                      StatisticsExport &statistics) {
  py::dict row;
  row["state"] = state;
  row["statistics"] = statistics.fields(state, sampler);
  return row;
}
}

void bind_resident(py::module_ &module) {
  module.def("mean_relational_estimate", [](py::dict sampled, py::dict duckdb) {
    return to_python(semcache::sampling::mean_relational_estimate(
        from_python(sampled), from_python(duckdb)));
  });
  module.def("prune_join_filter_cost",
      [](const semcache::StateProblem &problem,
         const std::vector<semcache::StatePtr> &states,
         const py::list &statistics, const py::dict &calibration) {
        std::vector<Json> stats;
        for (auto row : statistics)
          stats.push_back(from_python(row));
        const auto rates = from_python(calibration);
        semcache::JoinFilterPruning result;
        {
          py::gil_scoped_release release;
          result = semcache::prune_join_filter_cost(problem, states, stats, rates);
        }
        py::dict output;
        output["excluded"] = result.excluded;
        output["decisions"] = to_python(semcache::join_filter_decisions(result));
        return output;
      }, py::arg("problem"), py::arg("states"), py::arg("statistics"), py::arg("calibration"));
  module.def("prune_structure_groups",
      [](const semcache::StateProblem &problem,
         const std::vector<semcache::StatePtr> &states,
         const py::dict &prediction, std::size_t maximum_groups,
         std::size_t per_group) {
        const auto values = prediction_from_python(prediction, states.size());
        semcache::GroupedPruning result;
        {
          py::gil_scoped_release release;
          result = semcache::prune_structure_groups(
              problem, states, values, maximum_groups, per_group);
        }
        py::dict output;
        output["order"] = result.order;
        output["membership"] = result.membership;
        output["candidate_groups"] = result.groups;
        output["mean_predictions"] = result.mean_predictions;
        return output;
      }, py::arg("problem"), py::arg("states"), py::arg("prediction"),
      py::arg("maximum_groups") = 15, py::arg("per_group") = 6);
  py::class_<SearchEnvironment, std::shared_ptr<SearchEnvironment>>(
      module, "SearchEnvironment");
  py::class_<ProfileEnvironment, SearchEnvironment,
             std::shared_ptr<ProfileEnvironment>>(module, "ProfileEnvironment")
      .def(py::init([](py::dict config) {
        const auto native = from_python(config);
        py::gil_scoped_release release;
        return std::make_shared<ProfileEnvironment>(native);
      }))
      .def("observe", &ProfileEnvironment::observe,
           py::call_guard<py::gil_scoped_release>())
      .def("refresh", &ProfileEnvironment::refresh,
           py::call_guard<py::gil_scoped_release>())
      .def("accuracy", &ProfileEnvironment::accuracy,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "apply_runtime",
          [](ProfileEnvironment &env, const semcache::StatePtr &state,
             semcache::Id oid, py::dict entry, py::object observation) {
            const auto native_entry = from_python(entry),
                       native_observation = from_python(observation);
            py::gil_scoped_release release;
            env.apply_runtime(state, oid, native_entry, native_observation);
          },
          py::arg("state"), py::arg("operator_id"), py::arg("entry"),
          py::arg("observation") = py::none())
      .def("statistics",
           [](const ProfileEnvironment &env, const semcache::StatePtr &state) {
             return to_python(env.statistics(state));
           })
      .def("full_statistics",
           [](const ProfileEnvironment &env, const semcache::StatePtr &state) {
             return to_python(env.full_statistics(state));
           })
      .def("cache",
           [](const ProfileEnvironment &env, const semcache::StatePtr &state) {
             return to_python(env.cache(state));
           })
      .def("profile",
           [](const ProfileEnvironment &env) {
             return to_python(env.profile());
           })
      .def("history",
           [](const ProfileEnvironment &env) {
             return to_python(env.history());
           })
      .def("calls",
           [](const ProfileEnvironment &env) { return to_python(env.calls()); })
      .def("timings",
           [](const ProfileEnvironment &env) {
             return to_python(env.timings());
           })
      .def("config", [](const ProfileEnvironment &env) {
        return to_python(env.config());
      });
  module.def("search_runtime", [](const semcache::StateProblem &problem,
                                  ProfileEnvironment &environment,
                                  py::dict config, py::function predict) {
    const auto native = from_python(config);
    Json result;
    {
      py::gil_scoped_release release;
      result = semcache::search_runtime(
          problem, environment, native,
          [&](const std::vector<semcache::StatePtr> &states) {
            py::gil_scoped_acquire acquire;
            return prediction_from_python(
                predict(raw_batch(states, environment)), states.size());
          });
    }
    return to_python(result);
  });
  module.def("select_completions",
             [](py::handle scores,
                const std::vector<std::optional<double>> &times,
                const std::vector<std::optional<double>> &accuracies,
                std::optional<double> minimum) {
               return export_selection(semcache::select_completions(
                   scores_from_python(scores, times.size()), times, accuracies,
                   minimum));
             });
  py::class_<NativeSQLBackend, std::shared_ptr<NativeSQLBackend>>(
      module, "NativeSQLBackend")
      .def(py::init([](py::dict config) {
        const auto native = from_python(config);
        py::gil_scoped_release release;
        return std::make_shared<NativeSQLBackend>(native);
      }))
      .def("load",
           [](NativeSQLBackend &backend, const std::string &relation,
              const std::string &sql) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.load(relation, sql);
             }
             return to_python(result);
           })
      .def(
          "execute",
          [](NativeSQLBackend &backend, py::handle op, py::handle inputs,
             py::handle action) {
            const auto operation = from_python(op), bound = from_python(inputs),
                       chosen = from_python(action);
            Json result;
            {
              py::gil_scoped_release release;
              result = backend.execute(operation,
                                       bound.get<std::vector<Json>>(), chosen);
            }
            return to_python(result);
          },
          py::arg("op"), py::arg("inputs"), py::arg("action") = py::none())
      .def("prepare_inputs",
           [](NativeSQLBackend &backend, py::handle op, py::handle inputs) {
             const auto operation = from_python(op),
                        bound = from_python(inputs);
             std::vector<Json> result;
             {
               py::gil_scoped_release release;
               result = backend.prepare_inputs(operation,
                                               bound.get<std::vector<Json>>());
             }
             return to_python(result);
           })
      .def("join_pairs",
           [](NativeSQLBackend &backend, py::handle op, py::handle pairs,
              py::handle action) {
             const auto operation = from_python(op), bound = from_python(pairs),
                        chosen = from_python(action);
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.join_pairs(operation, bound, chosen);
             }
             return to_python(result);
           })
      .def("estimate_relation",
           [](NativeSQLBackend &backend, py::handle op, py::handle inputs) {
             const auto operation = from_python(op),
                        bound = from_python(inputs);
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.estimate_relation(
                   operation, bound.get<std::vector<Json>>());
             }
             return to_python(result);
           })
      .def("sample_population", [](NativeSQLBackend &backend, py::dict op) {
        const auto operation = from_python(op);
        Json result;
        {
          py::gil_scoped_release release;
          result = backend.sample_population(operation);
        }
        return to_python(result);
      })
      .def_property_readonly("events",
                             [](const NativeSQLBackend &backend) {
                               return to_python(backend.events());
                             })
      .def_property_readonly("observations",
                             [](const NativeSQLBackend &backend) {
                               return to_python(backend.observations());
                             })
      .def("selection_progress",
           [](const NativeSQLBackend &backend) {
             return to_python(backend.selection_progress());
           })
      .def("drain_events",
           [](NativeSQLBackend &backend) {
             return to_python(backend.drain_events());
           })
      .def("drain_observations",
           [](NativeSQLBackend &backend) {
             return to_python(backend.drain_observations());
           })
      .def("diagnostics",
           [](const NativeSQLBackend &backend) {
             return to_python(backend.diagnostics());
           })
      .def("timings",
           [](const NativeSQLBackend &backend) {
             return to_python(backend.timings());
           })
      .def("close", &NativeSQLBackend::close,
           py::call_guard<py::gil_scoped_release>());

  py::class_<NativeSampler, SearchEnvironment, std::shared_ptr<NativeSampler>>(
      module, "NativeSampler")
      .def(py::init(
          [](py::dict config, std::shared_ptr<NativeSQLBackend> backend) {
            const auto native = from_python(config);
            py::gil_scoped_release release;
            return std::make_shared<NativeSampler>(native, std::move(backend));
          }))
      .def("calibrate_time",
           [](NativeSampler &sampler, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.calibrate_time(state);
             }
             return to_python(result);
           })
      .def("observe", &NativeSampler::observe,
           py::call_guard<py::gil_scoped_release>())
      .def("refresh", &NativeSampler::refresh,
           py::call_guard<py::gil_scoped_release>())
      .def("accuracy", &NativeSampler::accuracy,
           py::call_guard<py::gil_scoped_release>())
      .def("accuracy_batch", &NativeSampler::accuracy_batch,
           py::call_guard<py::gil_scoped_release>())
      .def("statistics",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             return to_python(sampler.statistics(state));
           })
      .def("full_statistics",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             return to_python(sampler.full_statistics(state));
           })
      .def("samples",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             return to_python(sampler.samples(state));
           })
      .def("cache",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             return to_python(sampler.cache(state));
           })
      .def("profile",
           [](const NativeSampler &sampler) {
             return to_python(sampler.profile());
           })
      .def("history",
           [](const NativeSampler &sampler) {
             return to_python(sampler.history());
           })
      .def("calls",
           [](const NativeSampler &sampler) {
             return to_python(sampler.calls());
           })
      .def("timings",
           [](const NativeSampler &sampler) {
             return to_python(sampler.timings());
           })
      .def("telemetry", [](const NativeSampler &sampler) {
        return to_python(sampler.telemetry());
      })
      .def("config", [](const NativeSampler &sampler) {
        return to_python(sampler.config());
      });

  py::class_<semcache::NativeSearchEngine>(module, "NativeSearchEngine")
      .def_property_readonly("expansions",
                             &semcache::NativeSearchEngine::expansions)
      .def(py::init<const semcache::StateProblem &,
                    std::shared_ptr<SearchEnvironment>>())
      .def(
          "search",
          [](semcache::NativeSearchEngine &engine, const py::dict &config,
             const py::function &predict,
             const std::vector<std::uint64_t> &random_state,
             const std::vector<semcache::StatePtr> &original) {
            const semcache::ResidentOptions options{
                config["beam_width"].cast<std::size_t>(),
                config["max_expansions"].cast<std::size_t>(),
                config["max_seconds"].cast<double>(),
                config["training"].cast<bool>(),
                config["exploration_fraction"].cast<double>(),
                config.contains("a_min")
                    ? config["a_min"].cast<std::optional<double>>()
                    : std::nullopt,
                config.contains("trace_timings") &&
                    config["trace_timings"].cast<bool>(),
                config.contains("action_priorities")
                    ? config["action_priorities"]
                          .cast<semcache::ActionPriorities>()
                    : semcache::ActionPriorities{},
                config.contains("mode") ? config["mode"].cast<std::string>()
                                        : "heuristic_requests",
                config.contains("fixed_model_id")
                    ? config["fixed_model_id"]
                          .cast<std::optional<std::string>>()
                    : std::optional<std::string>{"gemma4-e2b"},
                config.contains("planner_workers")
                    ? config["planner_workers"].cast<std::size_t>()
                    : std::size_t(1),
                config.contains("preference")
                    ? config["preference"].cast<std::string>()
                    : "fastest",
                config.contains("layer_pruning")
                    ? config["layer_pruning"].cast<std::string>()
                    : "pareto",
                config.contains("max_structure_groups")
                    ? config["max_structure_groups"].cast<std::size_t>() : 15,
                config.contains("max_implementations_per_group")
                    ? config["max_implementations_per_group"].cast<std::size_t>() : 6};
            semcache::ResidentRandom random(random_state);
            const bool compact_requests =
                options.mode == "heuristic_requests" &&
                config.contains("compact_requests") &&
                config["compact_requests"].cast<bool>();
            const bool compact_features = config.contains("compact_features") &&
                                          config["compact_features"].cast<bool>();
            py::object feature_context = config.contains("feature_context")
                ? py::reinterpret_borrow<py::object>(config["feature_context"])
                : py::none();
            semcache::ResidentResult native;
            {
              py::gil_scoped_release release;
              native = engine.search(
                  options, random, original,
                  [&](const std::vector<semcache::StatePtr> &states) {
                    py::gil_scoped_acquire acquire;
                    py::object batch = feature_context.is_none()
                        ? py::none()
                        : native_feature_batch(feature_context, states, *engine.sampler());
                    if (batch.is_none())
                      batch = raw_batch(states, *engine.sampler(), compact_requests, compact_features);
                    return prediction_from_python(predict(batch),
                                                  states.size());
                  });
            }
            const auto &sampler = *engine.sampler();
            py::dict result;
            result["_environment"] = py::cast(engine.sampler());
            py::list completed, trajectories, trace;
            const auto export_started = std::chrono::steady_clock::now();
            std::unordered_map<std::uint64_t, py::dict> exported;
            StatisticsExport statistics;
            auto snapshot = [&](const semcache::StatePtr &state) {
              auto found = exported.find(state->handle_id);
              if (found == exported.end())
                found = exported
                            .emplace(state->handle_id,
                                     export_state(state, sampler, statistics))
                            .first;
              return found->second;
            };
            for (const auto &state : native.completed) {
              completed.append(snapshot(state));
              std::vector<semcache::StatePtr> path;
              for (auto current = state; current; current = current->parent)
                path.push_back(current);
              py::list rows;
              for (auto it = path.rbegin(); it != path.rend(); ++it)
                rows.append(snapshot(*it));
              trajectories.append(rows);
            }
            const auto export_ended = std::chrono::steady_clock::now();
            const auto export_states_seconds =
                std::chrono::duration<double>(export_ended - export_started)
                    .count();
            for (const auto &row : native.trace) {
              py::dict item;
              item["depth"] = row.depth;
              item["enumerated"] = row.enumerated;
              item["retained"] = row.retained;
              item["beam_width"] = row.beam_width;
              item["complete"] = row.complete;
              item["fronts"] = row.fronts;
              item["quality_pruned"] = row.quality_pruned;
              item["cost_pruned"] = row.cost_pruned;
              item["heuristic_guard"] = row.heuristic_guard;
              item["random_exploration"] = row.random_exploration;
              item["accuracy_threshold"] = row.accuracy_threshold;
              item["accuracy_relaxations"] = row.accuracy_relaxations;
              trace.append(item);
            }
            result["completed"] = completed;
            py::list group_trace;
            for (const auto &row : native.group_trace) {
              py::dict item;
              item["depth"] = row.depth;
              item["candidate_groups"] = row.candidate_groups;
              item["retained_group_sizes"] = row.retained_group_sizes;
              item["retained_group_means"] = row.retained_group_means;
              item["retained_groups"] = row.retained_group_sizes.size();
              group_trace.append(item);
            }
            result["group_trace"] = group_trace;
            result["join_filter_trace"] = to_python(native.join_filter_trace);
            result["trajectories"] = trajectories;
            result["scores"] = native.scores;
            result["times"] = native.prediction.times;
            result["accuracies"] = native.prediction.accuracies;
            result["time_reasons"] = native.prediction.time_reasons;
            result["selection"] = export_selection(native.selection);
            const auto context_started = std::chrono::steady_clock::now();
            result["profile"] = to_python(sampler.profile());
            result["history"] = to_python(sampler.history());
            result["calls"] = to_python(sampler.calls());
            result["timings"] = native.timings;
            if (options.trace_timings) {
              result["timing_spans"] = native.timing_spans;
              py::list diagnostics;
              for (const auto &row : native.state_diagnostics) {
                py::dict item, statistics;
                item["kind"] = row.kind;
                item["layer"] = row.layer;
                item["state_id"] = row.state->handle_id;
                item["parent_state_id"] =
                    row.state->parent ? py::cast(row.state->parent->handle_id)
                                      : py::none();
                item["depth"] = row.state->depth;
                item["remaining"] = row.state->remaining().size();
                item["roots"] = row.state->root_ids();
                item["complete"] = row.state->remaining().empty() &&
                                   row.state->roots.size() == 1;
                item["action"] =
                    row.state->parent
                        ? to_python(Json::parse(row.state->action_json))
                        : py::none();
                item["enumerate_transition_ns"] = row.enumerate_span;
                item["observe_ns"] = row.observe_span;
                item["observe_spans_ns"] = row.observe_spans;
                for (const auto &entry : row.statistics)
                  statistics[py::str(std::to_string(entry.first))] =
                      to_python(*entry.second);
                item["statistics"] = statistics;
                diagnostics.append(std::move(item));
              }
              result["state_diagnostics"] = diagnostics;
            }
            result["sampling_timings"] = to_python(sampler.timings());
            py::dict export_timings;
            export_timings["states_seconds"] = export_states_seconds;
            const auto context_ended = std::chrono::steady_clock::now();
            export_timings["context_seconds"] =
                std::chrono::duration<double>(context_ended - context_started)
                    .count();
            if (options.trace_timings) {
              auto ns = [](auto t) {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t.time_since_epoch())
                    .count();
              };
              export_timings["states_span_ns"] =
                  py::make_tuple(ns(export_started), ns(export_ended));
              export_timings["context_span_ns"] =
                  py::make_tuple(ns(context_started), ns(context_ended));
            }
            result["export_timings"] = export_timings;
            result["trace"] = trace;
            result["expansions"] = native.expansions;
            result["budget_exhausted"] = native.exhausted;
            result["action_budget_exhausted"] = native.action_budget_exhausted;
            result["heuristic_guard_status"] = native.heuristic_guard_status;
            result["heuristic_guard_complete"] =
                native.heuristic_guard_complete;
            return result;
          },
          py::arg("config"), py::arg("predict"), py::arg("random_state"),
          py::arg("original") = std::vector<semcache::StatePtr>{});
}
