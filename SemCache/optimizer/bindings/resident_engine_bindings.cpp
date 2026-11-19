#include "resident_bindings_internal.hpp"
#include <chrono>

namespace semcache::resident_bindings_detail {
void bind_resident_engine(py::module_ &module) {
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
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.statistics(state);
             }
             return to_python(result);
           })
      .def("full_statistics",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.full_statistics(state);
             }
             return to_python(result);
           })
      .def("samples",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.samples(state);
             }
             return to_python(result);
           })
      .def("cache",
           [](const NativeSampler &sampler, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.cache(state);
             }
             return to_python(result);
           })
      .def("profile",
           [](const NativeSampler &sampler) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.profile();
             }
             return to_python(result);
           })
      .def("history",
           [](const NativeSampler &sampler) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.history();
             }
             return to_python(result);
           })
      .def("calls",
           [](const NativeSampler &sampler) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.calls();
             }
             return to_python(result);
           })
      .def("timings",
           [](const NativeSampler &sampler) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.timings();
             }
             return to_python(result);
           })
      .def("telemetry",
           [](const NativeSampler &sampler) {
             Json result;
             {
               py::gil_scoped_release release;
               result = sampler.telemetry();
             }
             return to_python(result);
           })
      .def("config", [](const NativeSampler &sampler) {
        Json result;
        {
          py::gil_scoped_release release;
          result = sampler.config();
        }
        return to_python(result);
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
} // namespace semcache::resident_bindings_detail
