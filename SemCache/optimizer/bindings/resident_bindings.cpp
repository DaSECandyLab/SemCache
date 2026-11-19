#include "resident_bindings_internal.hpp"

namespace semcache::resident_bindings_detail {
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
                   bool compact_requests,
                   bool compact_features) {
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
py::dict StatisticsExport::fields(const semcache::StatePtr &state,
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
py::dict export_state(const semcache::StatePtr &state,
                      const SearchEnvironment &sampler,
                      StatisticsExport &statistics) {
  py::dict row;
  row["state"] = state;
  row["statistics"] = statistics.fields(state, sampler);
  return row;
}
} // namespace semcache::resident_bindings_detail

using namespace semcache::resident_bindings_detail;
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
             Json result;
             {
               py::gil_scoped_release release;
               result = env.statistics(state);
             }
             return to_python(result);
           })
      .def("full_statistics",
           [](const ProfileEnvironment &env, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.full_statistics(state);
             }
             return to_python(result);
           })
      .def("cache",
           [](const ProfileEnvironment &env, const semcache::StatePtr &state) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.cache(state);
             }
             return to_python(result);
           })
      .def("profile",
           [](const ProfileEnvironment &env) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.profile();
             }
             return to_python(result);
           })
      .def("history",
           [](const ProfileEnvironment &env) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.history();
             }
             return to_python(result);
           })
      .def("calls",
           [](const ProfileEnvironment &env) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.calls();
             }
             return to_python(result);
           })
      .def("timings",
           [](const ProfileEnvironment &env) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.timings();
             }
             return to_python(result);
           })
      .def("config",
           [](const ProfileEnvironment &env) {
             Json result;
             {
               py::gil_scoped_release release;
               result = env.config();
             }
             return to_python(result);
           });
  module.def(
      "search_runtime",
      [](const semcache::StateProblem &problem, ProfileEnvironment &environment,
         py::dict config, py::function predict, py::object feature_context) {
        const auto native = from_python(config);
        Json result;
        {
          py::gil_scoped_release release;
          result = semcache::search_runtime(
              problem, environment, native,
              [&](const std::vector<semcache::StatePtr> &states) {
                py::gil_scoped_acquire acquire;
                py::object batch = feature_context.is_none()
                    ? py::none()
                    : native_feature_batch(feature_context, states, environment);
                if (batch.is_none())
                  batch = raw_batch(states, environment);
                return prediction_from_python(predict(batch), states.size());
              });
        }
        return to_python(result);
      },
      py::arg("problem"), py::arg("environment"), py::arg("config"),
      py::arg("predict"), py::arg("feature_context") = py::none());
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
                               Json result;
                               {
                                 py::gil_scoped_release release;
                                 result = backend.events();
                               }
                               return to_python(result);
                             })
      .def_property_readonly("observations",
                             [](const NativeSQLBackend &backend) {
                               Json result;
                               {
                                 py::gil_scoped_release release;
                                 result = backend.observations();
                               }
                               return to_python(result);
                             })
      .def("selection_progress",
           [](const NativeSQLBackend &backend) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.selection_progress();
             }
             return to_python(result);
           })
      .def("drain_events",
           [](NativeSQLBackend &backend) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.drain_events();
             }
             return to_python(result);
           })
      .def("drain_observations",
           [](NativeSQLBackend &backend) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.drain_observations();
             }
             return to_python(result);
           })
      .def("diagnostics",
           [](const NativeSQLBackend &backend) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.diagnostics();
             }
             return to_python(result);
           })
      .def("timings",
           [](const NativeSQLBackend &backend) {
             Json result;
             {
               py::gil_scoped_release release;
               result = backend.timings();
             }
             return to_python(result);
           })
      .def("close", &NativeSQLBackend::close,
           py::call_guard<py::gil_scoped_release>());
  bind_resident_engine(module);
}
