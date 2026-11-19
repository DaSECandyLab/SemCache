#pragma once
#include "bridge.hpp"
#include "features/native_features.hpp"
#include "drivers/resident_driver.hpp"
#include "pruning/grouped_pruning.hpp"
#include "pruning/join_filter_cost.hpp"
#include "drivers/runtime_driver.hpp"
#include "sampling/profile/environment.hpp"
#include "sampling/resident/sampler.hpp"
#include "sampling/statistics/relational_estimate.hpp"
#include <pybind11/stl.h>

namespace semcache::resident_bindings_detail {
namespace py = pybind11;
using semcache::sampling::from_python;
using semcache::sampling::Json;
using semcache::sampling::NativeSampler;
using semcache::sampling::NativeSQLBackend;
using semcache::sampling::ProfileEnvironment;
using semcache::sampling::SearchEnvironment;
using semcache::sampling::to_python;

py::object frozen_scalar(const Json &value, const py::object &dictionary,
                         const py::object &array);
py::dict raw_batch(const std::vector<semcache::StatePtr> &states,
                   const SearchEnvironment &sampler,
                   bool compact_requests = false,
                   bool compact_features = false);
semcache::Scores scores_from_python(py::handle value, std::size_t expected);
semcache::ResidentPrediction prediction_from_python(py::handle value,
                                                    std::size_t expected);
py::dict export_selection(const semcache::CompletionSelection &native);
struct StatisticsExport {
  std::unordered_map<const Json *,
                     std::pair<std::shared_ptr<const Json>, py::object>>
      immutable;
  py::object freeze = py::module_::import("planning.observations.evidence")
                          .attr("freeze_evidence");
  py::dict fields(const semcache::StatePtr &state,
                  const SearchEnvironment &sampler);
};
py::dict export_state(const semcache::StatePtr &state,
                      const SearchEnvironment &sampler,
                      StatisticsExport &statistics);
void bind_resident_engine(py::module_ &module);
} // namespace semcache::resident_bindings_detail
