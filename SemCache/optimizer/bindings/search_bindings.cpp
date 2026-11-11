#include "observation_core.hpp"
#include "resident_driver.hpp"
#include "semcache/prefix_sketch.h"
#include "state_bindings.hpp"
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
void bind_cache_terms(pybind11::module_ &module);
void bind_feature_buffers(pybind11::module_ &module);
void bind_native_features(pybind11::module_ &module);
void bind_resident(pybind11::module_ &module);
namespace py = pybind11;

PYBIND11_MODULE(_search_core, module) {
  module.doc() =
      "Python adapters for the resident DuckDB beam_search extension";
  module.attr("backend") = "cpp17-resident";
  module.def("extension_path", [] {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&semcache::select_completions),
                &info) ||
        !info.dli_fname)
      throw std::runtime_error("Cannot locate loaded beam_search extension");
    return std::string(info.dli_fname);
  });
  module.def("statistics_path", [] {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&sem_prefix_version), &info) ||
        !info.dli_fname)
      throw std::runtime_error(
          "Cannot locate loaded semcache statistics library");
    return std::string(info.dli_fname);
  });
  bind_state(module);
  bind_observations(module);
  bind_cache_terms(module);
  bind_feature_buffers(module);
  bind_native_features(module);
  bind_resident(module);
}
