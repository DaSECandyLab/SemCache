#pragma once
#include "state_core.hpp"
#include "sampling/environment.hpp"
#include <pybind11/pybind11.h>
void bind_native_features(pybind11::module_ &module);
// None means the environment does not support frozen native inference.
pybind11::object
native_feature_batch(pybind11::handle context,
                     const std::vector<semcache::StatePtr> &states,
                     const semcache::sampling::SearchEnvironment &environment);
