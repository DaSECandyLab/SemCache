#pragma once
#include "../sampling/profile/environment.hpp"
#include "resident_driver.hpp"

namespace semcache {
sampling::Json search_runtime(const StateProblem &problem,
                              sampling::ProfileEnvironment &environment,
                              const sampling::Json &config,
                              const ResidentPredictor &predict);
}
