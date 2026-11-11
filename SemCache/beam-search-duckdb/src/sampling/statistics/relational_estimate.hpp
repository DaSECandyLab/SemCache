#pragma once
#include "statistics.hpp"

namespace semcache::sampling {
// Both rates must describe the same local input domain. Keep source evidence
// separate so propagation/replay never averages an already averaged value.
Json mean_relational_estimate(const Json &sampled, const Json &duckdb);
}
