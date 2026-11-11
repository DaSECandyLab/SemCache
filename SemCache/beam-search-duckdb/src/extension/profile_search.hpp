#pragma once
#include "search/resident_driver.hpp"

namespace semcache {
// SQL-only statistical fallback. The production learned-value callback remains
// the same resident predictor supplied by the Python adapter.
sampling::Json search_profile_configuration(const sampling::Json &config);
}
