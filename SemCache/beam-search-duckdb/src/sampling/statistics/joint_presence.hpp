#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semcache::sampling {
inline double prefix_presence(double rate, double count) {
  if (!std::isfinite(rate) || rate < 0 || rate > 1 || !std::isfinite(count) ||
      count < 0)
    throw std::invalid_argument("Invalid prefix survival or frequency");
  if (!count || !rate)
    return 0.;
  return rate == 1. ? 1. : -std::expm1(count * std::log1p(-rate));
}
// Exact for independent homogeneous base rows with nested same-row events.
// The supplied rates and overlap can themselves be sampling approximations.
inline double joint_presence(double source, double target, double cf, double cg,
                             double common) {
  if (!std::isfinite(common) || common < 0 || common > std::min(cf, cg))
    throw std::invalid_argument("Common frequency exceeds marginal frequency");
  const auto a = prefix_presence(source, cf), b = prefix_presence(target, cg);
  const auto correction = (1. - a) *
                          (1. - prefix_presence(target, cg - common)) *
                          prefix_presence(target, common);
  return std::min(std::min(a, b),
                  std::max(std::max(0., a + b - 1.), a * b + correction));
}
}
