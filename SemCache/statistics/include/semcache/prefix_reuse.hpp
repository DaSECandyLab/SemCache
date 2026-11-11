#pragma once

#include <algorithm>
#include <cmath>

namespace semcache::statistics {
// E[(X-1)+] for X ~ Binomial(count, alpha), §3.1.1. Direct subtraction
// loses the O(alpha^2) result when alpha*count is small. In that region use
// the binomial series beginning at k=2, stopping at floating-point precision.
// The cutoff selects a numerically stable evaluation, not a sampling policy.
inline double repeated_prefix_hits(double alpha, double count) {
  if (!alpha || count <= 1.)
    return 0.;
  if (alpha == 1.)
    return count - 1.;
  const double visits = alpha * count;
  if (visits >= .125)
    return std::max(0., visits + std::expm1(count * std::log1p(-alpha)));
  double term = visits * ((count - 1.) * alpha) / 2., total = term;
  for (double k = 2.; term != 0.; ++k) {
    term *= -((count - k) * alpha) / (k + 1.);
    const double next = total + term;
    if (next == total)
      break;
    total = next;
  }
  return std::max(0., total);
}
}
