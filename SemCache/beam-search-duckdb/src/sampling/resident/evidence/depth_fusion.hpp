#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace semcache::sampling::resident_detail {
struct DepthEvidence {
  double rate = .5, effective = 0;
  bool census = false;
  explicit operator bool() const {
    return std::isfinite(rate) && rate >= 0 && rate <= 1 &&
           std::isfinite(effective) && effective > 0;
  }
  double point() const {
    // A finite all-negative sample is evidence, not a population proof.
    return !census && rate == 0 ? .5 / (effective + 1.) : rate;
  }
};
struct DepthFusion {
  double rate = .5, strength = 0, primary_strength = 0;
  // Direct, same-depth shared, neighbour, scalar, independent, DuckDB prior.
  std::array<double, 6> weights{};
  bool repair = true;
};
inline double agreement(const DepthEvidence &a, const DepthEvidence &b) {
  if (!a || !b)
    return 1.;
  const double variance = 1. / (a.effective + 1.) + 1. / (b.effective + 1.);
  const double delta = a.point() - b.point();
  return variance / (variance + delta * delta);
}
inline DepthFusion fuse_depth(DepthEvidence direct, DepthEvidence shared,
                              DepthEvidence neighbour, DepthEvidence scalar,
                              DepthEvidence independent, double scale,
                              DepthEvidence duckdb = {}) {
  for (auto *value :
       {&direct, &shared, &neighbour, &scalar, &independent, &duckdb})
    if (!*value)
      *value = {};
  DepthFusion out;
  scale = std::max(1., scale);
  if (direct && direct.census)
    return {direct.rate,
            direct.effective,
            direct.effective,
            {1, 0, 0, 0, 0},
            false};
  const double n = direct ? direct.effective : 0.;
  const double s = shared ? shared.effective * agreement(direct, shared) : 0.;
  const double i =
      neighbour ? neighbour.effective * agreement(direct, neighbour) : 0.;
  // Auxiliary estimates overlap each other and local rows. They are not
  // independent pseudo-observations: cap their JOINT influence continuously.
  const double cap = scale * scale / (scale + n);
  const double auxiliary = std::min(cap, std::max(s, i));
  out.weights[0] = n;
  if (s + i > 0) {
    out.weights[1] = auxiliary * s / (s + i);
    out.weights[2] = auxiliary * i / (s + i);
  }
  const double mass = n + auxiliary;
  out.primary_strength = std::max(n, auxiliary);
  if (mass > 0) {
    for (auto &w : out.weights)
      w /= mass;
    out.rate = out.weights[0] * direct.point() +
               out.weights[1] * shared.point() +
               out.weights[2] * neighbour.point();
  }
  // Same-row marginals have a different conditional domain. At most one
  // unit of proxy strength; they cannot manufacture depth coverage.
  const double o = scalar ? scalar.effective / (scalar.effective + 1.) : 0.;
  const double f =
      independent ? independent.effective * agreement(scalar, independent) : 0.;
  const double db = duckdb ? duckdb.effective / (duckdb.effective + 1.) : 0.;
  const double second = o + f + db;
  const double gap = std::max(0., 1. - out.primary_strength / scale);
  if (second > 0 && gap > 0) {
    const double fraction = mass > 0 ? gap : 1.;
    for (auto &w : out.weights)
      w *= 1. - fraction;
    out.weights[3] = fraction * o / second;
    out.weights[4] = fraction * f / second;
    out.weights[5] = fraction * db / second;
    out.rate = (mass > 0 ? (1. - fraction) * out.rate : 0.) +
               out.weights[3] * scalar.point() +
               out.weights[4] * independent.point() +
               out.weights[5] * duckdb.point();
  }
  out.strength = std::max(out.primary_strength, f);
  out.repair = out.strength < scale;
  out.rate = std::clamp(out.rate, 0., 1.);
  return out;
}
inline DepthEvidence
interpolate_depth(std::size_t depth,
                  const std::map<std::size_t, DepthEvidence> &anchors) {
  if (!depth || anchors.empty())
    return {};
  auto upper = anchors.upper_bound(depth);
  auto lower = anchors.lower_bound(depth);
  if (lower == anchors.begin())
    lower = anchors.end();
  else
    --lower;
  auto discount = [depth](auto at) {
    const double distance = std::abs(std::log2(double(depth) / at->first));
    return at->second.effective / (1. + distance);
  };
  if (lower != anchors.end() && upper != anchors.end()) {
    const double ratio =
        double(depth - lower->first) / (upper->first - lower->first);
    const double p =
        lower->second.point() * (1. - ratio) + upper->second.point() * ratio;
    // Nested depth populations: never add the two sample counts.
    return {p, std::min(discount(lower), discount(upper))};
  }
  if (lower != anchors.end())
    return {lower->second.point(), discount(lower)};
  if (upper != anchors.end())
    return {upper->second.point(), discount(upper)};
  return {};
}
}
