#pragma once

#include "../statistics.hpp"
#include "../../common/exact_integer.hpp"
#include "semcache/prefix_entries.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace semcache::sampling {
namespace stats_detail {

using Hash = semcache::statistics::PrefixHash;

Hash rank(std::string_view value);
std::string float_text(double value);
std::string encode(const Json &value, bool sorted = true);

struct Sum {
  double hi = 0, lo = 0;
  void add(double x) {
    double t = hi + x;
    lo += std::abs(hi) >= std::abs(x) ? (hi - t) + x : (x - t) + hi;
    hi = t;
  }
  double get() const { return lo && std::isfinite(lo) ? hi + lo : hi; }
};

double present(double a, double count);
double integral(std::map<std::size_t, double> values,
                std::optional<std::size_t> max_observed = std::nullopt);
std::optional<std::size_t> summary_observed_max(const Json &s);
std::optional<std::size_t> cross_observed_max(const Json &a, const Json &b);
bool compatible(const Json &a, const Json &b);
void validate_summary(const Json &s, bool field_serializer = true);
double fraction(const Hash &h);

struct Sample {
  double tau = 1;
  std::map<Hash, std::uint64_t> counts;
};
Sample sample(const Json &sketch);

struct CrossSample {
  double tau = 1;
  std::map<Hash, std::array<std::uint64_t, 2>> counts;
};
std::size_t cross_capacity(const Json &a, const Json &b);

template <class Counts>
CrossSample cross_sample_counts(const Counts &left, const Counts &right, std::size_t capacity) {
  // Decode fixed-size digests once, then merge sorted hashes directly. Neither
  // membership nor bottom-k thresholds hash/compare the original prefix text.
  auto l = left.begin(), r = right.begin();
  std::size_t visited = 0;
  CrossSample out;
  while (l != left.end() || r != right.end()) {
    const auto hash = r == right.end() || (l != left.end() && l->first < r->first)
                          ? l->first : r->first;
    if (++visited == capacity) {
      out.tau = fraction(hash);
      break;
    }
    const bool in_left = l != left.end() && l->first == hash,
               in_right = r != right.end() && r->first == hash;
    if (in_left && in_right)
      out.counts.emplace(hash, std::array<std::uint64_t, 2>{l->second, r->second});
    if (in_left)
      ++l;
    if (in_right)
      ++r;
  }
  return out;
}

CrossSample cross_sample(const Json &a, const Json &b);

struct DecodedSketch {
  std::vector<std::pair<Hash, std::uint64_t>> counts;
  double tau = 1.;
  std::size_t sample_size = 0;
};
DecodedSketch decode_sketch(const Json &sketch);
bool number(const Json &v);

} // namespace stats_detail
} // namespace semcache::sampling
