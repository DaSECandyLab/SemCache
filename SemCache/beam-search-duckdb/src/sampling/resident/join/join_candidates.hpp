#pragma once
#include "../../common/inputs.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace semcache::sampling::resident_detail {
struct JoinCoordinate {
  std::array<std::size_t, 2> indices;
  std::array<std::size_t, 2> members;
};

// Own only a prefix of coordinates. Input publications outlive the cursor;
// probabilities belong to a requested prefix, not to the stored coordinate.
class JoinCoordinateCursor {
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  JoinCoordinateCursor(const SampleInputs &inputs,
                       const std::set<std::string> &fields,
                       std::size_t index_limit, std::uint64_t seed,
                       const std::set<std::string> &dependencies = {});
  ~JoinCoordinateCursor();
  void extend(std::size_t target);
  std::size_t size() const;
  std::size_t population() const;
  std::size_t group_population() const;
  const JoinCoordinate &at(std::size_t index) const;
  double inclusion(std::size_t side, std::size_t members,
                   std::size_t prefix = 0) const;
};
struct JoinCandidate {
  Json unit;
  double expansion;
  std::vector<double> row_inclusions;
  std::array<std::size_t, 2> indices;
};
struct JoinCandidates {
  std::vector<JoinCandidate> probes;
  std::size_t population = 0;
  std::size_t group_population = 0;
  bool compact_values = false;
};

// Fixed, label-independent sample. Groups retain bag multiplicity; large or
// factorized sides use virtual indices without scanning the Cartesian space.
JoinCandidates join_candidates(const SampleInputs &inputs,
                               const std::set<std::string> &fields,
                               std::size_t limit, std::size_t index_limit,
                               std::uint64_t seed,
                               const std::set<std::string> &dependencies = {});
// Input sides are already final random table samples, in independent random
// order. Pair their coordinates directly; do not sample another Cartesian pool.
JoinCandidates direct_join_candidates(const SampleInputs &inputs,
                                      std::size_t limit);
SharedSamples direct_product_samples(const std::vector<const Json *> &factors,
                                     std::size_t limit);
}
