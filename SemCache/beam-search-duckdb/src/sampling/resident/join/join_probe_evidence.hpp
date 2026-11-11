#pragma once
#include "join_candidates.hpp"
#include "join_stream.hpp"
#include "../numeric.hpp"
#include "../survival/source_evidence.hpp"
#include <functional>

namespace semcache::sampling::resident_detail {
using QuotaKey = std::pair<std::string, std::size_t>;
using QuotaBank = std::map<QuotaKey, std::pair<double, std::size_t>>;
class JoinProbeEvidence;

// A publication freezes numeric/source evidence, while borrowing append-only
// coordinates. Later observations cannot change this prefix's facts.
struct JoinProbeSnapshot {
  std::shared_ptr<const JoinProbeEvidence> owner;
  std::size_t prefix = 0;
  double input_mass = 0, output_mass = 0, effective = 0, normalization = 1;
  bool census = false;
  SourceEvidence arrived, probed, passed;
  std::vector<std::size_t> matches;
  std::size_t novel_matches = 0;
};

class JoinProbeEvidence
    : public std::enable_shared_from_this<JoinProbeEvidence> {
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  using Key = std::function<std::string(std::size_t, std::size_t)>;
  using Label = std::function<bool(const std::string &)>;
  using Stratum = std::function<QuotaKey(const Json &)>;
  JoinProbeEvidence(OwnedInputs inputs, const std::set<std::string> &fields,
                    const std::set<std::string> &dependencies,
                    std::size_t index_limit, std::size_t origin_limit,
                    std::uint64_t seed, QuotaBank quotas, Stratum stratum,
                    std::size_t origin_scale = 1);
  ~JoinProbeEvidence();
  // Set once for this immutable input/cohort; dictionaries survive extensions.
  void configure_request(RequestKeyPlan plan);
  std::vector<std::size_t> prepare(std::size_t limit);
  std::vector<std::size_t> prepare(std::size_t limit, const Key &key);
  const std::string &key(std::size_t index) const;
  // Prepare label-independent source inclusions while the SQL transport owns
  // its separate input copies. This never resolves a probe or publishes it.
  void prepare_snapshot(std::size_t prefix, double input_mass) const;
  void complete(const Label &label);
  std::shared_ptr<const JoinProbeSnapshot> snapshot(std::size_t prefix,
                                                    double input_mass) const;
  std::vector<Json> unit(std::size_t index,
                         const JoinProbeSnapshot *snapshot = nullptr,
                         bool full_values = false) const;
  Json output(std::size_t index, const JoinProbeSnapshot &snapshot) const;
  const std::string &identity(std::size_t index) const;
  std::size_t identity_computations() const;
  std::shared_ptr<const JoinOutputStream::GeneratedPopulation>
  output_population(std::shared_ptr<const JoinProbeSnapshot> snapshot) const;
  const Json &value(std::size_t index, const std::string &field,
                    bool right) const;
  // Same parameter grouping/order, but explicit State quotas and row recovery
  // only for groups selected for individual quality evaluation.
  std::vector<std::pair<std::size_t, double>>
  quality(const JoinProbeSnapshot &snapshot, const Counts &counts) const;
  std::size_t population() const;
  std::size_t group_population() const;
  std::size_t size() const;
  std::size_t distinct(const std::string &field, std::size_t prefix) const;
  const OwnedInputs &inputs() const;
};
}
