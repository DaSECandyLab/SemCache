#pragma once
#include "join_probe_evidence.hpp"
#include "../origin_multiplicities.hpp"
#include "../../requests/request_memo.hpp"
#include <cstring>
#include <deque>
#include <unordered_map>

namespace semcache::sampling::resident_detail {
struct JoinProbeEvidence::Impl {
  struct Origin {
    std::string key, relation, id;
    std::optional<double> prior;
  };
  struct Row {
    const Json *borrowed = nullptr;
    std::optional<Json> owned;
    std::vector<Origin> origins;
    std::vector<QuotaKey> recipe;
    double weight = 0;
    const Json &metadata() const { return borrowed ? *borrowed : *owned; }
  };
  struct Totals {
    Sum mass, squares, positive;
  };
  struct Record {
    std::size_t parameter;
    bool novel = false, success = false;
    double weight = 0;
    Totals totals;
  };
  struct InclusionClass {
    std::size_t side, members;
    std::optional<double> prior;
    bool approximate, incomplete;
    bool operator<(const InclusionClass &b) const {
      return std::tie(side, members, prior, approximate, incomplete) <
             std::tie(b.side, b.members, b.prior, b.approximate, b.incomplete);
    }
  };
  struct Source {
    std::string relation, id;
    std::map<InclusionClass, std::size_t> classes;
  };
  OwnedInputs inputs;
  std::set<std::string> fields, dependencies, payload;
  QuotaBank quotas;
  Stratum stratum;
  JoinCoordinateCursor cursor;
  std::array<std::map<std::size_t, Row>, 2> rows;
  std::array<std::map<std::string, double>, 2> copies;
  std::array<bool, 2> indexed{};
  std::array<std::size_t, 2> source_sizes{};
  std::size_t origin_min, origin_scale;
  static std::size_t largest_leaf(const Json &rows) {
    if (!product(rows))
      return rows.size();
    return std::max(largest_leaf(rows.at("product")[0]),
                    largest_leaf(rows.at("product")[1]));
  }
  std::size_t source_limit(std::size_t prefix) const {
    const auto scaled =
        prefix > std::numeric_limits<std::size_t>::max() / origin_scale
            ? std::numeric_limits<std::size_t>::max()
            : prefix * origin_scale;
    return std::max(origin_min, scaled);
  }
  bool sources_known(std::size_t side, std::size_t prefix) const {
    return source_sizes[side] <= source_limit(prefix);
  }
  void index_sources(std::size_t prefix) {
    for (std::size_t side = 0; side < inputs.size(); ++side)
      if (!indexed[side] && sources_known(side, prefix)) {
        indexed[side] = origin_multiplicities(
            *inputs[side], 1., source_limit(prefix), copies[side]);
        if (!indexed[side])
          throw std::logic_error("Invalid source index size");
      }
  }
  std::map<std::string, Source> arrived, passed;
  struct Representative {
    std::size_t first;
    double weight, base_weight;
  };
  std::array<std::map<std::string, Representative>, 2> representatives;
  std::vector<Record> records;
  std::vector<std::string> parameters;
  std::map<std::string, std::size_t> parameter_index;
  std::unordered_map<const Json *, std::string> signatures;
  std::map<std::string, std::map<std::string, std::size_t>> passed_values;
  std::vector<std::size_t> novel, deferred;
  std::vector<std::size_t> positive_novel, positive_deferred;
  mutable std::unordered_map<std::size_t, std::string> identities;
  mutable std::unordered_map<std::string, std::size_t> identity_coordinates;
  mutable std::size_t identity_indexed_prefix = 0;
  mutable bool unique_identities = true;
  mutable std::size_t identity_computations = 0;
  std::map<std::string, std::set<std::string>> seen_values;
  std::size_t completed = 0, requested = 0;
  std::unique_ptr<RequestMemo> request_memo;
  struct Publication {
    std::size_t prefix;
    std::uint64_t mass;
    std::weak_ptr<const JoinProbeSnapshot> snapshot;
  };
  // Snapshots own the evidence. Weak entries avoid an ownership cycle, and a
  // small FIFO bounds stale keys without retaining old publications.
  mutable std::deque<Publication> publications;
  // This unpublished preparation has no owner pointer (and thus no ownership
  // cycle). Labels and positive source evidence are filled only at snapshot().
  mutable std::shared_ptr<JoinProbeSnapshot> prepared;
  static constexpr std::size_t publication_capacity = 4;

  Impl(OwnedInputs in, const std::set<std::string> &f,
       const std::set<std::string> &d, std::size_t index_limit,
       std::size_t origin_limit, std::uint64_t seed, QuotaBank q, Stratum s,
       std::size_t scale)
      : inputs(std::move(in)), fields(f), dependencies(d), payload(f),
        quotas(std::move(q)), stratum(std::move(s)),
        cursor(SampleInputs(inputs), fields, index_limit, seed, dependencies),
        origin_min(origin_limit), origin_scale(scale) {
    payload.insert(dependencies.begin(), dependencies.end());
    for (std::size_t side = 0; side < inputs.size(); ++side)
      source_sizes[side] = largest_leaf(*inputs[side]);
  }
  const std::string &signature(const Json &value);
  double quota_weight(const std::vector<QuotaKey> &recipe,
                      const Counts *counts) const;
  const Row &row(std::size_t side, std::size_t index);
  void add_sources(std::map<std::string, Source> &bank, const Row &row,
                   std::size_t side, std::size_t members,
                   std::size_t position);
  SourceEvidence sources(const std::map<std::string, Source> &bank,
                         std::size_t prefix, bool census, bool matches,
                         bool *incomplete = nullptr) const;
  Json sampled_row(std::size_t side, std::size_t index, std::size_t members,
                   const JoinProbeSnapshot &snapshot, bool full) const;
};
}
