#pragma once
#include "../../common/inputs.hpp"
#include "../populations.hpp"

namespace semcache::sampling::resident_detail {
// Source inclusion does not depend on the field. Gather it once per boundary;
// every field retains its own lengths, strata and confidence evidence.
struct SourceEvidence {
  using Inclusion = std::map<std::string, Json>;
  std::map<std::string, Inclusion> sources;
  bool approximate = false;

  SourceEvidence() = default;

  explicit SourceEvidence(const SampleInputs &inputs) {
    for (const auto &rows : inputs)
      collect(rows);
  }

private:
  void collect(const Json &rows) {
    if (product(rows)) {
      for (const auto &part : rows.at("product"))
        collect(part);
      return;
    }
    for (const auto &row : rows) {
      approximate |= row.value("_sample_origin_approximate", false) ||
                     row.value("_sample_match_incomplete", false);
      const auto found = row.find("_sample_origin_probabilities");
      for (const auto &origin : row.at("lineage")) {
        const Json *probability = nullptr;
        if (found != row.end()) {
          const auto entry = found->find(canonical(origin));
          if (entry != found->end())
            probability = &entry.value();
        }
        const auto value = probability ? *probability
                           : row.contains("_sample_weight_factor")
                               ? Json(nullptr)
                               : Json(1.);
        auto &bank = sources[relkey(origin[0])];
        auto [entry, inserted] =
            bank.emplace(origin[1].get<std::string>(), value);
        if (!inserted && (entry->second.is_null() || value.is_null()))
          entry->second = nullptr;
        else if (!inserted)
          entry->second =
              std::max(entry->second.get<double>(), value.get<double>());
      }
    }
  }
};
}
