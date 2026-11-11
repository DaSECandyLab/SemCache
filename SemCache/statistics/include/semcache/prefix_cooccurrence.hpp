#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include "semcache/prefix_entries.hpp"

namespace semcache::statistics {
// Read-only: all rows were selected by the explicit offline table builder.
// The smaller marginal supplies the conditional sample; no row/LLM sampling
// is performed here. Missing matches intentionally yield zero.
template <class Json>
Json offline_prefix_common(const Json &bank, const Json &left,
                           const Json &right, const std::string &f,
                           const std::string &g) {
  hashed_prefix_entries(bank);
  Json common = {{"entry_encoding", prefix_entry_encoding}};
  const auto &fields = bank.at("fields");
  auto field_index = [&](const std::string &name) -> std::size_t {
    auto found = std::find(fields.begin(), fields.end(), Json(name));
    if (found == fields.end())
      throw std::invalid_argument("Missing offline cooccurrence column: " +
                                  name);
    return found - fields.begin();
  };
  const auto fi = field_index(f), gi = field_index(g);
  const auto &samples = bank.at("samples"), &rows = bank.at("rows");
  for (auto depth = left.at("sketches").begin();
       depth != left.at("sketches").end(); ++depth) {
    if (!right.at("sketches").contains(depth.key()))
      continue;
    const auto left_counts = prefix_counts(depth.value()),
               right_counts = prefix_counts(right.at("sketches").at(depth.key()));
    for (const auto &[hash, count] : left_counts) {
      const auto other = right_counts.find(hash);
      if (other == right_counts.end())
        continue;
      const auto key = encode_prefix_hash(hash);
      const double a = count, b = other->second;
      const bool from_left = a < b || (a == b && f < g);
      const auto &source = from_left ? f : g;
      double estimate = 0;
      if (samples.contains(source) &&
          samples.at(source).contains(depth.key()) &&
          samples.at(source).at(depth.key()).contains(key)) {
        const auto &ids = samples.at(source).at(depth.key()).at(key);
        const auto &sketch = from_left ? depth.value()
                                       : right.at("sketches").at(depth.key());
        const auto &prefix = sketch.at("prefixes").at(key)
                                  .template get_ref<const std::string &>();
        std::size_t matches = 0;
        for (const auto &id : ids) {
          const auto &value = rows.at(id.template get<std::size_t>())
                                  .at(from_left ? gi : fi)
                                  .template get_ref<const std::string &>();
          matches += value.compare(0, prefix.size(), prefix) == 0;
        }
        if (!ids.empty())
          estimate = std::min(a, b) * double(matches) / ids.size();
      }
      common[depth.key()][key] = estimate;
    }
  }
  return common;
}
}
