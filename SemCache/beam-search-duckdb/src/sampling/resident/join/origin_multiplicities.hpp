#pragma once
#include "../populations.hpp"

namespace semcache::sampling::resident_detail {
// Preserve lineage encounter order and the factorized bag multiplicity.
inline bool origin_multiplicities(const Json &rows, double scale,
                                  std::size_t limit,
                                  std::map<std::string, double> &counts) {
  if (product(rows)) {
    const auto &parts = rows.at("product");
    return origin_multiplicities(parts[0], scale * row_count(parts[1]), limit,
                                 counts) &&
           origin_multiplicities(parts[1], scale * row_count(parts[0]), limit,
                                 counts);
  }
  if (rows.size() > limit)
    return false;
  for (const auto &row : rows)
    for (const auto &origin : row.at("lineage"))
      counts[canonical(origin)] += scale;
  return true;
}
}
