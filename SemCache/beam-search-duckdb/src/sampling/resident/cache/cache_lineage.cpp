#include "../detail.hpp"
#include "semcache/prefix_cooccurrence.hpp"

namespace semcache::sampling {
CrossDependence
NativeSampler::Impl::cache_dependence(const Observation &source,
                                      const Observation &target) const {
  if (source.cache_relation.empty() ||
      source.cache_relation != target.cache_relation)
    return {};
  if (source.cache_field == target.cache_field)
    return {true, {}};
  const auto &relation = relations.at(source.cache_relation);
  if (!relation.offline_cooccurrence ||
      !relation.offline_columns.count(source.cache_field) ||
      !relation.offline_columns.count(target.cache_field))
    throw std::invalid_argument(
        "Missing offline same-table prefix cooccurrence statistics; "
        "build table statistics before value-guided search");
  const auto key = std::make_tuple(
      source.cache_relation, std::min(source.cache_field, target.cache_field),
      std::max(source.cache_field, target.cache_field));
  if (auto found = prefix_cooccurrences.find(key);
      found != prefix_cooccurrences.end())
    return {false, found->second};
  auto owner =
      std::make_shared<const Json>(semcache::statistics::offline_prefix_common(
          *relation.offline_cooccurrence, *source.summary, *target.summary,
          relation.offline_columns.at(source.cache_field),
          relation.offline_columns.at(target.cache_field)));
  prefix_cooccurrences.emplace(key, owner);
  return {false, std::move(owner)};
}
}
