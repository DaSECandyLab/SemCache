#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::sampled_field_rates(
    const std::string &key, const std::string &name,
    const resident_detail::SourceEvidence &reached,
    const resident_detail::SourceEvidence &eligible, bool conditional,
    bool complete, bool match_incomplete,
    const std::vector<std::size_t> *selected) const {
  PhaseTimer timer(phase_metrics, "sampled_depth_rates", trace_timings);
  static const resident_detail::SourceEvidence::Inclusion empty;
  const auto passed_it = reached.sources.find(key),
             candidates_it = eligible.sources.find(key);
  const auto &passed =
      passed_it == reached.sources.end() ? empty : passed_it->second;
  const auto &candidates =
      candidates_it == eligible.sources.end() ? empty : candidates_it->second;
  const bool union_approximation =
      match_incomplete || reached.approximate || eligible.approximate;
  const auto &relation = relations.at(key);
  std::set<std::size_t> depths{0};
  for (auto it = relation.summaries.at(name)->at("sketches").begin();
       it != relation.summaries.at(name)->at("sketches").end(); ++it)
    depths.insert(std::stoull(it.key()));
  resident_detail::SampledRateKey cache_key;
  cache_key.text(key);
  cache_key.text(name);
  cache_key.scalar(relation.version);
  cache_key.scalar(min_depth_samples);
  cache_key.scalar(conditional);
  cache_key.scalar(complete);
  cache_key.scalar(match_incomplete);
  cache_key.scalar(reached.approximate);
  cache_key.scalar(eligible.approximate);
  cache_key.scalar(relation.strata.size());
  for (std::size_t i = 0; i < relation.strata.size(); ++i) {
    const auto &stratum = relation.strata[i];
    cache_key.scalar(stratum.population);
    cache_key.scalar(selected ? selected->at(i) : stratum.selected);
  }
  cache_key.scalar(depths.size());
  for (auto depth : depths)
    cache_key.scalar(depth);
  cache_key.sources(passed);
  cache_key.sources(candidates);
  const auto cache_hash = std::hash<std::string>{}(cache_key.bytes());
  if (auto cached = sampled_rate_cache.find(cache_key, cache_hash))
    return resident_detail::publish_sampled_rates(*cached);

  const auto &index = field_index(key, name);
  // Group source rows by prefix length. ID-ordered suffix aggregates compute population, observed, passed and ESS quantities in O(rows log rows + depths log rows).
  struct Bucket {
    std::size_t observed = 0, passed = 0, unknown = 0;
    Sum denominator, sampled, squares, numerator;
  };
  struct Aggregate {
    std::size_t observed = 0, passed = 0, unknown = 0;
    Sum denominator, sampled, squares, numerator;
  };
  struct Suffix {
    std::size_t length = 0;
    Aggregate aggregate;
  };
  std::vector<std::map<std::size_t, Bucket>> buckets(relation.strata.size());
  for (const auto &entry : candidates) {
    const auto stratum = relation.row_strata.at(entry.first);
    const auto length = relation.row_lengths.at({entry.first, name});
    const bool success = passed.count(entry.first) != 0;
    const bool known = finite_number(entry.second) && entry.second != 0;
    const auto &quota = relation.strata.at(stratum);
    const auto admitted = selected ? selected->at(stratum) : quota.selected;
    const double weight =
        known ? quota.population / admitted / entry.second.get<double>() : 0.;
    // ID-ordered candidates give deterministic bucket accumulation.
    auto &bucket = buckets[stratum][length];
    ++bucket.observed;
    bucket.passed += success;
    if (!known) {
      ++bucket.unknown;
      continue;
    }
    bucket.denominator.add(weight);
    bucket.sampled.add(weight);
    bucket.squares.add(weight * weight);
    if (success)
      bucket.numerator.add(weight);
  }
  std::vector<std::vector<Suffix>> suffixes(relation.strata.size());
  for (std::size_t s = 0; s < buckets.size(); ++s) {
    auto &out = suffixes[s];
    out.reserve(buckets[s].size());
    for (const auto &[length, _] : buckets[s])
      out.push_back({length, {}});
    Aggregate accumulated;
    for (std::size_t i = out.size(); i-- > 0;) {
      const auto &bucket = buckets[s].at(out[i].length);
      accumulated.observed += bucket.observed;
      accumulated.passed += bucket.passed;
      accumulated.unknown += bucket.unknown;
      accumulated.denominator.add(bucket.denominator.get());
      accumulated.sampled.add(bucket.sampled.get());
      accumulated.squares.add(bucket.squares.get());
      accumulated.numerator.add(bucket.numerator.get());
      out[i].aggregate = accumulated;
    }
  }
  auto at_least = [](const std::vector<Suffix> &suffix,
                     std::size_t depth) -> Aggregate {
    const auto found =
        std::lower_bound(suffix.begin(), suffix.end(), depth,
                         [](const Suffix &entry, std::size_t value) {
                           return entry.length < value;
                         });
    return found == suffix.end() ? Aggregate{} : found->aggregate;
  };
  auto result = std::make_shared<resident_detail::SampledRates>();
  result->reserve(depths.size());
  for (auto depth : depths) {
    // No admitted row reaches this prefix depth, so the conditional survival
    // rate has no evidence. Publishing zero would claim the operator rejects
    // everything there; leave the depth unknown for field-level estimation.
    // Depth zero is retained as the empty-field anchor.
    Sum reachable;
    for (std::size_t s = 0; s < relation.strata.size(); ++s)
      reachable.add(index.populations[s].at_least(depth));
    if (depth && !reachable.get())
      continue;
    Sum population, denominator, numerator, squares, sampled;
    std::vector<resident_detail::SampledStratumRate> strata;
    strata.reserve(relation.strata.size());
    std::size_t observed = 0, positives = 0;
    bool known = complete;
    for (std::size_t s = 0; s < relation.strata.size(); ++s) {
      const auto p = index.populations[s].at_least(depth);
      population.add(p);
      const auto aggregate = at_least(suffixes[s], depth);
      known &= aggregate.unknown == 0;
      denominator.add(aggregate.denominator.get());
      sampled.add(aggregate.sampled.get());
      squares.add(aggregate.squares.get());
      numerator.add(aggregate.numerator.get());
      observed += aggregate.observed;
      positives += aggregate.passed;
      strata.push_back({p, aggregate.observed, aggregate.passed});
    }
    const auto pop = population.get(),
               den = conditional ? denominator.get() : pop;
    const auto eff =
        squares.get() ? sampled.get() * sampled.get() / squares.get() : 0.;
    known &= !pop || observed != 0;
    const std::optional<double> rate =
        !pop ? std::optional<double>(0.)
        : known && den
            ? std::optional<double>(std::clamp(numerator.get() / den, 0., 1.))
            : std::nullopt;
    result->push_back({depth, rate, pop, eff, observed, positives,
                       bool(rate) && !union_approximation &&
                           (!pop || eff >= min_depth_samples),
                       match_incomplete, std::move(strata)});
  }
  // Publish only fully computed immutable results. A failed computation leaves
  // no partial entry; a rollback simply reuses the exact former evidence key.
  sampled_rate_cache.remember(cache_key, cache_hash, result);
  return resident_detail::publish_sampled_rates(*result);
}
}
