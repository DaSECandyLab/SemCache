#include "impl_internal.hpp"
#include "../joint_presence.hpp"
#include "semcache/prefix_reuse.hpp"
#include <cmath>
#include <unordered_map>

namespace semcache::sampling {

using namespace stats_detail;

struct ReuseCache::Impl {
  static std::size_t json_bytes(const Json &value) {
    std::size_t bytes = sizeof(Json) + 32;
    if (value.is_object()) {
      const auto &object = value.get_ref<const Json::object_t &>();
      bytes += object.capacity() * sizeof(Json::object_t::value_type);
      for (auto it = value.begin(); it != value.end(); ++it)
        bytes += it.key().capacity() + 1 + json_bytes(it.value());
    } else if (value.is_array()) {
      bytes += value.get_ref<const Json::array_t &>().capacity() * sizeof(Json);
      for (const auto &entry : value)
        bytes += json_bytes(entry);
    } else if (value.is_string())
      bytes += sizeof(std::string) +
               value.get_ref<const std::string &>().capacity() + 1;
    return bytes;
  }
  // Results belong to the immutable summary/pair bank, so eviction releases
  // both the decoded terms and final values. Key every changing argument by
  // value: equal population handles alone never imply equal survivals/costs.
  struct Results {
    struct Entry {
      Json key, value;
    };
    std::vector<Entry> entries;
    std::size_t next = 0;
    template <class Evaluate>
    Json get(Json key, Evaluate evaluate, std::size_t &hits,
             std::size_t &misses) {
      for (const auto &entry : entries)
        if (entry.key == key) {
          ++hits;
          return entry.value;
        }
      ++misses;
      auto value = evaluate(); // Never memoize exceptions or failed validation.
      // Eight records occupy at most 64 KiB of retained key storage. Large
      // diagnostic payloads are computed normally, not retained indefinitely.
      if (json_bytes(key) > 8192)
        return value;
      Entry entry{std::move(key), value};
      if (entries.size() < 8)
        entries.push_back(std::move(entry));
      else {
        entries[next] = std::move(entry);
        next = (next + 1) % entries.size();
      }
      return value;
    }
  };
  struct Summary {
    std::shared_ptr<const Json> owner;
    // Retain the full bottom-k set, including its threshold item. Self reuse
    // excludes that item, whereas pairwise union thresholds need it. One
    // immutable decoded vector serves both paths and every history source.
    std::map<std::string, std::shared_ptr<const DecodedSketch>> depths;
    std::size_t entries = 0;
    std::uint64_t touched = 0;
    std::size_t bytes = 0;
    Results self_results, same_field_results;
  };
  struct CrossValues {
    std::map<std::string, std::pair<double, std::vector<std::array<double, 3>>>>
        depths;
    Results results;
  };
  struct Cross {
    std::shared_ptr<const Json> source, target, common;
    std::shared_ptr<CrossValues> values;
    std::size_t entries = 0;
    std::uint64_t touched = 0;
    std::size_t bytes = 0;
  };
  std::map<const Json *, Summary> summaries;
  std::map<std::tuple<const Json *, const Json *, const Json *>, Cross> crosses;
  std::size_t summary_entries = 0, cross_entries = 0;
  std::size_t retained_bytes = 0, peak_bytes = 0;
  // Summary and cross entries share immutable JSON owners. Account each owner
  // once instead of charging the full sketch payload to every cross pair.
  struct Owner {
    std::size_t references = 0, bytes = 0;
  };
  std::map<const Json *, Owner> owners;
  using OwnerCosts = std::vector<std::pair<const Json *, std::size_t>>;
  std::uint64_t clock = 0;
  std::size_t result_hits = 0, result_misses = 0;
  std::size_t summary_decodes = 0, summary_evictions = 0, cross_evictions = 0;
  std::size_t sketch_decodes = 0, decoded_prefixes = 0;
  // A pair of 13-depth text sketches can exceed 64K entries. Keep a bounded
  // working set large enough for both Join sides instead of alternately
  // evicting/re-hashing the same immutable fields on every State.
  static constexpr std::size_t capacity = 512, max_entries = 262144;
  static constexpr std::size_t max_bytes = 64 * 1024 * 1024;
  static std::size_t count(const Json &value) {
    std::size_t result = 0;
    for (auto it = value.at("sketches").begin();
         it != value.at("sketches").end(); ++it)
      result += it.value().at("entries").size();
    return result;
  }
  std::size_t owner_size(const std::shared_ptr<const Json> &owner) const {
    const auto found = owners.find(owner.get());
    return found == owners.end() ? json_bytes(*owner) : found->second.bytes;
  }
  std::size_t additional_bytes(const OwnerCosts &costs) const {
    std::size_t bytes = 0;
    for (const auto &[owner, size] : costs)
      if (!owners.count(owner))
        bytes += size;
    return bytes;
  }
  OwnerCosts
  owner_costs(std::initializer_list<std::shared_ptr<const Json>> values) const {
    OwnerCosts result;
    for (const auto &value : values) {
      if (!value ||
          std::any_of(result.begin(), result.end(), [&](const auto &entry) {
            return entry.first == value.get();
          }))
        continue;
      result.emplace_back(value.get(), owner_size(value));
    }
    return result;
  }
  void retain_owner(const std::shared_ptr<const Json> &owner,
                    const OwnerCosts &costs) {
    if (!owner)
      return;
    const auto *key = owner.get();
    auto &entry = owners[key];
    if (entry.references++ == 0) {
      entry.bytes =
          std::find_if(costs.begin(), costs.end(), [&](const auto &cost) {
            return cost.first == key;
          })->second;
      retained_bytes += entry.bytes;
    }
  }
  void release_owner(const std::shared_ptr<const Json> &owner) {
    if (!owner)
      return;
    const auto *key = owner.get();
    auto found = owners.find(key);
    if (found == owners.end())
      return;
    if (--found->second.references != 0)
      return;
    retained_bytes -= found->second.bytes;
    owners.erase(found);
  }
  void evict() {
    auto oldest_summary = std::min_element(
        summaries.begin(), summaries.end(),
        [](auto &a, auto &b) { return a.second.touched < b.second.touched; });
    auto oldest_cross =
        std::min_element(crosses.begin(), crosses.end(), [](auto &a, auto &b) {
          return a.second.touched < b.second.touched;
        });
    if (oldest_summary != summaries.end() &&
        (oldest_cross == crosses.end() ||
         oldest_summary->second.touched < oldest_cross->second.touched)) {
      summary_entries -= oldest_summary->second.entries;
      retained_bytes -= oldest_summary->second.bytes;
      release_owner(oldest_summary->second.owner);
      summaries.erase(oldest_summary);
      ++summary_evictions;
    } else {
      cross_entries -= oldest_cross->second.entries;
      retained_bytes -= oldest_cross->second.bytes;
      release_owner(oldest_cross->second.source);
      release_owner(oldest_cross->second.target);
      release_owner(oldest_cross->second.common);
      crosses.erase(oldest_cross);
      ++cross_evictions;
    }
  }
  Summary *summary(const std::shared_ptr<const Json> &value) {
    auto it = summaries.find(value.get());
    if (it == summaries.end()) {
      validate_summary(*value);
      auto entries = count(*value);
      // Sorted fixed-size hash/count vectors share the existing bounded bank.
      // Keep the conservative allowance for vectors, depth metadata and results.
      const auto costs = owner_costs({value});
      const auto bytes = costs.front().second + entries * 96 + 128 * 1024;
      if (bytes > max_bytes || costs.front().second > max_bytes - bytes)
        return nullptr;
      while (!summaries.empty() && (summaries.size() >= capacity ||
                                    summary_entries + entries > max_entries)) {
        auto oldest = std::min_element(
            summaries.begin(), summaries.end(), [](auto &a, auto &b) {
              return a.second.touched < b.second.touched;
            });
        summary_entries -= oldest->second.entries;
        retained_bytes -= oldest->second.bytes;
        release_owner(oldest->second.owner);
        summaries.erase(oldest);
        ++summary_evictions;
      }
      while (!summaries.empty() || !crosses.empty()) {
        if (retained_bytes + bytes + additional_bytes(costs) <= max_bytes)
          break;
        evict();
      }
      ++summary_decodes;
      it = summaries
               .emplace(value.get(),
                        Summary{value, {}, entries, ++clock, bytes, {}, {}})
               .first;
      retain_owner(value, costs);
      summary_entries += entries;
      retained_bytes += bytes;
      peak_bytes = std::max(peak_bytes, retained_bytes);
    } else
      it->second.touched = ++clock;
    return &it->second;
  }
  std::shared_ptr<const DecodedSketch>
  decoded(const std::shared_ptr<const Json> &owner, const std::string &depth) {
    auto *bank = summary(owner);
    if (bank) {
      const auto found = bank->depths.find(depth);
      if (found != bank->depths.end())
        return found->second;
    }
    auto value = std::make_shared<const DecodedSketch>(
        decode_sketch(owner->at("sketches").at(depth)));
    ++sketch_decodes;
    decoded_prefixes += value->counts.size();
    if (bank)
      bank->depths.emplace(depth, value);
    return value;
  }
};
ReuseCache::ReuseCache() : impl(std::make_unique<Impl>()) {}
ReuseCache::~ReuseCache() = default;
Json ReuseCache::diagnostics() const {
  return {{"hits", impl->result_hits},
          {"misses", impl->result_misses},
          {"summary_decodes", impl->summary_decodes},
          {"sketch_decodes", impl->sketch_decodes},
          {"decoded_prefixes", impl->decoded_prefixes},
          {"summary_evictions", impl->summary_evictions},
          {"cross_evictions", impl->cross_evictions},
          {"summary_entries", impl->summary_entries},
          {"cross_entries", impl->cross_entries},
          {"entry_limit", Impl::max_entries},
          {"retained_bytes_estimate", impl->retained_bytes},
          {"peak_bytes_estimate", impl->peak_bytes},
          {"byte_limit", Impl::max_bytes}};
}
Json ReuseCache::self(const std::shared_ptr<const Json> &s, const Json &alpha,
                      const Json &requests) {
  if (Impl::count(*s) > Impl::max_entries)
    return self_reuse(*s, alpha, requests);
  auto *bank = impl->summary(s);
  if (!bank)
    return self_reuse(*s, alpha, requests);
  return bank->self_results.get(
      Json::array({alpha, requests}),
      [&]() -> Json {
        auto r = survival_rate(alpha);
        double count = s->at("count").get<double>();
        if (r.is_null())
          return count ? Json(nullptr) : Json(0.);
        const double distinct = r.get<double>() * count;
        double denominator =
            requests.is_null() ? distinct : requests.get<double>();
        if (!denominator)
          return 0.;
        if (!distinct)
          return nullptr;
        const double scale = denominator / distinct;
        std::map<std::size_t, double> values;
        for (auto it = s->at("sketches").begin(); it != s->at("sketches").end();
             ++it) {
          auto sampled = impl->decoded(s, it.key());
          auto a = survival_rate(alpha, it.key());
          if (a.is_null() && sampled->sample_size)
            return nullptr;
          double rate = a.is_null() ? 0 : a.get<double>();
          Sum total;
          for (std::size_t i = 0; i < sampled->sample_size; ++i) {
            const auto &entry = sampled->counts[i];
            total.add(std::max(
                0.,
                scale < 1.
                    ? semcache::statistics::repeated_prefix_hits(scale * rate,
                                                                 entry.second)
                    : semcache::statistics::repeated_prefix_hits(rate,
                                                                 entry.second) +
                          (scale - 1.) * rate * entry.second));
          }
          values[std::stoull(it.key())] = total.get() / sampled->tau;
        }
        return integral(std::move(values), summary_observed_max(*s)) /
               denominator;
      },
      impl->result_hits, impl->result_misses);
}
Json ReuseCache::cross(const std::shared_ptr<const Json> &source,
                       const std::shared_ptr<const Json> &target,
                       const Json &as, const Json &at,
                       const CrossDependence &dependence,
                       const Json &requests) {
  validate_summary(*source, false);
  validate_summary(*target);
  if (!compatible(*source, *target))
    return nullptr;
  auto r = survival_rate(at);
  double count = target->at("count").get<double>();
  if (r.is_null())
    return count ? Json(nullptr) : Json(0.);
  const double distinct = r.get<double>() * count;
  double denominator =
      requests.is_null() ? distinct : requests.get<double>();
  if (denominator <= 0.)
    return 0.;
  if (!requests.is_null() && distinct > 0. && denominator < distinct)
    denominator = distinct;
  if (dependence.same_field) {
    std::size_t entries;
    try {
      entries = Impl::count(*target);
    } catch (const Json::exception &) {
      // Preserve lazy validation: an earlier unknown depth can return NULL
      // before a later malformed sketch is inspected by the public formula.
      return cross_reuse(*source, *target, as, at, dependence, requests);
    }
    if (entries > Impl::max_entries)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    auto *bank = impl->summary(target);
    if (!bank)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    return bank->same_field_results.get(
        Json::array({at, requests}),
        [&]() -> Json {
          std::map<std::size_t, double> values;
          for (auto it = target->at("sketches").begin();
               it != target->at("sketches").end(); ++it) {
            auto sampled = impl->decoded(target, it.key());
            const auto rate = survival_rate(at, it.key());
            if (rate.is_null() && sampled->sample_size)
              return nullptr;
            Sum total;
            for (std::size_t i = 0; i < sampled->sample_size; ++i) {
              const auto &entry = sampled->counts[i];
              total.add(present(rate.is_null() ? 0. : rate.get<double>(),
                                entry.second));
            }
            values[std::stoull(it.key())] =
                std::min(denominator, total.get() / sampled->tau);
          }
          return integral(std::move(values), summary_observed_max(*target)) /
                 denominator;
        },
        impl->result_hits, impl->result_misses);
  }
  auto entries = Impl::count(*source) + Impl::count(*target);
  if (entries > Impl::max_entries)
    return cross_reuse(*source, *target, as, at, dependence, requests);
  auto key =
      std::make_tuple(source.get(), target.get(), dependence.common.get());
  auto found = impl->crosses.find(key);
  if (found == impl->crosses.end()) {
    // Only intersections are retained in Cross::depths. Charging both full
    // input sketches to every pair needlessly evicts the same working set.
    entries = 0;
    for (auto depth = source->at("sketches").begin();
         depth != source->at("sketches").end(); ++depth) {
      const auto matched = target->at("sketches").find(depth.key());
      if (matched != target->at("sketches").end())
        entries += std::min(depth->at("entries").size(),
                            matched->at("entries").size());
    }
    const auto costs = impl->owner_costs({source, target, dependence.common});
    const auto bytes = entries * 32 + 64 * 1024;
    std::size_t owner_total = 0;
    for (const auto &[_, size] : costs)
      owner_total += size;
    if (bytes > Impl::max_bytes || owner_total > Impl::max_bytes - bytes)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    while (!impl->crosses.empty() &&
           (impl->crosses.size() >= Impl::capacity ||
            impl->cross_entries + entries > Impl::max_entries)) {
      auto oldest = std::min_element(
          impl->crosses.begin(), impl->crosses.end(),
          [](auto &a, auto &b) { return a.second.touched < b.second.touched; });
      impl->cross_entries -= oldest->second.entries;
      impl->retained_bytes -= oldest->second.bytes;
      impl->release_owner(oldest->second.source);
      impl->release_owner(oldest->second.target);
      impl->release_owner(oldest->second.common);
      impl->crosses.erase(oldest);
      ++impl->cross_evictions;
    }
    while (!impl->summaries.empty() || !impl->crosses.empty()) {
      if (impl->retained_bytes + bytes + impl->additional_bytes(costs) <=
          Impl::max_bytes)
        break;
      impl->evict();
    }
    found = impl->crosses
                .emplace(key, Impl::Cross{source,
                                          target,
                                          dependence.common,
                                          std::make_shared<Impl::CrossValues>(),
                                          entries,
                                          ++impl->clock,
                                          bytes})
                .first;
    impl->retain_owner(source, costs);
    impl->retain_owner(target, costs);
    impl->retain_owner(dependence.common, costs);
    impl->cross_entries += entries;
    impl->retained_bytes += bytes;
    impl->peak_bytes = std::max(impl->peak_bytes, impl->retained_bytes);
  } else
    found->second.touched = ++impl->clock;
  // Decoding a previously unseen operand can evict the pair from the bounded
  // cache. Retain its numeric results until this evaluation has finished.
  auto bank = found->second.values;
  return bank->results.get(
      Json::array({as, at, requests}),
      [&]() -> Json {
        std::map<std::size_t, double> values;
        for (auto it = source->at("sketches").begin();
             it != source->at("sketches").end(); ++it) {
          if (!target->at("sketches").contains(it.key()))
            continue;
          auto term = bank->depths.find(it.key());
          if (term == bank->depths.end()) {
            const auto left = impl->decoded(source, it.key()),
                       right = impl->decoded(target, it.key());
            auto sampled = cross_sample_counts(
                left->counts, right->counts,
                cross_capacity(it.value(), target->at("sketches").at(it.key())));
            std::vector<std::array<double, 3>> counts;
            counts.reserve(sampled.counts.size());
            const auto overlap = dependence.common
                ? semcache::statistics::prefix_common_counts(*dependence.common, it.key())
                : std::map<Hash, double>{};
            for (auto &entry : sampled.counts)
              counts.push_back(
                  {static_cast<double>(entry.second[0]),
                   static_cast<double>(entry.second[1]),
                   overlap.count(entry.first) ? overlap.at(entry.first) : 0.});
            term = bank->depths
                       .emplace(it.key(),
                                std::make_pair(sampled.tau, std::move(counts)))
                       .first;
          }
          auto a = survival_rate(as, it.key()), b = survival_rate(at, it.key());
          if ((a.is_null() || b.is_null()) && !term->second.second.empty())
            return nullptr;
          Sum total;
          if (!a.is_null() && !b.is_null())
            for (auto &counts : term->second.second)
              total.add(dependence.common
                            ? joint_presence(a.get<double>(), b.get<double>(),
                                             counts[0], counts[1], counts[2])
                            : present(a.get<double>(), counts[0]) *
                                  present(b.get<double>(), counts[1]));
          values[std::stoull(it.key())] =
              std::min(denominator, total.get() / term->second.first);
        }
        return integral(std::move(values),
                        cross_observed_max(*source, *target)) /
               denominator;
      },
      impl->result_hits, impl->result_misses);
}
Json ReuseCache::join(const std::shared_ptr<const Json> &l,
                      const std::shared_ptr<const Json> &r, const Json &length,
                      const Json &al, const Json &ar, const Json &left_requests,
                      const Json &right_requests) {
  auto a = survival_rate(al), b = survival_rate(ar);
  double lc = l->at("count").get<double>(), rc = r->at("count").get<double>();
  if ((!a.is_null() && a == 0) || (!b.is_null() && b == 0) || !lc || !rc)
    return 0.;
  if (a.is_null() || b.is_null())
    return nullptr;
  double right = right_requests.is_null() ? rc * b.get<double>()
                                          : right_requests.get<double>();
  if (right <= 0.)
    return 0.;
  const double eff_right = std::max(1., right);
  auto x = self(l, al, left_requests), y = self(r, ar, right_requests);
  if (x.is_null() || y.is_null())
    return nullptr;
  double mean =
      (length.is_null() ? l->at("total_length") : length).get<double>() / lc;
  return x.get<double>() / eff_right + (eff_right - 1.) / eff_right * mean +
         y.get<double>();
}

} // namespace semcache::sampling
