#include "detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::slim(const Json &statistics) const {
  PhaseTimer timer(phase_metrics, "copy_scalar_statistics", trace_timings,
                   true);
  Json result = Json::object();
  static const std::set<std::string> large{"summary",
                                           "right_summary",
                                           "conditional_sketch_plan",
                                           "adaptive",
                                           "input_field_summaries",
                                           "input_field_statistics",
                                           "output_field_statistics",
                                           "depth_survival_statistics",
                                           "depth_selectivity_statistics",
                                           "prefix_selectivity_statistics",
                                           "depth_selectivity_sources",
                                           "depth_selectivity_fusion"};
  for (auto it = statistics.begin(); it != statistics.end(); ++it)
    if (!large.count(it.key()))
      result[it.key()] = it.value();
  return result;
}

double NativeSampler::Impl::raw_weight(const Json &row,
                                       const Counts *counts) const {
  double value = sample_factor(row);
  const auto &lineage = row.at("lineage");
  std::set<std::string> seen;
  for (const auto &pair : lineage) {
    if (lineage.size() > 1 && !seen.insert(canonical(pair)).second)
      continue;
    auto key = relkey(pair[0]);
    auto &rel = relations.at(key);
    auto index = rel.row_strata.at(pair[1].get<std::string>());
    auto n = counts ? counts->at({key, index}) : rel.strata[index].selected;
    if (!n)
      throw std::invalid_argument("Missing admitted stratum quota");
    value *= rel.strata[index].population / n;
  }
  return value;
}

double NativeSampler::Impl::weight(const Json &row,
                                   const Counts *counts) const {
  if (counts)
    return raw_weight(row, counts);
  return row_weights.weight(
      row, [this](const Json &value) { return raw_weight(value, nullptr); });
}

PopulationMoments::Result
NativeSampler::Impl::weighted_moments(const Json &rows) const {
  return population_moments.get(rows, [&]() -> PopulationMoments::Result {
    if (product(rows)) {
      const auto &parts = rows.at("product");
      const auto left = weighted_moments(parts[0]),
                 right = weighted_moments(parts[1]);
      return {left.first * right.first, left.second * right.second};
    }
    Sum total, squares;
    for (const auto &row : rows) {
      const auto w = weight(row);
      total.add(w);
      squares.add(w * w);
    }
    const auto mass = total.get();
    return {mass, squares.get() ? mass * mass / squares.get() : 0.};
  });
}

double NativeSampler::Impl::mass(const Json &rows, const Counts *counts) const {
  PhaseTimer timer(phase_metrics, "row_mass", trace_timings, true);
  if (!counts)
    return weighted_moments(rows).first;
  if (product(rows))
    return mass(rows["product"][0], counts) * mass(rows["product"][1], counts);
  Sum total;
  for (auto &r : rows)
    total.add(weight(r, counts));
  return total.get();
}

double NativeSampler::Impl::effective(const Json &rows,
                                      const Counts *counts) const {
  PhaseTimer timer(phase_metrics, "row_effective", trace_timings, true);
  if (!counts)
    return weighted_moments(rows).second;
  if (product(rows))
    return effective(rows["product"][0], counts) *
           effective(rows["product"][1], counts);
  std::vector<double> w;
  for (auto &r : rows)
    w.push_back(weight(r, counts));
  return effective_samples(w);
}

Json NativeSampler::Impl::boundary(const Json &rows) const {
  PhaseTimer timer(phase_metrics, "boundary_statistics", trace_timings);
  if (!row_count(rows))
    return Json::array({nullptr, Json::object()});
  if (product(rows)) {
    auto a = boundary(rows["product"][0]), b = boundary(rows["product"][1]);
    Json fields = Json::object();
    double eff = effective(rows);
    for (int side = 0; side < 2; ++side) {
      auto &bank = side ? b : a;
      double repeats = row_count(rows["product"][1 - side]);
      for (auto it = bank[1].begin(); it != bank[1].end(); ++it) {
        Json f = it.value();
        if (!f["sample_distinct_fraction"].is_null())
          f["sample_distinct_fraction"] =
              f["sample_distinct_fraction"].get<double>() / repeats;
        f["sample_effective_samples"] = eff;
        fields[it.key()] = f;
      }
    }
    return Json::array({!a[0].is_null() && !b[0].is_null()
                            ? Json(a[0].get<double>() + b[0].get<double>())
                            : Json(nullptr),
                        fields});
  }
  struct Field {
    std::size_t present = 0, nonnull = 0, top = 0;
    Sum lengths, bytes;
    FSum nulls;
    std::array<FSum, 8> hist;
    std::unordered_map<std::size_t, std::size_t> counts;
  };
  std::map<std::string, Field> fields;
  ValueMetrics metrics(&text_metrics);
  std::vector<double> weights;
  FSum accurate;
  Sum normal, width;
  for (auto &row : rows) {
    double w = weight(row);
    if (!std::isfinite(w) || w <= 0)
      throw std::invalid_argument(
          "Field summaries require finite positive sampling weights");
    weights.push_back(w);
    accurate.add(w);
    normal.add(w);
    double bytes = 0;
    for (auto it = row.at("values").begin(); it != row.at("values").end();
         ++it) {
      const auto metric = metrics.measure(it.value());
      double length = metric.characters;
      auto &f = fields[it.key()];
      ++f.present;
      f.lengths.add(length * w);
      f.bytes.add(metric.bytes * w);
      f.hist[metric.bin].add(w);
      if (it.value().is_null())
        f.nulls.add(w);
      else {
        ++f.nonnull;
        f.top = std::max(f.top, ++f.counts[metric.identity]);
      }
      if (it.key().rfind("__semantic_input:", 0) != 0)
        bytes += metric.bytes;
    }
    width.add(w * bytes);
  }
  double total = accurate.get();
  Json out = Json::object();
  Sum squares;
  for (auto w : weights)
    squares.add(w * w);
  double eff = squares.get() ? total * total / squares.get() : 0;
  for (auto &entry : fields) {
    auto &f = entry.second;
    if (f.present != rows.size())
      continue;
    Json hist = Json::array();
    for (auto &bin : f.hist)
      hist.push_back(bin.get() / total);
    out[entry.first] = {
        {"null_fraction", f.nulls.get() / total},
        {"average_length", f.lengths.get() / total},
        {"average_bytes", f.bytes.get() / total},
        {"sample_distinct_fraction",
         f.nonnull ? Json(double(f.counts.size()) / f.nonnull) : Json(nullptr)},
        {"sample_top_frequency",
         f.nonnull ? Json(double(f.top) / f.nonnull) : Json(nullptr)},
        {"sample_effective_samples", eff},
        {"length_histogram", hist},
        {"basis", "weighted admitted sample; serialized characters"}};
  }
  return Json::array({width.get() / normal.get(), out});
}

const FieldDepthIndex &
NativeSampler::Impl::field_index(const std::string &key,
                                 const std::string &name) const {
  auto &rel = relations.at(key);
  auto &index = field_indexes[{key, name}];
  if (index.version == rel.version)
    return index;
  if (index.populations.empty()) {
    for (std::size_t i = 0; i < rel.strata.size(); ++i) {
      auto counts = rel.length_counts.find({i, name});
      index.populations.emplace_back(counts == rel.length_counts.end()
                                         ? std::map<std::size_t, double>{}
                                         : counts->second);
    }
  }
  std::vector<std::vector<std::pair<std::string, std::size_t>>> members(
      rel.strata.size());
  for (const auto &row : rel.admitted) {
    auto id = row.at("id").get<std::string>();
    members[rel.row_strata.at(id)].emplace_back(id,
                                                rel.row_lengths.at({id, name}));
  }
  index.cohorts.clear();
  for (auto &group : members)
    index.cohorts.emplace_back(std::move(group));
  index.version = rel.version;
  return index;
}

Json NativeSampler::Impl::field_rates(
    const std::string &key, const std::string &name,
    const std::set<std::string> &reached,
    const std::optional<std::set<std::string>> &eligible, bool complete,
    bool census_allowed, const std::vector<std::size_t> *selected) const {
  PhaseTimer timer(phase_metrics, "depth_rates", trace_timings);
  auto &rel = relations.at(key);
  const auto &summary = *rel.summaries.at(name);
  // IDs and field/relation names may contain separators. Preserve every
  // boundary so different source sets cannot reuse one another's rates.
  resident_detail::SampledRateKey cache_identity;
  cache_identity.text(key);
  cache_identity.text(name);
  cache_identity.scalar(rel.version);
  cache_identity.scalar(complete);
  cache_identity.scalar(census_allowed);
  cache_identity.scalar(selected != nullptr);
  if (selected) {
    cache_identity.scalar(selected->size());
    for (auto count : *selected)
      cache_identity.scalar(count);
  }
  cache_identity.scalar(reached.size());
  for (auto &id : reached)
    cache_identity.text(id);
  cache_identity.scalar(bool(eligible));
  if (eligible) {
    cache_identity.scalar(eligible->size());
    for (auto &id : *eligible)
      cache_identity.text(id);
  }
  const auto &cache_key = cache_identity.bytes();
  if (auto it = depth_cache.find(cache_key); it != depth_cache.end())
    return *it->second;
  std::set<std::size_t> depths{0};
  for (auto it = summary.at("sketches").begin();
       it != summary.at("sketches").end(); ++it)
    depths.insert(std::stoull(it.key()));
  const auto &index = field_index(key, name);
  std::vector<DepthCounts> observed_cohorts;
  observed_cohorts.reserve(index.cohorts.size());
  for (std::size_t i = 0; i < index.cohorts.size(); ++i) {
    if (!selected || selected->at(i) == rel.strata[i].selected) {
      observed_cohorts.push_back(index.cohorts[i].observe(reached, eligible));
      continue;
    }
    // An old immutable core may be refreshed after another branch admits
    // rows. Interpret its evidence with its original stratum quotas.
    std::vector<std::pair<std::size_t, bool>> members;
    members.reserve(selected->at(i));
    for (std::size_t j = 0; j < selected->at(i); ++j) {
      const auto id = rel.strata[i].rows[j].at("id").get<std::string>();
      if (!eligible || eligible->count(id))
        members.emplace_back(rel.row_lengths.at({id, name}),
                             reached.count(id) != 0);
    }
    observed_cohorts.emplace_back(std::move(members));
  }
  Json out = Json::object();
  for (auto depth : depths) {
    // Nothing in the sample reaches this prefix depth, so survival here is
    // unmeasured. Publishing zero would claim the operator rejects every row
    // there; an absent entry stays unknown for field-level estimation.
    // Depth zero remains the empty-field anchor.
    double reachable = 0;
    for (std::size_t i = 0; i < rel.strata.size(); ++i)
      reachable += index.populations[i].at_least(depth);
    if (depth && !reachable)
      continue;
    struct G {
      double population, observed, passed, mass;
    };
    std::vector<G> groups;
    Sum population, denominator;
    double observed = 0, passed = 0;
    bool census = census_allowed, represented = true;
    for (std::size_t i = 0; i < rel.strata.size(); ++i) {
      auto &h = rel.strata[i];
      const auto counts = observed_cohorts[i].at_least(depth);
      double p = index.populations[i].at_least(depth), n = counts.first,
             k = counts.second;
      const auto quota = selected ? selected->at(i) : h.selected;
      double weight = quota ? h.population / quota : 0,
             m = eligible ? n * weight : p;
      groups.push_back({p, n, k, m});
      population.add(p);
      denominator.add(m);
      observed += n;
      passed += k;
      represented &= !m || n;
      census &= quota == h.population;
    }
    double pop = population.get(), den = denominator.get();
    bool known = !pop || (complete && observed && represented);
    Sum numerator, sampled, squares;
    Json strata = Json::array();
    for (auto &g : groups) {
      if (g.observed) {
        numerator.add(g.mass * g.passed / g.observed);
        sampled.add(g.mass);
        squares.add(g.mass * g.mass / g.observed);
      }
      strata.push_back({{"population", g.population},
                        {"observed", std::size_t(g.observed)},
                        {"passed", std::size_t(g.passed)}});
    }
    double eff =
        squares.get() ? sampled.get() * sampled.get() / squares.get() : 0;
    Json rate = known && den ? Json(std::clamp(numerator.get() / den, 0., 1.))
                : known && !pop ? Json(0.)
                                : Json(nullptr);
    out[std::to_string(depth)] = {
        {"rate", rate},
        {"observed", std::size_t(observed)},
        {"passed", std::size_t(passed)},
        {"effective_samples", eff},
        {"population", pop},
        {"reliable", known && (!pop ||
                               (complete && census_allowed &&
                                (observed == pop || census)) ||
                               eff >= min_depth_samples)},
        {"strata", strata},
        {"precision_basis", "stratified effective samples or full census; "
                            "no per-node interval"}};
  }
  if (depth_cache.size() >= 512)
    depth_cache.erase(depth_cache.begin());
  depth_cache[cache_key] = std::make_shared<const Json>(out);
  return out;
}

Json NativeSampler::Impl::depths(const NodePtr &node, const SampleInputs &rows,
                                 const std::optional<SampleInputs> &inputs,
                                 bool complete, bool census_allowed,
                                 bool match_incomplete) const {
  Json result = Json::object();
  if (!census_allowed || match_incomplete) {
    auto gather = [&](const SampleInputs &samples) {
      PhaseTimer timer(phase_metrics, "source_inclusions", trace_timings);
      return resident_detail::SourceEvidence(samples);
    };
    const auto reached = gather(rows);
    std::optional<resident_detail::SourceEvidence> eligible;
    if (inputs)
      eligible.emplace(gather(*inputs));
    return source_depths(node, reached, eligible ? &*eligible : nullptr,
                         complete, census_allowed, match_incomplete);
  } else {
    Origins reached, eligible;
    for (const auto &group : rows)
      for (const auto &entry : origins(group))
        reached[entry.first].insert(entry.second.begin(), entry.second.end());
    if (inputs)
      for (const auto &group : *inputs)
        for (const auto &entry : origins(group))
          eligible[entry.first].insert(entry.second.begin(),
                                       entry.second.end());
    for (const auto &raw : node->relations) {
      const auto key = node_relkey(raw);
      const auto &reached_ids = reached[key];
      const auto eligible_ids =
          inputs ? std::optional<std::set<std::string>>(eligible[key])
                 : std::nullopt;
      for (const auto &field : relations.at(key).summaries)
        result[key + ":" + field.first] =
            field_rates(key, field.first, reached_ids, eligible_ids, complete,
                        census_allowed);
    }
  }
  return result;
}

Json NativeSampler::Impl::source_depths(
    const NodePtr &node, const resident_detail::SourceEvidence &reached,
    const resident_detail::SourceEvidence *eligible, bool complete,
    bool census_allowed, bool match_incomplete) const {
  Json result = Json::object();
  for (const auto &raw : node->relations) {
    const auto key = node_relkey(raw);
    auto identities = [&](const resident_detail::SourceEvidence &evidence) {
      std::set<std::string> ids;
      auto bank = evidence.sources.find(key);
      if (bank != evidence.sources.end())
        for (const auto &entry : bank->second)
          ids.insert(entry.first);
      return ids;
    };
    std::set<std::string> reached_ids;
    std::optional<std::set<std::string>> eligible_ids;
    if (census_allowed && !match_incomplete) {
      reached_ids = identities(reached);
      if (eligible)
        eligible_ids = identities(*eligible);
    }
    for (const auto &field : relations.at(key).summaries) {
      if (!census_allowed || match_incomplete)
        result[key + ":" + field.first] = sampled_field_rates(
            key, field.first, reached, eligible ? *eligible : reached,
            bool(eligible), complete, match_incomplete);
      else {
        result[key + ":" + field.first] =
            field_rates(key, field.first, reached_ids, eligible_ids, complete,
                        census_allowed);
      }
    }
  }
  return result;
}
}
