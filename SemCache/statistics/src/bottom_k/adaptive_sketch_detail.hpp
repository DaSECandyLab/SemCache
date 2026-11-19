// Included privately by prefix_sketch.cpp to share the validated PrefixSummary
// builder. The source callback supplies bounded batches; no full-data copy or
// disk spool.
using AdaptiveFetch = sem_adaptive_fetch;
#include "field_workers.hpp"
using Counts = std::map<Hash, uint64_t>;
using Curve = std::map<size_t, double>;

struct AutoConfig {
  Json json;
  size_t sample, cells, maximum, candidates, anchors, budget, min_capacity,
      exact_threshold, cooccurrence_rows;
  std::vector<size_t> capacities, seeds;
  bool automatic_capacities;
  size_t parallelism = 1;
  double tail, relative, absolute;
  explicit AutoConfig(const Json &supplied) {
    json = {{"sample_size", 4096},    {"pilot_cells", 524288},
            {"max_depth", 4096},      {"max_candidates", 64},
            {"max_anchors", 24},      {"budget_bytes", 67108864},
            {"capacities", nullptr},  {"min_capacity", 256},
            {"exact_threshold", 256}, {"seeds", {17, 43, 97}},
            {"tail_relative", 0.01},  {"relative_error", 0.05},
            {"absolute_error", 0.5},  {"retained_fraction", 0.1},
            {"fixed_grid", true},     {"cooccurrence_sample_rows", 8}};
    if (!supplied.is_object())
      throw std::invalid_argument("Adaptive configuration must be an object");
    for (auto it = supplied.begin(); it != supplied.end(); ++it) {
      if (it.key() == "build_parallelism") {
        if (!it.value().is_number_integer() || it.value() < 1 ||
            it.value() > 64)
          throw std::invalid_argument("build_parallelism must be in [1,64]");
        parallelism = it.value().get<size_t>();
        continue; // Operational setting must not change statistical identity.
      }
      if (!json.contains(it.key()))
        throw std::invalid_argument("Unknown adaptive option: " + it.key());
      json[it.key()] = it.value();
    }
    auto integer = [&](const char *key) -> size_t {
      const auto &v = json.at(key);
      if (!v.is_number_integer() || v.get<int64_t>() <= 0)
        throw std::invalid_argument(std::string(key) +
                                    " must be a positive integer");
      return v.get<size_t>();
    };
    sample = integer("sample_size");
    cells = integer("pilot_cells");
    maximum = integer("max_depth");
    candidates = integer("max_candidates");
    anchors = integer("max_anchors");
    budget = integer("budget_bytes");
    min_capacity = integer("min_capacity");
    exact_threshold = integer("exact_threshold");
    const auto &row_limit = json.at("cooccurrence_sample_rows");
    if (!row_limit.is_number_integer() || row_limit.get<int64_t>() < 0 ||
        row_limit.get<int64_t>() > 1024)
      throw std::invalid_argument(
          "cooccurrence_sample_rows must be in [0,1024]");
    cooccurrence_rows = row_limit.get<size_t>();
    if (min_capacity < 2)
      throw std::invalid_argument("min_capacity must be at least 2");
    automatic_capacities = json["capacities"].is_null();
    if (candidates < 4 || anchors < 2)
      throw std::invalid_argument("Insufficient adaptive grid budget");
    for (const auto *key : {"capacities", "seeds"}) {
      if (std::string(key) == "capacities" && automatic_capacities)
        continue;
      const auto &a = json.at(key);
      if (!a.is_array() || a.empty())
        throw std::invalid_argument("Empty capacities/seeds");
      for (const auto &v : a) {
        if (!v.is_number_integer() || v.get<int64_t>() < 0 ||
            (std::string(key) == "capacities" && v.get<int64_t>() < 2))
          throw std::invalid_argument("Invalid capacities/seeds");
      }
    }
    capacities = automatic_capacities
                     ? std::vector<size_t>{2}
                     : json["capacities"].get<std::vector<size_t>>();
    seeds = json["seeds"].get<std::vector<size_t>>();
    if (!std::is_sorted(capacities.begin(), capacities.end()) ||
        std::adjacent_find(capacities.begin(), capacities.end()) !=
            capacities.end())
      throw std::invalid_argument("Capacities must strictly increase");
    auto real = [&](const char *key) {
      if (!json[key].is_number())
        throw std::invalid_argument("Invalid error tolerance");
      double v = json[key].get<double>();
      if (!std::isfinite(v) || v < 0)
        throw std::invalid_argument("Invalid error tolerance");
      return v;
    };
    tail = real("tail_relative");
    relative = real("relative_error");
    absolute = real("absolute_error");
    if (real("retained_fraction") > 1)
      throw std::invalid_argument("retained_fraction must be in [0,1]");
    if (!json["fixed_grid"].is_boolean())
      throw std::invalid_argument("fixed_grid must be Boolean");
  }
};

// Retain byte offsets only up to the configured character ceiling, but validate
// the entire value and count its true length (Unicode, including embedded NUL).
std::pair<size_t, std::vector<size_t>>
CharacterEnds(const char *data, size_t bytes, size_t maximum) {
  std::vector<size_t> ends;
  const auto characters =
      VisitCharacters(data, bytes, [&](size_t count, size_t offset) {
        if (count <= maximum)
          ends.push_back(offset);
      });
  return {characters, std::move(ends)};
}

double Presence(double alpha, uint64_t count) {
  return alpha == 1 ? 1 : -std::expm1(count * std::log1p(-alpha));
}

double Integral(const Curve &values) {
  if (values.empty())
    return 0;
  auto prev = values.begin();
  double result = prev->second;
  for (auto it = std::next(prev); it != values.end(); ++it) {
    double gap = it->first - prev->first;
    result += (gap - 1) * prev->second / 2 + (gap + 1) * it->second / 2;
    prev = it;
  }
  return result;
}

double Residual(const Curve &curve, const std::vector<size_t> &anchors) {
  Curve values;
  for (const auto &kv : curve) {
    auto next = std::upper_bound(anchors.begin(), anchors.end(), kv.first);
    size_t a = *std::prev(next);
    double estimate = curve.at(a);
    if (a != kv.first && next != anchors.end())
      estimate +=
          (curve.at(*next) - estimate) * double(kv.first - a) / (*next - a);
    values[kv.first] = std::abs(kv.second - estimate);
  }
  return Integral(values);
}

double HashFraction(const Hash &h) {
  double value = 0, scale = 1;
  for (auto byte : h) {
    scale /= 256;
    value += byte * scale;
  }
  return value;
}

std::string Hex(const Hash &h) {
  static const char *digits = "0123456789abcdef";
  std::string result;
  for (auto c : h) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}

struct PilotValue {
  size_t index, length;
};

struct AutoField {
  std::string name;
  size_t population, limit = 1;
  double tail = 0, mean = 0;
  std::vector<PilotValue> pilot;
  std::map<size_t, Counts> counts;
};

struct AutoCurve {
  std::vector<size_t> sources;
  double alpha;
  Curve values;
  double tolerance;
};

// Stable Floyd sampling uses O(sample_size) work even for a large relation.
std::vector<size_t> PilotIndices(const AutoField &field, size_t pilot_size) {
  Hash hash = Rank(field.name);
  uint64_t seed = 0;
  for (size_t i = 0; i < 8; ++i)
    seed = (seed << 8) | hash[i];
  std::mt19937_64 rng(seed);
  size_t count = std::min(pilot_size, field.population);
  std::set<size_t> selected;
  for (size_t j = field.population - count; j < field.population; ++j) {
    size_t candidate = std::uniform_int_distribution<size_t>(0, j)(rng);
    selected.insert(selected.count(candidate) ? j : candidate);
  }
  return {selected.begin(), selected.end()};
}

void SetPilotLengthStatistics(AutoField &field, const AutoConfig &cfg) {
  double total = 0;
  size_t longest = 1;
  for (const auto &value : field.pilot) {
    total += value.length;
    longest = std::max(longest, value.length);
  }
  auto discarded_tail = [&](size_t depth) {
    double characters = 0;
    for (const auto &value : field.pilot)
      if (value.length > depth)
        characters += value.length - depth;
    return characters;
  };
  size_t low = 1, high = std::min(longest, cfg.maximum);
  while (low < high) {
    size_t middle = low + (high - low) / 2;
    if (discarded_tail(middle) <= cfg.tail * total)
      high = middle;
    else
      low = middle + 1;
  }
  field.limit = low;
  field.tail = discarded_tail(low) / std::max<size_t>(field.pilot.size(), 1);
  field.mean = total / std::max<size_t>(field.pilot.size(), 1);
}


inline double AdaptiveFieldStorageBytes(
    const std::vector<AutoField> &fields,
    const std::vector<Curve> &distinct_by_depth, size_t f, size_t m,
    const std::vector<size_t> &grid) {
  if (!fields[f].population)
    return 0.0;
  double bytes = 0;
  for (auto k : grid)
    bytes += 128 + (192.0 + 4.0 * k) *
                       std::min(double(m), distinct_by_depth[f].at(k));
  return bytes;
}

inline double MinimumAdaptiveBytes(
    const std::vector<AutoField> &fields,
    const std::vector<Curve> &distinct_by_depth,
    const std::vector<size_t> &capacities, const std::vector<size_t> &grid) {
  double sum = 0;
  for (size_t f = 0; f < fields.size(); ++f)
    sum += AdaptiveFieldStorageBytes(fields, distinct_by_depth, f,
                                     capacities[f], grid);
  return sum;
}

struct PreparedAdaptiveGrid {
  std::vector<AutoField> fields;
  std::vector<size_t> depths;
  std::vector<size_t> anchors;
  std::set<std::pair<size_t, size_t>> pairs;
  std::vector<AutoCurve> curves;
  std::vector<Curve> distinct_by_depth;
  std::vector<double> distinct;
  std::vector<double> targets;
  std::vector<size_t> capacities;
  std::vector<size_t> ceilings;
  std::vector<double> errors;
};

template <class BatchSourceFactory>
PreparedAdaptiveGrid PrepareAdaptiveGrid(
    AutoConfig &cfg, const size_t *population, size_t field_count,
    const Json &names, bool table, FieldWorkers &workers,
    BatchSourceFactory &&batch_source) {
  if (!names.is_array() || names.size() != field_count || !field_count)
    throw std::invalid_argument("Expected nonempty field-name array");
  if (table)
    for (size_t f = 1; f < field_count; ++f)
      if (population[f] != population[0])
        throw std::invalid_argument("Table fields must have aligned rows");
  if (field_count > cfg.cells / cfg.candidates)
    throw std::invalid_argument("Too many fields for pilot_cells");
  // Automatic candidates grow with the actual input population, with no fixed
  // 1024 ceiling. Explicit candidates remain an opt-in resource restriction.
  if (cfg.automatic_capacities) {
    size_t largest = std::max(cfg.min_capacity, cfg.exact_threshold);
    for (size_t f = 0; f < field_count; ++f)
      largest = std::max(largest, population[f]);
    while (cfg.capacities.back() < largest) {
      size_t previous = cfg.capacities.back();
      if (previous > std::numeric_limits<size_t>::max() / 2) {
        cfg.capacities.push_back(largest);
        break;
      }
      cfg.capacities.push_back(previous * 2);
    }
  }
  size_t pilot_size =
      std::min(cfg.sample, cfg.cells / (field_count * cfg.candidates));
  std::vector<AutoField> fields(field_count);
  std::set<std::string> unique_names;
  for (size_t f = 0; f < field_count; ++f) {
    fields[f].name = names[f].get<std::string>();
    fields[f].population = population[f];
    if (!unique_names.insert(fields[f].name).second)
      throw std::invalid_argument("Duplicate field name");
  }
  workers.Run(field_count, [&](size_t f) {
    auto source = batch_source();
    auto &data = source.data;
    auto &sizes = source.sizes;
    auto &field = fields[f];
    std::vector<size_t> ids = PilotIndices(field, pilot_size);
    for (size_t start = 0; start < ids.size(); start += 1024) {
      std::vector<size_t> batch(ids.begin() + start,
                                ids.begin() +
                                    std::min(start + 1024, ids.size()));
      source.Read(f, batch);
      auto repeated = GroupBatch(data.data(), sizes.data(), data.size());
      std::vector<size_t> lengths(batch.size());
      for (const auto &group : repeated.groups)
        lengths[group.first] =
            CharacterEnds(data[group.first], sizes[group.first], 0).first;
      for (size_t i = 0; i < batch.size(); ++i)
        field.pilot.push_back({batch[i], lengths[repeated.representatives[i]]});
    }
    SetPilotLengthStatistics(field, cfg);
  });
  size_t maximum = 1;
  for (const auto &field : fields)
    maximum = std::max(maximum, field.limit);
  // Independent banks share dyadic coordinates through the same ceiling.
  // Capacity can differ without discarding the deeper common cache range.
  if (cfg.json["fixed_grid"].get<bool>() &&
      std::any_of(fields.begin(), fields.end(),
                  [](const AutoField &field) { return field.population != 0; }))
    maximum = cfg.maximum;
  std::set<size_t> base = {1, maximum}, extras;
  for (size_t k = 1; k <= maximum;) {
    base.insert(k);
    if (k > maximum / 2)
      break;
    k *= 2;
  }
  for (const auto &field : fields) {
    std::vector<size_t> lengths;
    for (const auto &v : field.pilot)
      lengths.push_back(v.length);
    std::sort(lengths.begin(), lengths.end());
    if (!lengths.empty())
      for (double q : {0.25, 0.5, 0.75, 0.9, 0.99}) {
        size_t n =
            lengths[std::min(lengths.size() - 1, size_t(q * lengths.size()))];
        for (size_t k : {n, n + 1, field.limit})
          if (k >= 1 && k <= maximum)
            extras.insert(k);
      }
  }
  for (auto it = std::next(base.begin()); it != base.end(); ++it)
    for (size_t q : {1, 2, 3})
      extras.insert(*std::prev(it) + (*it - *std::prev(it)) * q / 4);
  if (base.size() > cfg.candidates || base.size() > cfg.anchors)
    throw std::invalid_argument(
        "Adaptive grid budget cannot contain dyadic anchors");
  for (auto k : base)
    extras.erase(k);
  std::vector<size_t> extra(extras.begin(), extras.end()),
      depths(base.begin(), base.end()), anchors = depths;
  size_t room = cfg.candidates - base.size();
  for (size_t i = 0; i < std::min(room, extra.size()); ++i)
    depths.push_back(extra[extra.size() > room ? i * extra.size() / room : i]);
  std::sort(depths.begin(), depths.end());
  workers.Run(field_count, [&](size_t f) {
    auto source = batch_source();
    auto &data = source.data;
    auto &sizes = source.sizes;
    auto &field = fields[f];
    for (auto k : depths)
      field.counts[k];
    for (size_t start = 0; start < field.pilot.size(); start += 1024) {
      std::vector<size_t> ids;
      for (size_t i = start; i < std::min(start + 1024, field.pilot.size());
           ++i)
        ids.push_back(field.pilot[i].index);
      source.Read(f, ids);
      auto repeated = GroupBatch(data.data(), sizes.data(), data.size());
      std::vector<size_t> lengths(ids.size());
      for (const auto &group : repeated.groups) {
        const size_t i = group.first;
        auto decoded = CharacterEnds(data[i], sizes[i], maximum);
        lengths[i] = decoded.first;
        PrefixHash digest;
        for (auto k : depths) {
          if (k > decoded.second.size())
            break;
          size_t end = decoded.second[k - 1];
          Increment(field.counts[k][digest.At(data[i], end)], group.second);
        }
      }
      for (size_t i = 0; i < ids.size(); ++i)
        if (lengths[repeated.representatives[i]] !=
            field.pilot[start + i].length)
          throw std::invalid_argument("Adaptive source changed between passes");
    }
  });
  std::vector<AutoCurve> curves;
  for (size_t f = 0; f < field_count; ++f)
    for (double alpha : {0.1, 0.5, 1.0}) {
      AutoCurve c;
      c.sources = {f};
      c.alpha = alpha;
      for (auto k : depths) {
        double h = 0;
        for (const auto &p : fields[f].counts[k])
          h += std::max(0.0, alpha * p.second - Presence(alpha, p.second));
        c.values[k] = h / std::max(1e-30, alpha * fields[f].pilot.size());
      }
      c.tolerance = cfg.absolute + cfg.relative * Integral(c.values);
      curves.push_back(c);
    }
  std::set<std::pair<size_t, size_t>> pairs;
  for (size_t f = 0; f < field_count; ++f)
    if (field_count > 1)
      pairs.insert(std::minmax(f, (f + 1) % field_count));
  for (const auto &p : pairs) {
    AutoCurve c;
    c.sources = {p.first, p.second};
    c.alpha = 1;
    for (auto k : depths) {
      size_t hits = 0;
      for (const auto &entry : fields[p.first].counts[k])
        hits += fields[p.second].counts[k].count(entry.first);
      c.values[k] =
          double(hits) /
          std::max<size_t>(1, std::min(fields[p.first].pilot.size(),
                                       fields[p.second].pilot.size()));
    }
    c.tolerance = cfg.absolute + cfg.relative * Integral(c.values);
    curves.push_back(c);
  }
  // Estimate depth-specific diversity once; empty/deep anchors and repeated
  // prefixes do not consume m payload entries merely because rows are numerous.
  std::vector<Curve> distinct_by_depth(field_count);
  for (size_t f = 0; f < field_count; ++f)
    for (const auto &depth : fields[f].counts) {
      size_t observed = 0, singletons = 0;
      for (const auto &entry : depth.second) {
        observed += entry.second;
        singletons += entry.second == 1;
      }
      double population = fields[f].pilot.empty()
                              ? 0
                              : double(fields[f].population) * observed /
                                    fields[f].pilot.size();
      double unseen = observed ? double(singletons) / observed *
                                     std::max(0.0, population - observed)
                               : 0;
      distinct_by_depth[f][depth.first] = depth.second.size() + unseen;
    }
  std::vector<double> distinct(field_count, 0), targets(field_count, 0);
  std::vector<size_t> capacities(field_count), ceilings(field_count);
  for (size_t f = 0; f < field_count; ++f) {
    for (const auto &depth : distinct_by_depth[f])
      distinct[f] = std::max(distinct[f], depth.second);
    bool census = fields[f].pilot.size() == fields[f].population;
    // A census can certify exact small fields. Otherwise enforce the sampled
    // minimum, even if the pilot misses rare prefixes. Unused slots cost no
    // payload memory when a large relation contains few distinct strings.
    double required = census && distinct[f] <= cfg.exact_threshold
                          ? std::max(2.0, distinct[f])
                          : double(cfg.min_capacity);
    auto floor = std::lower_bound(cfg.capacities.begin(), cfg.capacities.end(),
                                  required);
    if (floor == cfg.capacities.end())
      throw std::invalid_argument(
          "capacities cannot satisfy min_capacity / exact_threshold");
    capacities[f] = *floor;
    auto ceiling = std::lower_bound(cfg.capacities.begin(),
                                    cfg.capacities.end(), distinct[f]);
    ceilings[f] =
        ceiling == cfg.capacities.end() ? cfg.capacities.back() : *ceiling;
    ceilings[f] = std::max(ceilings[f], capacities[f]);
    targets[f] =
        distinct[f] <= cfg.exact_threshold
            ? std::max(required, distinct[f])
            : std::max(required,
                       std::ceil(cfg.json["retained_fraction"].get<double>() *
                                 distinct[f]));
  }
  auto minimum = [&](const std::vector<size_t> &grid) {
    return MinimumAdaptiveBytes(fields, distinct_by_depth, capacities, grid);
  };
  if (minimum(anchors) > cfg.budget)
    throw std::invalid_argument(
        "Minimum adaptive sketches exceed budget_bytes");
  auto residuals = [&](const std::vector<size_t> &grid) {
    std::vector<double> r;
    for (const auto &c : curves)
      r.push_back(Residual(c.values, grid));
    return r;
  };
  auto errors = residuals(anchors);
  while (anchors.size() < cfg.anchors) {
    double best = 0;
    std::vector<size_t> best_grid;
    std::vector<double> best_errors;
    for (auto k : depths) {
      if (std::binary_search(anchors.begin(), anchors.end(), k))
        continue;
      auto grid = anchors;
      grid.insert(std::lower_bound(grid.begin(), grid.end(), k), k);
      if (minimum(grid) > cfg.budget)
        continue;
      auto next = residuals(grid);
      double gain = 0;
      for (size_t i = 0; i < curves.size(); ++i)
        gain += std::max(0.0, errors[i] - curves[i].tolerance) -
                std::max(0.0, next[i] - curves[i].tolerance);
      gain /= field_count * cfg.capacities[0] * (192.0 + 4.0 * k);
      if (gain > best) {
        best = gain;
        best_grid = std::move(grid);
        best_errors = std::move(next);
      }
    }
    if (best <= 0)
      break;
    anchors = std::move(best_grid);
    errors = std::move(best_errors);
  }
  return PreparedAdaptiveGrid{
      std::move(fields),     std::move(depths),
      std::move(anchors),    std::move(pairs),
      std::move(curves),     std::move(distinct_by_depth),
      std::move(distinct),   std::move(targets),
      std::move(capacities), std::move(ceilings),
      std::move(errors)};
}
