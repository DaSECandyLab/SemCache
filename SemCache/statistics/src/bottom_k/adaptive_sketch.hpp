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

Json BuildAdaptive(const Json &configuration, const Json &names,
                   const size_t *population, size_t field_count,
                   AdaptiveFetch fetch, void *context, bool table = false) {
  AutoConfig cfg(configuration);
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
  const auto width =
      table ? std::min(
                  {cfg.parallelism, field_count,
                   size_t(std::max(1u, std::thread::hardware_concurrency()))})
            : 1;
  FieldWorkers workers(width);
  std::mutex fetch_mutex;
  auto batch_source = [&] {
    return FieldBatch{fetch, context, width > 1 ? &fetch_mutex : nullptr,
                      {},    {},      {}};
  };
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
  auto storage = [&](size_t f, size_t m, const std::vector<size_t> &grid) {
    if (!fields[f].population)
      return 0.0;
    double bytes = 0;
    for (auto k : grid)
      bytes += 128 + (192.0 + 4.0 * k) *
                         std::min(double(m), distinct_by_depth[f].at(k));
    return bytes;
  };
  auto minimum = [&](const std::vector<size_t> &grid) {
    double sum = 0;
    for (size_t f = 0; f < field_count; ++f)
      sum += storage(f, capacities[f], grid);
    return sum;
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
  using Order =
      std::vector<std::pair<Hash, Hash>>; // trial rank, original fingerprint
  std::map<std::tuple<size_t, size_t, size_t>, Order> orders;
  for (size_t f = 0; f < field_count; ++f)
    for (auto k : anchors)
      for (auto seed : cfg.seeds) {
        auto &order = orders[std::make_tuple(f, k, seed)];
        std::string salt = "adaptive-pilot-" + std::to_string(seed) + ":";
        for (const auto &p : fields[f].counts[k])
          order.push_back(
              {Rank(salt +
                    std::string(reinterpret_cast<const char *>(p.first.data()),
                                32)),
               p.first});
        std::sort(order.begin(), order.end());
      }
  std::map<std::vector<size_t>, std::map<size_t, double>> losses;
  for (auto m : cfg.capacities)
    for (const auto &c : curves) {
      double &worst = losses[c.sources][m];
      Curve selected;
      for (auto k : anchors)
        selected[k] = c.values.at(k);
      double exact = Integral(selected);
      for (auto seed : cfg.seeds) {
        Curve estimated;
        for (auto k : anchors) {
          const auto &source =
              orders.at(std::make_tuple(c.sources[0], k, seed));
          Order order(source.begin(),
                      source.begin() + std::min(m, source.size()));
          bool complete = source.size() <= m;
          if (c.sources.size() == 2) {
            const auto &target =
                orders.at(std::make_tuple(c.sources[1], k, seed));
            order.insert(order.end(), target.begin(),
                         target.begin() + std::min(m, target.size()));
            std::sort(order.begin(), order.end());
            order.erase(std::unique(order.begin(), order.end()), order.end());
            complete = complete && target.size() <= m && order.size() <= m;
          }
          size_t n = complete ? order.size() : m - 1;
          double tau = complete ? 1 : HashFraction(order.at(m - 1).first),
                 value = 0;
          for (size_t i = 0; i < n; ++i) {
            const auto &hash = order[i].second;
            const auto &a = fields[c.sources[0]].counts[k];
            if (c.sources.size() == 1) {
              double count = a.at(hash);
              value +=
                  std::max(0.0, c.alpha * count - Presence(c.alpha, count));
            } else
              value +=
                  a.count(hash) && fields[c.sources[1]].counts[k].count(hash);
          }
          double denominator =
              c.sources.size() == 1
                  ? std::max(1e-30, c.alpha * fields[c.sources[0]].pilot.size())
                  : std::max<size_t>(
                        1, std::min(fields[c.sources[0]].pilot.size(),
                                    fields[c.sources[1]].pilot.size()));
          estimated[k] = value / tau / denominator;
        }
        worst = std::max(
            worst, std::abs(Integral(estimated) - exact) /
                       std::max(1e-30, cfg.absolute + cfg.relative * exact));
      }
    }
  double used = minimum(anchors);
  auto objective = [&](const std::vector<size_t> &allocation) {
    double result = 0;
    for (const auto &l : losses) {
      size_t m = cfg.capacities.back();
      for (auto f : l.first)
        if (allocation[f] < distinct[f])
          m = std::min(m, allocation[f]);
      result += std::max(0.0, l.second.at(m) - 1);
    }
    for (size_t f = 0; f < field_count; ++f)
      result += std::max(0.0, targets[f] / allocation[f] - 1);
    return result;
  };
  while (true) {
    double best = 0, extra_best = 0, before = objective(capacities);
    std::vector<size_t> next_best;
    for (const auto &l : losses)
      for (auto m : cfg.capacities) {
        auto next = capacities;
        double extra_bytes = 0;
        for (auto f : l.first) {
          next[f] = std::min(ceilings[f], std::max(next[f], m));
          extra_bytes +=
              storage(f, next[f], anchors) - storage(f, capacities[f], anchors);
        }
        if (used + extra_bytes > cfg.budget)
          continue;
        double gain = (before - objective(next)) / std::max(extra_bytes, 1.0);
        if (gain > best) {
          best = gain;
          extra_best = extra_bytes;
          next_best = std::move(next);
        }
      }
    if (best <= 0)
      break;
    capacities = std::move(next_best);
    used += extra_best;
  }
  Json plan = {
      {"version", 5},
      {"backend", "native"},
      {"config", cfg.json},
      {"anchors", anchors},
      {"capacity_candidates", cfg.capacities},
      {"capacities", Json::object()},
      {"fields", Json::object()},
      {"estimated_bytes", used},
      {"budget_satisfied", used <= cfg.budget},
      {"cross_pairs", Json::array()},
      {"epoch",
       Hex(Rank(Json({anchors, cfg.json, "native-floyd-v1"}).dump()))}};
  for (const auto &p : pairs)
    plan["cross_pairs"].push_back(
        {fields[p.first].name, fields[p.second].name});
  Json summaries = Json::object(), samples = Json::object();
  std::set<size_t> sampled_rows;
  std::vector<Json> built_summaries(field_count),
      built_samples(field_count, Json::object()), built_metadata(field_count);
  std::vector<size_t> order(field_count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return fields[a].population * fields[a].mean >
           fields[b].population * fields[b].mean;
  });
  workers.Run(
      field_count,
      [&](size_t f) {
        auto source = batch_source();
        auto &data = source.data;
        auto &sizes = source.sizes;
        auto &field = fields[f];
        double normalized = 0, interpolation = 0;
        for (const auto &l : losses)
          if (std::find(l.first.begin(), l.first.end(), f) != l.first.end()) {
            size_t m = cfg.capacities.back();
            for (auto s : l.first)
              if (capacities[s] < distinct[s])
                m = std::min(m, capacities[s]);
            normalized = std::max(normalized, l.second.at(m));
          }
        for (size_t i = 0; i < curves.size(); ++i)
          if (std::find(curves[i].sources.begin(), curves[i].sources.end(),
                        f) != curves[i].sources.end())
            interpolation =
                std::max(interpolation,
                         errors[i] / std::max(curves[i].tolerance, 1e-30));
        PrefixSummary summary(capacities[f], anchors.back(), anchors.data(),
                              anchors.size(),
                              table ? cfg.cooccurrence_rows : 0);
        uint64_t total = 0, tail = 0;
        for (size_t start = 0; start < field.population;) {
          size_t n = std::min<size_t>(1024, field.population - start);
          std::vector<size_t> ids(n);
          std::iota(ids.begin(), ids.end(), start);
          source.Read(f, ids);
          auto repeated = GroupBatch(data.data(), sizes.data(), data.size());
          std::vector<RowSample> rows;
          if (table && cfg.cooccurrence_rows) {
            rows.resize(n);
            // All equal values share their group bottom-b across every depth.
            for (size_t i = 0; i < n; ++i)
              KeepRow(rows[repeated.representatives[i]],
                      {RowRank(start + i), start + i}, cfg.cooccurrence_rows);
          }
          for (const auto &group : repeated.groups) {
            const size_t i = group.first;
            size_t length = summary.Add(data[i], sizes[i], group.second,
                                        rows.empty() ? nullptr : &rows[i]);
            Increment(total, Multiply(length, group.second));
            if (length > anchors.back())
              Increment(tail, Multiply(length - anchors.back(), group.second));
          }
          start += n;
        }
        built_summaries[f] = summary.Snapshot();
        if (table && cfg.cooccurrence_rows)
          for (const auto &depth : summary.sketches)
            for (const auto &entry : depth.second.counts) {
              auto &ids =
                  built_samples[f][std::to_string(depth.first)]
                                  [semcache::statistics::encode_prefix_hash(entry.second.hash)];
              ids = Json::array();
              for (const auto &row : entry.second.rows) {
                ids.push_back(row.second);
              }
            }
        bool tail_ok = tail <= cfg.tail * total;
        built_metadata[f] = {
            {"pilot_rows", field.pilot.size()},
            {"population_rows", field.population},
            {"estimated_distinct_prefixes", distinct[f]},
            {"capacity_target", targets[f]},
            {"capacity_target_met", capacities[f] >= targets[f]},
            {"requested_max_depth", field.limit},
            {"pilot_tail_mean", field.tail},
            {"sampling_error_ratio", normalized},
            {"interpolation_error_ratio", interpolation},
            {"actual_tail_mean",
             double(tail) / std::max<size_t>(field.population, 1)},
            {"actual_tail_target_met", tail_ok},
            {"empirical_target_met", normalized <= 1 && interpolation <= 1 &&
                                         field.tail <= cfg.tail * field.mean &&
                                         tail_ok}};
        built_summaries[f]["metadata"]["adaptive"] = {
            {"version", 5}, {"config", cfg.json}, {"field", built_metadata[f]}};
      },
      std::move(order));
  for (size_t f = 0; f < field_count; ++f) {
    const auto &name = fields[f].name;
    summaries[name] = std::move(built_summaries[f]);
    if (!built_samples[f].empty())
      samples[name] = std::move(built_samples[f]);
    plan["fields"][name] = std::move(built_metadata[f]);
    plan["capacities"][name] = capacities[f];
  }
  Json result = {{"plan", plan}};
  if (table) {
    // An inverted prefix index replaces enumeration of every F² field pair.
    // A prefix retained by only one field cannot contribute to any cross-field
    // intersection, so release its row references before gathering row data.
    std::map<std::string, size_t> required_depths;
    {
      struct Owners {
        size_t count = 0;
        std::pair<uint64_t, std::string> largest;
      };
      std::map<std::string, std::unordered_map<std::string, Owners>> owners;
      for (auto field = samples.begin(); field != samples.end(); ++field)
        for (auto depth = field->begin(); depth != field->end(); ++depth)
          for (auto prefix = depth->begin(); prefix != depth->end(); ++prefix) {
            auto &owner = owners[depth.key()][prefix.key()];
            ++owner.count;
            owner.largest =
                std::max(owner.largest, std::make_pair(summaries.at(field.key())
                                                           .at("sketches")
                                                           .at(depth.key())
                                                           .at("entries")
                                                           .at(prefix.key())
                                                           .get<uint64_t>(),
                                                       field.key()));
          }
      for (auto field = samples.begin(); field != samples.end(); ++field)
        for (auto depth = field->begin(); depth != field->end(); ++depth)
          for (auto prefix = depth->begin(); prefix != depth->end();) {
            const auto &owner = owners.at(depth.key()).at(prefix.key());
            if (owner.count < 2) {
              prefix = depth->erase(prefix);
              continue;
            }
            required_depths[field.key()] = std::max<size_t>(
                required_depths[field.key()], std::stoull(depth.key()));
            // The largest marginal (ties by field name) is never the chosen
            // source for any pair in this group. Keep its target prefix, but
            // do not retain its unused row reservoir or gather its row IDs.
            if (field.key() == owner.largest.second) {
              prefix = depth->erase(prefix);
              continue;
            }
            for (const auto &row : prefix.value())
              sampled_rows.insert(row.get<size_t>());
            ++prefix;
          }
    }
    // Text is auxiliary only for a retained conditional row reservoir. All
    // other prefix entries persist just the construction digest and count.
    for (auto field = summaries.begin(); field != summaries.end(); ++field)
      for (auto depth = field->at("sketches").begin();
           depth != field->at("sketches").end(); ++depth) {
        if (!depth->contains("prefixes"))
          continue;
        auto &text = depth->at("prefixes");
        for (auto prefix = text.begin(); prefix != text.end();)
          if (!samples.contains(field.key()) ||
              !samples.at(field.key()).contains(depth.key()) ||
              !samples.at(field.key()).at(depth.key()).contains(prefix.key()))
            prefix = text.erase(prefix);
          else
            ++prefix;
        if (text.empty())
          depth->erase("prefixes");
      }
    // Gather the union once per field, not once per field pair or prefix. Rows
    // selected at several depths/fields share one record in the table bank.
    std::vector<size_t> ids(sampled_rows.begin(), sampled_rows.end());
    Json records = Json::array();
    for (size_t i = 0; i < ids.size(); ++i)
      records.push_back(Json::array());
    auto source = batch_source();
    auto &data = source.data;
    auto &sizes = source.sizes;
    for (size_t f = 0; f < field_count; ++f) {
      const auto limit = required_depths[fields[f].name];
      if (!limit) {
        for (auto &record : records)
          record.push_back("");
        continue;
      }
      for (size_t start = 0; start < ids.size(); start += 1024) {
        std::vector<size_t> batch(ids.begin() + start,
                                  ids.begin() +
                                      std::min(start + 1024, ids.size()));
        source.Read(f, batch);
        for (size_t i = 0; i < batch.size(); ++i) {
          // The immutable full scan already validated this entire value.
          // Clip only the necessary prefix instead of decoding a long tail
          // again while gathering the shared row bank.
          size_t end = 0, characters = 0;
          while (end < sizes[i]) {
            if ((static_cast<unsigned char>(data[i][end]) & 0xc0) != 0x80 &&
                characters++ == limit)
              break;
            ++end;
          }
          records[start + i].push_back(
              std::string(data[i] ? data[i] : "", end));
        }
      }
    }
    for (auto &field : samples)
      for (auto &depth : field)
        for (auto &prefix : depth)
          for (auto &id : prefix)
            id = std::lower_bound(ids.begin(), ids.end(), id.get<size_t>()) -
                 ids.begin();
    result["cooccurrence"] = {{"version", 1},
                              {"entry_encoding", semcache::statistics::prefix_entry_encoding},
                              {"sample_rows", cfg.cooccurrence_rows},
                              {"fields", names},
                              {"field_depths", required_depths},
                              {"row_count", population[0]},
                              {"rows", std::move(records)},
                              {"samples", std::move(samples)}};
  }
  result["summaries"] = std::move(summaries);
  return result;
}
