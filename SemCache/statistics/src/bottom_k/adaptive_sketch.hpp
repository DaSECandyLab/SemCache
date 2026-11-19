// Included privately by prefix_sketch.cpp to share the validated PrefixSummary
// builder. The source callback supplies bounded batches; no full-data copy or
// disk spool.
#include "adaptive_sketch_detail.hpp"

Json BuildAdaptive(const Json &configuration, const Json &names,
                   const size_t *population, size_t field_count,
                   AdaptiveFetch fetch, void *context, bool table = false) {
  AutoConfig cfg(configuration);
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
  auto prepared = PrepareAdaptiveGrid(cfg, population, field_count, names, table,
                                      workers, batch_source);
  auto &fields = prepared.fields;
  auto &anchors = prepared.anchors;
  const auto &pairs = prepared.pairs;
  const auto &curves = prepared.curves;
  const auto &distinct_by_depth = prepared.distinct_by_depth;
  const auto &distinct = prepared.distinct;
  const auto &targets = prepared.targets;
  auto &capacities = prepared.capacities;
  const auto &ceilings = prepared.ceilings;
  const auto &errors = prepared.errors;
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
            complete = complete && target.size() <= m;
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
  double used = MinimumAdaptiveBytes(fields, distinct_by_depth, capacities, anchors);
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
              AdaptiveFieldStorageBytes(fields, distinct_by_depth, f, next[f],
                                        anchors) -
              AdaptiveFieldStorageBytes(fields, distinct_by_depth, f,
                                        capacities[f], anchors);
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
