#include "detail.hpp"

namespace semcache::sampling {
NativeSampler::Impl::Impl(const Json &config,
                          std::shared_ptr<NativeSQLBackend> b)
    : imported(config), profile_(config.value("profile", Json::object())),
      history_(config.value("history", Json::array())),
      settings(config.value("sampling", Json::object())),
      sketch_config(settings.value("adaptive_sketch", Json::object())),
      backend(std::move(b)), ledger(0) {
  const auto parallelism = positive("parallelism", 1);
  const auto requested_workers =
      positive("cpu_workers", std::min<std::size_t>(parallelism, 64));
  if (requested_workers > 64)
    throw std::invalid_argument("CPU worker count must be between 1 and 64");
  const auto workers = std::min(parallelism, requested_workers);
  cpu_workers = std::make_unique<ParallelRanges>(workers);
  for (const auto *name : {"selectivity_batch_size", "accuracy_batch_size"})
    if (settings.contains(name) && settings.at(name) != 1)
      throw std::invalid_argument(
          "Sampling requires one record/pair per prompt");
  settings["selectivity_batch_size"] = 1;
  settings["accuracy_batch_size"] = 1;
  trace_timings = settings.value("trace_timings", false);
  sample_depth_selectivity = settings.value("sample_depth_selectivity", true);
  ensure_prefix_depth_rates = settings.value("ensure_prefix_depth_rates", true);
  selectivity_estimator = settings.value("selectivity_estimator", "mask");
  if (selectivity_estimator != "legacy" && selectivity_estimator != "mask" &&
      selectivity_estimator != "ancestor")
    throw std::invalid_argument("Unknown selectivity estimator");
  // Labels are independent operator evidence; conditional cross-state priors
  // cannot change their measured acceptance rates.
  share_state_statistics = false;
  auto configured_seed = settings.value("seed", Json(42));
  if (!configured_seed.is_number_integer() && !exact_integer(configured_seed))
    throw std::invalid_argument("Sampling seed must be an integer");
  budget = positive("budget", 128);
  const auto operator_rows = positive("operator_sample_rows", 20);
  if (positive("max_operator_sample_rows", 50) < operator_rows)
    throw std::invalid_argument("max_operator_sample_rows must be >= operator_sample_rows");
  relational_sample_rows = positive("relational_sample_rows", 200);
  base_sample_size = std::max(operator_rows, relational_sample_rows);
  // The resident source pool serves both kinds of operators. Its storage
  // budget is separate from the ledger's physical model-request budget.
  budget = std::max(budget, base_sample_size);
  join_pair_budget =
      std::min(positive("join_pair_budget", operator_rows), operator_rows);
  initial_size = std::min(operator_rows, budget);
  batch_size = positive("batch_size", 16);
  accuracy_size = positive("accuracy_size", 8);
  max_strata = positive("max_strata", 4);
  positive("depth_targets", 4);
  min_depth_samples = positive("min_depth_samples", 6);
  const auto depth_error = settings.value("depth_error", 0.15);
  if (!std::isfinite(depth_error) || depth_error <= 0 || depth_error >= 1)
    throw std::invalid_argument("Depth error must be finite and in (0,1)");
  // Per-depth precision is descriptive; correlated Join samples do not guarantee normal-approximation precision.
  input_samples = std::min(initial_size, min_depth_samples);
  Json operators = config.value("operators", Json::array());
  if (operators.is_object()) {
    for (auto it = operators.begin(); it != operators.end(); ++it)
      ops[it.value().at("operator_id").get<Id>()] = it.value();
  } else
    for (auto &op : operators)
      ops[op.at("operator_id").get<Id>()] = op;
  if (profile_.value("plan_generation", std::string()) ==
      "heuristic_requests") {
    const auto layouts = profile_.value("layouts", Json::object());
    for (auto &[oid, op] : ops) {
      const auto key = std::to_string(oid);
      if (!semantic_operator(op))
        continue;
      if (layouts.contains(key) && layouts.at(key).is_array() &&
          layouts.at(key).size() == 1)
        op["default_layout"] = layouts.at(key)[0];
      else
        op["default_layout"]["block_order"] = Json::array(
            {"OPERATOR_SPECIFICATION", "TASK_DESCRIPTION", "DATA_BLOCK"});
    }
  }
  // State workers are I/O owners, not another copy of the DuckDB CPU pool.
  // Keep at least one reusable I/O worker per distinct candidate model so
  // different services can wait concurrently even when the CPU setting is 1.
  // This count is not a same-model lane: Beam transport is unrestricted and
  // several workers may use the same physical model simultaneously.
  std::set<std::string> model_ids;
  for (const auto &[_, op] : ops)
    if (op.contains("models") && op["models"].is_array())
      for (const auto &model : op["models"])
        if (model.is_string())
          model_ids.insert(model.get<std::string>());
  const auto state_capacity = std::min<std::size_t>(
      64, std::max<std::size_t>(workers, model_ids.size()));
  state_workers = std::make_unique<OrderedTasks>(state_capacity);
  initialize_accuracy_budget(operators);
  ledger = RequestLedger(positive("max_model_calls", 512));
  for (auto &entry : history_) {
    if (entry.contains("summary"))
      history_summaries[entry.at("id").get<std::string>()] =
          std::make_shared<const Json>(entry.at("summary"));
    std::string model = entry.at("model_id");
    auto &load = loads[model];
    if (entry.contains("requests_before") &&
        !entry["requests_before"].is_null())
      load.first = std::max(
          load.first, entry["requests_before"].get<std::uint64_t>() +
                          entry.value("requests_since", std::uint64_t(0)));
    if (entry.contains("characters_before") &&
        !entry["characters_before"].is_null())
      load.second = std::max(
          load.second, entry["characters_before"].get<std::uint64_t>() +
                           entry.value("characters_since", std::uint64_t(0)));
  }
  rebuild_history_index();
  if (config.contains("model_loads"))
    for (auto it = config["model_loads"].begin();
         it != config["model_loads"].end(); ++it)
      loads[it.key()] = {it.value()[0].get<std::uint64_t>(),
                         it.value()[1].get<std::uint64_t>()};
  Json sources = config.value("sources", Json::object()),
       populations = config.value("populations", Json::object());
  direct_sampling = config.contains("direct_sources");
  if (direct_sampling)
    max_strata = 1;
  for (auto it = sources.begin(); it != sources.end(); ++it) {
    Relation rel;
    std::string key = it.key();
    try {
      rel.id = Json::parse(key);
    } catch (...) {
      rel.id = key;
    }
    const Json population = populations.value(key, Json(it.value().size()));
    if (!finite_number(population) ||
        population.get<double>() < it.value().size())
      throw std::invalid_argument(
          "Base population must be finite and no smaller than sample");
    rel.population = population.get<double>();
    if (imported.contains("offline_statistics") &&
        imported.at("offline_statistics").contains(key)) {
      const auto &offline = imported.at("offline_statistics").at(key);
      rel.offline_cooccurrence = &offline.at("cooccurrence");
      rel.offline_columns =
          offline.at("columns").get<std::map<std::string, std::string>>();
      for (auto f = offline.at("summaries").begin();
           f != offline.at("summaries").end(); ++f)
        rel.summaries[f.key()] = std::make_shared<const Json>(f.value());
    }
    prepare(rel, it.value(), settings.value("seed", Json(42)));
    for (auto &h : rel.strata)
      for (auto &row : h.rows)
        for (auto &field : rel.summaries)
          rel.row_lengths[{row.at("id").get<std::string>(), field.first}] =
              character_length(serialize_value(
                  row.at("values").value(field.first, Json(nullptr))));
    relations.emplace(key, std::move(rel));
    if (!direct_sampling && relations[key].admitted.size() < base_sample_size)
      admit(relations[key], base_sample_size - relations[key].admitted.size(),
            {});
  }
  if (direct_sampling) {
    for (auto rel = config.at("direct_sources").begin();
         rel != config.at("direct_sources").end(); ++rel) {
      const auto registry_size = relations.at(rel.key()).admitted.size();
      std::map<std::string, const Json *> registry;
      for (const auto &row : relations.at(rel.key()).admitted)
        registry.emplace(row.at("id").get<std::string>(), &row);
      for (auto cohort = rel->begin(); cohort != rel->end(); ++cohort) {
        Json rows = Json::array();
        for (const auto &id : cohort.value())
          rows.push_back(*registry.at(id.get<std::string>()));
        const auto size = rows.size();
        counters["direct_source_rows"][rel.key()][cohort.key()] = size;
        for (auto &row : rows) {
          sample_factor(row, double(registry_size) / size);
          Json probabilities = Json::object();
          for (const auto &origin : row.at("lineage"))
            probabilities[canonical(origin)] = double(size) / registry_size;
          row["_sample_origin_probabilities"] = std::move(probabilities);
        }
        direct_sources.emplace(
            std::make_pair(rel.key(), std::stoull(cohort.key())),
            std::make_shared<const Json>(std::move(rows)));
      }
    }
    counters["source_sampling"] = "direct_base_sql";
    counters["source_resampling"] = false;
    if (config.contains("operator_direct_sources"))
      for (auto op = config.at("operator_direct_sources").begin();
           op != config.at("operator_direct_sources").end(); ++op) {
        const auto oid = std::stoll(op.key());
        operator_sample_sizes[oid] =
            config.at("operator_sample_sizes").at(op.key());
        for (auto rel = op->begin(); rel != op->end(); ++rel) {
          const auto &registered = relations.at(rel.key()).admitted;
          std::map<std::string, const Json *> by_id;
          for (const auto &row : registered)
            by_id.emplace(row.at("id").get<std::string>(), &row);
          for (auto cohort = rel->begin(); cohort != rel->end(); ++cohort) {
            Json rows = Json::array();
            const auto size = cohort->size();
            for (const auto &id : cohort.value()) {
              auto row = *by_id.at(id.get<std::string>());
              sample_factor(row, double(registered.size()) / size);
              Json probabilities = Json::object();
              for (const auto &origin : row.at("lineage"))
                probabilities[canonical(origin)] = double(size) / registered.size();
              row["_sample_origin_probabilities"] = std::move(probabilities);
              rows.push_back(std::move(row));
            }
            operator_direct_sources.emplace(
                std::make_tuple(oid, rel.key(), std::stoull(cohort.key())),
                std::make_shared<const Json>(std::move(rows)));
          }
        }
      }
  }
  profile_["adaptive_sketch"] = sketch_config;
}

std::size_t NativeSampler::Impl::positive(const char *key,
                                          std::size_t fallback) {
  auto value = settings.value(key, Json(fallback));
  if (!value.is_number_integer() || value.get<std::int64_t>() < 1)
    throw std::invalid_argument(
        std::string("Sampling limit must be positive: ") + key);
  return value.get<std::size_t>();
}

void NativeSampler::Impl::prepare(Relation &rel, const Json &rows,
                                  const Json &seed) {
  if (!seed.is_number_integer() && !exact_integer(seed))
    throw std::invalid_argument("Sampling seed must be an integer");
  std::vector<Json> all;
  std::set<std::string> ids, names;
  std::map<std::string, std::map<std::string, std::size_t>> lengths;
  for (auto row : rows) {
    std::string id = row.at("id");
    if (id.empty() || !ids.insert(id).second)
      throw std::invalid_argument(
          "Base sample IDs must be nonempty and unique");
    row["lineage"] = Json::array({Json::array({rel.id, id})});
    for (auto f = row.at("values").begin(); f != row.at("values").end(); ++f) {
      names.insert(f.key());
      lengths[id][f.key()] = character_length(serialize_value(f.value()));
    }
    all.push_back(std::move(row));
  }
  std::sort(all.begin(), all.end(),
            [](auto &a, auto &b) { return a.at("id") < b.at("id"); });
  std::vector<std::vector<Json>> groups;
  if (!all.empty())
    groups.push_back(all);
  auto bits = [](std::size_t n) {
    int b = 0;
    for (; n; n >>= 1)
      ++b;
    return b;
  };
  while (groups.size() < std::min({max_strata, budget, all.size()})) {
    std::optional<std::tuple<double, std::size_t, std::string, int>> best;
    for (std::size_t i = 0; i < groups.size(); ++i)
      for (auto &name : names) {
        std::set<int> bins;
        std::size_t low = SIZE_MAX, high = 0;
        for (auto &r : groups[i]) {
          auto &values = lengths[r.at("id").template get<std::string>()];
          auto n = values.count(name) ? values[name] : 4;
          bins.insert(bits(n));
          low = std::min(low, n);
          high = std::max(high, n);
        }
        if (bins.size() > 1) {
          auto position = bins.begin();
          std::advance(position, (bins.size() - 1) / 2);
          auto choice = std::make_tuple(
              (high - low) * std::sqrt(groups[i].size()), i, name, *position);
          if (!best || choice > *best)
            best = choice;
        }
      }
    if (!best)
      break;
    auto index = std::get<1>(*best);
    auto name = std::get<2>(*best);
    int boundary = std::get<3>(*best);
    std::vector<Json> low, high;
    for (auto &r : groups[index]) {
      auto &values = lengths[r.at("id").template get<std::string>()];
      (bits(values.count(name) ? values[name] : 4) <= boundary ? low : high)
          .push_back(r);
    }
    groups[index] = low;
    groups.insert(groups.begin() + index + 1, high);
  }
  double scale = rows.empty() ? 1 : rel.population / rows.size();
  Json fields = Json::object();
  for (auto &name : names) {
    if (rel.summaries.count(name))
      continue;
    fields[name] = Json::array();
    for (auto &r : rows)
      fields[name].push_back(
          serialize_value(r["values"].value(name, Json(nullptr))));
  }
  if (!fields.empty()) {
    auto built = build_summaries(fields, sketch_config);
    for (auto f = built["summaries"].begin(); f != built["summaries"].end();
         ++f)
      rel.summaries[f.key()] =
          std::make_shared<const Json>(std::move(f.value()));
    profile_["base_sketch_plans"][relkey(rel.id)] = built["plan"];
  }
  for (std::size_t i = 0; i < groups.size(); ++i) {
    for (auto &name : names)
      for (auto &r : groups[i]) {
        auto &values = lengths[r.at("id").template get<std::string>()];
        rel.length_counts[{i, name}][values.count(name) ? values[name] : 4] +=
            scale;
      }
    Random rng(std::stoull(digest(Json::array({seed, rel.id, i})).substr(0, 16),
                           nullptr, 16));
    if (!direct_sampling)
      rng.shuffle(groups[i]);
    Stratum h;
    h.population = groups[i].size() * scale;
    const auto retained = direct_sampling ? groups[i].size()
                                         : std::min(budget, groups[i].size());
    for (std::size_t j = 0; j < retained; ++j) {
      h.rows.push_back(groups[i][j]);
      rel.row_strata[groups[i][j].at("id").template get<std::string>()] = i;
    }
    rel.strata.push_back(std::move(h));
  }
  if (direct_sampling && !rel.strata.empty()) {
    rel.admitted = rel.strata.front().rows;
    rel.strata.front().selected = rows.size();
    rel.version = ++cohort_serial;
  }
}

bool NativeSampler::Impl::admit(
    Relation &rel, std::size_t count,
    const std::map<std::size_t, double> &priorities) {
  bool changed = false;
  for (std::size_t n = 0; n < count && rel.admitted.size() < budget; ++n) {
    std::optional<std::size_t> chosen;
    std::pair<int, double> best;
    for (std::size_t i = 0; i < rel.strata.size(); ++i) {
      auto &h = rel.strata[i];
      if (h.selected >= h.rows.size())
        continue;
      std::pair<int, double> score =
          h.selected
              ? std::make_pair(
                    0, priorities.empty()
                           ? std::sqrt(h.population) / (h.selected + 1)
                           : (priorities.count(i) ? priorities.at(i) : 0) /
                                 (h.selected * (h.selected + 1)))
              : std::make_pair(1, h.population);
      if (!chosen || score > best) {
        chosen = i;
        best = score;
      }
    }
    if (!chosen)
      break;
    auto &h = rel.strata[*chosen];
    rel.admitted.push_back(h.rows[h.selected++]);
    changed = true;
  }
  if (changed)
    rel.version = ++cohort_serial;
  return changed;
}
}
