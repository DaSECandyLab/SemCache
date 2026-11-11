#include "detail.hpp"

namespace semcache::sampling {
namespace {
std::string quote_sample_id(const std::string &value) {
  std::string result = "'";
  for (const auto c : value) {
    result += c;
    if (c == '\'')
      result += c;
  }
  return result + "'";
}
}

void NativeSampler::resample_empty(const StatePtr &state) {
  // This runs before the one-shot accuracy comparison and before Beam. Rebuild
  // independent cohort evidence after a draw; retain paid request decisions,
  // response moments and history. Positive operators keep their original IDs.
  std::set<Id> exhausted;
  std::size_t wave = 0;
  while (!impl->deadline_reached()) {
    std::vector<std::pair<Id, std::set<std::string>>> empty;
    std::map<Id, std::size_t> observed_units;
    for (const auto &[_, observation] :
         impl->states.at(state->handle_id).observations) {
      const auto &core = *observation->core;
      if (!core.node->operator_id || !observation->complete)
        continue;
      const auto oid = *core.node->operator_id;
      observed_units[oid] =
          core.statistics->value("sample_input_rows", std::size_t(0));
      const auto &op = impl->ops.at(oid);
      if (exhausted.count(oid) || !semantic_operator(op) ||
          op.at("kind").get<std::string>().find("PROJECTION") !=
              std::string::npos ||
          row_count(*observation->rows))
        continue;
      std::set<std::string> sources;
      for (const auto &input : core.inputs)
        for (const auto &[relation, ids] : origins(*input))
          sources.insert(relation);
      if (!sources.empty())
        empty.emplace_back(oid, std::move(sources));
    }
    if (empty.empty())
      break;
    Json config = impl->imported;
    config["profile"] = impl->profile_;
    config["history"] = impl->history_;
    config["model_loads"] = Json::object();
    for (const auto &[model, loads] : impl->loads)
      config["model_loads"][model] = {loads.first, loads.second};
    std::size_t increased = 0;
    for (const auto &[oid, source_keys] : empty) {
      const auto name = std::to_string(oid);
      const auto limit = impl->initial_size;
      const auto size_key = std::to_string(limit);
      const auto old_target = impl->operator_sample_sizes.count(oid)
                                  ? impl->operator_sample_sizes.at(oid)
                                  : limit;
      const auto maximum = impl->positive("max_operator_sample_rows", 50);
      if (old_target >= maximum) {
        exhausted.insert(oid);
        impl->counters["zero_output_sampling"][name] = {
            {"status", observed_units.at(oid) >= maximum ? "sample_cap"
                                                         : "source_exhausted"},
            {"sampled_units", observed_units.at(oid)},
            {"estimated_selectivity",
             observed_units.at(oid) >= maximum ? 1. / maximum : 0.}};
        continue;
      }
      // Draw the remaining allowance once and rebuild statistics once.
      const auto draw_size = maximum - old_target;
      config["operator_sample_sizes"][name] = old_target;
      bool added = false;
      std::size_t population = 1;
      for (const auto &relation : source_keys) {
        auto &cohort =
            config["operator_direct_sources"][name][relation][size_key];
        if (cohort.is_null())
          cohort = config.at("direct_sources").at(relation).at(size_key);
        std::string excluded;
        for (const auto &id : cohort) {
          if (!excluded.empty())
            excluded += ',';
          excluded += quote_sample_id(id.get<std::string>());
        }
        const Json *descriptor = nullptr;
        for (const auto &source : config.at("query").at("relations"))
          if (relkey(source.at("node_id")) == relation) {
            descriptor = &source;
            break;
          }
        if (!descriptor)
          throw std::invalid_argument("Refill requires a bound base source");
        const auto seed =
            (std::stoull(
                 digest(Json::array({impl->settings.value("seed", Json(42)),
                                     oid, relation, old_target}))
                     .substr(0, 8),
                 nullptr, 16) %
             (std::uint64_t(1) << 31));
        const auto sql =
            "SELECT * FROM (SELECT * FROM (" +
            descriptor->at("source_sql").get<std::string>() +
            ") source WHERE CAST(__sample_id AS VARCHAR) NOT IN (" +
            (excluded.empty() ? "NULL" : excluded) +
            ")) USING SAMPLE reservoir(" + std::to_string(draw_size) +
            " ROWS) REPEATABLE (" + std::to_string(seed) +
            ") ORDER BY hash(__sample_id," + std::to_string(seed) + ")";
        const auto backend = impl->backend;
        auto rows = impl->state_backend_call(
            [&] { return backend->load(relation, sql); });
        added |= !rows.empty();
        auto &registry = config["sources"][relation];
        std::set<std::string> known;
        for (const auto &row : registry)
          known.insert(row.at("id").get<std::string>());
        for (auto &row : rows) {
          cohort.push_back(row.at("id"));
          if (known.insert(row.at("id").get<std::string>()).second)
            registry.push_back(std::move(row));
        }
        config["populations"][relation] =
            std::max(config.at("populations").at(relation).get<double>(),
                     double(registry.size()));
        if (cohort.size() &&
            population >
                std::numeric_limits<std::size_t>::max() / cohort.size())
          throw std::overflow_error("Refill pair population overflow");
        population *= cohort.size();
      }
      if (!added && old_target >= population) {
        exhausted.insert(oid);
        impl->counters["zero_output_sampling"][name] = {
            {"status", "source_exhausted"},
            {"sampled_units", std::min(old_target, population)}};
        continue;
      }
      if (draw_size > std::numeric_limits<std::size_t>::max() - old_target)
        throw std::overflow_error("Refill sample count overflow");
      config["operator_sample_sizes"][name] = old_target + draw_size;
      increased += draw_size;
    }
    if (!increased || impl->deadline_reached())
      break;
    auto replacement = std::make_unique<Impl>(config, impl->backend);
    replacement->ledger = std::move(impl->ledger);
    // Refill continues until output; accuracy retains its separate per-operator allowance.
    replacement->ledger.allow_additional_selection(
        increased * std::max(std::size_t(1), impl->ops.size()) * 3);
    replacement->counters = impl->counters;
    replacement->counters["zero_output_refill_waves"] = ++wave;
    replacement->selection_requests_by_operator =
        std::move(impl->selection_requests_by_operator);
    replacement->selection_model_cursor = std::move(impl->selection_model_cursor);
    replacement->selection_workers = std::move(impl->selection_workers);
    replacement->moments = std::move(impl->moments);
    replacement->archived_events = std::move(impl->archived_events);
    replacement->archived_observations = std::move(impl->archived_observations);
    replacement->backend_seconds = impl->backend_seconds;
    replacement->phase_metrics = std::move(impl->phase_metrics);
    replacement->search_deadline = impl->search_deadline;
    impl = std::move(replacement);
    observe(state);
    for (const auto &[_, observation] :
         impl->states.at(state->handle_id).observations) {
      if (!observation->core->node->operator_id)
        continue;
      const auto oid = *observation->core->node->operator_id;
      if (impl->operator_sample_sizes.count(oid) &&
          row_count(*observation->rows))
        impl->counters["zero_output_sampling"][std::to_string(oid)] = {
            {"status", "output_found"},
            {"sampled_units",
             observation->core->statistics->at("sample_input_rows")}};
    }
  }
  if (impl->deadline_reached())
    impl->counters["zero_output_refill_deadline_reached"] = true;
}
}
