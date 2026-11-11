#include "../detail.hpp"
#include "field_coverage.hpp"

namespace semcache::sampling {
Json NativeSampler::calls() const {
  auto result = impl->counters;
  result["semantic_operator_count"] =
      std::count_if(impl->ops.begin(), impl->ops.end(), [](const auto &entry) {
        return semantic_operator(entry.second);
      });
  result["sampled_semantic_operator_count"] =
      impl->directly_sampled_operators.size();
  result["state_workers_peak"] = impl->state_workers->peak();
  // The physical serial identifies one immutable input/evidence snapshot.
  // Prior-only refreshes are alternative estimates of that snapshot: choose
  // the newest one, never the strongest source encountered in hash-map order.
  std::map<std::pair<Id, std::uint64_t>, const CoreObservation *> snapshots;
  std::map<std::tuple<Id, std::uint64_t, std::string>,
           std::pair<std::uint64_t, Json>>
      generated;
  for (const auto &[_, data] : impl->states)
    for (const auto &[__, observation] : data.observations) {
      const auto &core = *observation->core;
      if (!core.node->operator_id ||
          !semantic_operator(impl->ops.at(*core.node->operator_id)))
        continue;
      auto &current = snapshots[{*core.node->operator_id, core.serial}];
      if (!current || current->estimate_serial < core.estimate_serial)
        current = &core;
    }
  resident_detail::FieldDepthCoverage coverage;
  std::map<std::pair<Id, std::uint64_t>, std::set<std::string>> registered;
  for (const auto &[identity, core] : snapshots) {
    const auto &info = *core->statistics;
    const auto &op = impl->ops.at(identity.first);
    auto fields = input_keys(op);
    if (fields.empty())
      for (const auto &field : op.value("required_columns", Json::array()))
        fields.insert(field.is_string() ? field.get<std::string>()
                                        : field_key(field));
    const auto sources = info.find("depth_selectivity_sources");
    const auto fusion = info.find("depth_selectivity_fusion");
    const auto measured = info.find("depth_selectivity_statistics");
    const auto output = info.find("estimated_output_survival");
    for (const auto &field : fields) {
      bool base = false;
      for (const auto &raw : core->node->relations) {
        const auto relation = node_relkey(raw);
        const auto summary = impl->relations.at(relation).summaries.find(field);
        if (summary == impl->relations.at(relation).summaries.end())
          continue;
        base = true;
        const auto key = relation + ":" + field;
        registered[identity].insert(field);
        // Only nonempty prefix sketches define reachable positive depths.
        // A fixed grid can contain empty sketches far beyond field length.
        for (auto d = summary->second->at("sketches").begin();
             d != summary->second->at("sketches").end(); ++d) {
          if (d.key() == "0" || d->value("entries", Json::array()).empty())
            continue;
          std::string source = "unknown";
          if (sources != info.end() && sources->contains(key) &&
              sources->at(key).contains(d.key()))
            source = sources->at(key).at(d.key()).get<std::string>();
          const bool usable = output != info.end() && output->contains(key) &&
                              output->at(key).contains(d.key()) &&
                              finite_number(output->at(key).at(d.key()));
          const auto *evidence = measured != info.end() &&
                                         measured->contains(key) &&
                                         measured->at(key).contains(d.key())
                                     ? &measured->at(key).at(d.key())
                                     : nullptr;
          const bool finite_zero =
              evidence && evidence->value("rate", Json(nullptr)) == 0 &&
              evidence->value("effective_samples", 0.) > 0 &&
              !info.value("population_census", false);
          const Json *weights = nullptr;
          if (fusion != info.end() && fusion->contains(key) &&
              fusion->at(key).contains(d.key()) &&
              fusion->at(key).at(d.key()).is_object()) {
            const auto &entry = fusion->at(key).at(d.key());
            const auto found = entry.find("weights");
            if (found != entry.end())
              weights = &*found;
          }
          coverage.add(identity.first, identity.second, key, d.key(), source,
                       usable, finite_zero, weights);
        }
      }
      if (!base)
        coverage.pending(identity.first, identity.second, field);
    }
  }
  // Conditional/generated summaries live on the layout-specific observation.
  // Merge their depth catalogue without counting a missing summary as a fake
  // positive depth. Base fields were already registered from their sketches.
  for (const auto &[_, data] : impl->states)
    for (const auto &[__, observation] : data.observations) {
      const auto &core = *observation->core;
      if (!core.node->operator_id ||
          !semantic_operator(impl->ops.at(*core.node->operator_id)))
        continue;
      const auto oid = *core.node->operator_id;
      const auto catalog =
          observation->statistics.read().find("field_depth_catalog");
      if (catalog == observation->statistics.read().end() ||
          !catalog->is_object())
        continue;
      const auto fields = input_keys(impl->ops.at(oid));
      for (auto entry = catalog->begin(); entry != catalog->end(); ++entry) {
        const auto &descriptor = entry.value();
        const auto field = descriptor.value("field", std::string());
        if (field.empty() || !fields.count(field) ||
            registered[{oid, core.serial}].count(field))
          continue;
        const auto depths = descriptor.find("depths");
        if (descriptor.value("pending", true) || depths == descriptor.end() ||
            !depths->is_array())
          continue;
        auto &latest = generated[{oid, core.serial, field}];
        if (latest.second.is_null() || latest.first < core.estimate_serial)
          latest = {core.estimate_serial, descriptor};
      }
    }
  for (const auto &[identity, latest] : generated) {
    const auto &[oid, serial, field] = identity;
    const auto &descriptor = latest.second;
    coverage.resolved(oid, serial, field);
    const auto sources = descriptor.value("depth_sources", Json::object());
    const auto output =
        descriptor.value("depth_output_survival", Json::object());
    const auto fusion = descriptor.value("depth_fusion", Json::object());
    for (const auto &depth : descriptor.at("depths"))
      if (depth.is_string() && depth != "0" && depth != "unknown") {
        const auto d = depth.get<std::string>();
        const Json *weights = nullptr;
        if (fusion.contains(d) && fusion.at(d).contains("weights"))
          weights = &fusion.at(d).at("weights");
        coverage.add(oid, serial, "generated:" + field, d,
                     sources.value(d, std::string("prior")),
                     output.contains(d) && finite_number(output.at(d)), false,
                     weights);
      }
  }
  result.update(coverage.json());
  result["cpu_workers_peak"] = impl->cpu_workers->peak_size();
  result["cpu_serial_batches"] = impl->cpu_workers->serial_batches();
  result["cpu_parallel_batches"] = impl->cpu_workers->parallel_batches();
  result["sampling_sql_workers_peak"] =
      impl->backend ? impl->backend->sample_workers_peak() : 1;
  const auto calibration_requests =
      impl->counters.value("calibration_model_requests", std::size_t(0));
  result["calibration_model_requests"] = calibration_requests;
  result["model_requests"] = impl->ledger.used() + calibration_requests;
  result["model_http_requests"] = impl->ledger.used() + calibration_requests;
  result["selection_requests_by_operator"] = Json::object();
  for (const auto &[oid, count] : impl->selection_requests_by_operator)
    result["selection_requests_by_operator"][std::to_string(oid)] = count;
  result["selection_model_requests"] =
      impl->ledger.selection_used();
  result["selection_budget_used"] = impl->ledger.selection_used();
  result["accuracy_model_requests"] = impl->ledger.accuracy_used();
  result["backend_seconds"] = impl->backend_seconds;
  result["text_metric_hits"] = impl->text_metrics.hits();
  result["cache_input_view_hits"] = impl->cache_input_view_hits;
  result["reuse_result_cache"] = impl->reuse_cache.diagnostics();
  result["text_metric_evaluations"] = impl->text_metrics.misses();
  result["row_weight_hits"] = impl->row_weights.hits();
  result["row_weight_evaluations"] = impl->row_weights.misses();
  result["input_fingerprint_hits"] = impl->populations->hits();
  result["input_fingerprint_evaluations"] = impl->populations->measurements();
  std::size_t join_entries = 0;
  for (const auto &entry : impl->join_selection_keys)
    join_entries += entry.second.size();
  result["join_selection_cache_entries"] = join_entries;
  if (impl->backend) {
    result["sql_cache"] = impl->backend->diagnostics();
    for (const auto &worker : impl->selection_workers) {
      if (!worker)
        continue;
      result["sampling_sql_workers_peak"] =
          std::max(result["sampling_sql_workers_peak"].get<std::size_t>(),
                   worker->sample_workers_peak());
      const auto extra = worker->diagnostics();
      for (auto it = extra.begin(); it != extra.end(); ++it)
        result["sql_cache"][it.key()] =
            result["sql_cache"].value(it.key(), std::size_t(0)) +
            it.value().get<std::size_t>();
    }
    for (const auto &slot : impl->state_slots) {
      if (!slot || slot->backend == impl->backend)
        continue;
      result["sampling_sql_workers_peak"] =
          std::max(result["sampling_sql_workers_peak"].get<std::size_t>(),
                   slot->backend->sample_workers_peak());
      const auto extra = slot->backend->diagnostics();
      for (auto it = extra.begin(); it != extra.end(); ++it)
        result["sql_cache"][it.key()] =
            result["sql_cache"].value(it.key(), std::size_t(0)) +
            it.value().get<std::size_t>();
    }
    for (std::size_t i = 1; i < impl->quality_workers.size(); ++i) {
      const auto extra = impl->quality_workers[i]->backend->diagnostics();
      for (auto it = extra.begin(); it != extra.end(); ++it)
        result["sql_cache"][it.key()] =
            result["sql_cache"].value(it.key(), std::size_t(0)) +
            it.value().get<std::size_t>();
    }
  }
  return result;
}
Json NativeSampler::timings() const {
  Json result = impl->backend ? impl->backend->timings() : Json::object();
  for (const auto &slot : impl->state_slots) {
    if (!slot || slot->backend == impl->backend)
      continue;
    const auto other = slot->backend->timings();
    for (auto it = other.begin(); it != other.end(); ++it) {
      auto &metric = result[it.key()];
      if (!metric.is_object())
        metric = Json::object();
      metric["seconds"] =
          metric.value("seconds", 0.) + it.value().at("seconds").get<double>();
      metric["calls"] = metric.value("calls", std::size_t(0)) +
                        it.value().at("calls").get<std::size_t>();
      if (it.value().contains("spans_ns")) {
        if (!metric.contains("spans_ns"))
          metric["spans_ns"] = Json::array();
        for (const auto &span : it.value().at("spans_ns"))
          metric["spans_ns"].push_back(span);
      }
    }
  }
  for (std::size_t i = 1; i < impl->quality_workers.size(); ++i) {
    const auto other = impl->quality_workers[i]->backend->timings();
    for (auto it = other.begin(); it != other.end(); ++it) {
      auto &metric = result[it.key()];
      if (!metric.is_object())
        metric = Json::object();
      metric["seconds"] =
          metric.value("seconds", 0.) + it.value().at("seconds").get<double>();
      metric["calls"] = metric.value("calls", std::size_t(0)) +
                        it.value().at("calls").get<std::size_t>();
      if (it.value().contains("spans_ns")) {
        if (!metric.contains("spans_ns"))
          metric["spans_ns"] = Json::array();
        for (const auto &span : it.value().at("spans_ns"))
          metric["spans_ns"].push_back(span);
      }
    }
  }
  for (auto &e : impl->phase_metrics) {
    result[e.first] = {{"seconds", e.second.seconds},
                       {"calls", e.second.calls}};
    if (impl->trace_timings)
      result[e.first]["spans_ns"] = e.second.spans;
  }
  return result;
}
}
