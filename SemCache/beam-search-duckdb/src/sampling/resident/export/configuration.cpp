#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::telemetry() const {
  return {{"events", impl->archived_events},
          {"observations", impl->archived_observations}};
}
Json NativeSampler::config() const {
  Json out = impl->imported;
  out["profile"] = impl->profile_;
  out["history"] = impl->history_;
  out["model_loads"] = Json::object();
  for (auto &e : impl->loads)
    out["model_loads"][e.first] = {e.second.first, e.second.second};
  out["admitted"] = Json::object();
  out["population_counts"] = Json::object();
  out["sources"] = Json::object();
  for (auto &e : impl->relations) {
    auto &rel = e.second;
    out["sources"][e.first] = rel.admitted;
    out["admitted"][e.first] = rel.admitted.size();
    out["population_counts"][e.first] = rel.population;
  }
  out["resident_states"] = impl->states.size();
  out["resident_core_observations"] = impl->cores.size();
  out["calls"] = calls();
  out["timings"] = timings();
  out["events"] = impl->archived_events;
  out["observations"] = impl->archived_observations;
  Json accuracy_operators = Json::array();
  std::size_t accuracy_batches = 0;
  for (const auto id : impl->accuracy_operator_ids) {
    const auto limit = impl->accuracy_batch_limit;
    const auto used = impl->accuracy_batches_used.at(id);
    accuracy_batches += used;
    accuracy_operators.push_back(
        {{"operator_id", id},
         {"limit", limit},
         {"used", used},
         {"remaining", limit - std::min(limit, used)},
         {"http_requests", impl->accuracy_request_used.at(id)},
         {"sampled_units", impl->accuracy_sampled_inputs.at(id).size()}});
  }
  out["accuracy_budget"] = {{"allocation", "operator_batches"},
                            {"limit_per_operator", impl->accuracy_batch_limit},
                            {"used", accuracy_batches},
                            {"http_requests", impl->ledger.accuracy_used()},
                            {"operators", std::move(accuracy_operators)}};
  out["history_records"] = impl->history_;
  out["base_sketch_plans"] =
      impl->profile_.value("base_sketch_plans", Json::object());
  return out;
}
}
