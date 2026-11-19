#include "environment_internal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace semcache::sampling {

using namespace profile_detail;

ProfileEnvironment::ProfileEnvironment(const Json &config)
    : impl(std::make_unique<Impl>(config)) {}
ProfileEnvironment::~ProfileEnvironment() = default;
void ProfileEnvironment::observe(const StatePtr &state) {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  auto started = std::chrono::steady_clock::now();
  impl->observe(state);
  auto &metric = impl->metrics["profile_observe"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  ++metric.second;
}
void ProfileEnvironment::observe_batch(const std::vector<StatePtr> &states,
                                       std::size_t workers) {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  auto serial = [&] {
    for (const auto &state : states)
      observe(state);
  };
  if (workers <= 1 || impl->runtime_observed) {
    serial();
    return;
  }
  const auto started = std::chrono::steady_clock::now();
  std::vector<StatePtr> pending;
  std::vector<const StateData *> parents;
  std::unordered_set<std::uint64_t> visited;
  for (const auto &state : states) {
    if (impl->states.count(state->handle_id) ||
        !visited.insert(state->handle_id).second)
      continue;
    if (!state->parent || !impl->states.count(state->parent->handle_id)) {
      serial();
      return;
    }
    pending.push_back(state);
    parents.push_back(&impl->states.at(state->parent->handle_id));
  }
  workers = std::min({workers, pending.size(), std::size_t(32)});
  if (workers < 2 || pending.size() < workers * 16) {
    serial();
    return;
  }
  while (impl->observation_workers.size() < workers)
    impl->observation_workers.push_back(std::make_unique<Impl>(*impl));
  std::vector<std::optional<StateData>> results(pending.size());
  std::vector<std::exception_ptr> errors(pending.size());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  try {
    for (std::size_t worker = 0; worker < workers; ++worker)
      threads.emplace_back([&, worker] {
        auto &environment = *impl->observation_workers[worker];
        for (;;) {
          // Nearby candidates usually share a parent. Keep a small contiguous
          // range on one worker to reuse its input caches and avoid bouncing
          // shared evidence owners between cores for every sibling.
          const auto begin = next.fetch_add(64, std::memory_order_relaxed);
          if (begin >= pending.size())
            break;
          const auto end = std::min(begin + 64, pending.size());
          for (auto index = begin; index < end; ++index)
            try {
              results[index] =
                  environment.build_state(pending[index], parents[index]);
            } catch (...) {
              errors[index] = std::current_exception();
            }
        }
      });
  } catch (...) {
    for (auto &thread : threads)
      thread.join();
    throw;
  }
  for (auto &thread : threads)
    thread.join();
  for (std::size_t index = 0; index < pending.size(); ++index) {
    if (errors[index])
      std::rethrow_exception(errors[index]);
    impl->states.emplace(pending[index]->handle_id, std::move(*results[index]));
  }
  auto &metric = impl->metrics["profile_observe"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  metric.second += states.size();
}
void ProfileEnvironment::refresh(const std::vector<StatePtr> &states) {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  auto started = std::chrono::steady_clock::now();
  for (const auto &state : states) {
    impl->observe(state);
    // Prepared operator facts and ancestors are immutable during search; only runtime feedback refreshes them.
    if (!impl->runtime_observed)
      continue;
    auto &data = impl->states.at(state->handle_id);
    for (const auto &choice : data.actions) {
      const auto &chosen = *choice;
      if (!semantic(impl->ops.at(chosen.operator_id)))
        continue;
      data.node_records[impl->node_index->slots.at(chosen.node_id)] =
          std::make_shared<const Record>(impl->estimate(
              data, data, impl->indexed_node(data, chosen.node_id), chosen));
    }
  }
  auto &metric = impl->metrics["profile_refresh"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  ++metric.second;
}
void ProfileEnvironment::accuracy(const StatePtr &) {}
void ProfileEnvironment::retain(const std::vector<StatePtr> &live) {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::unordered_set<std::uint64_t> keep;
  std::unordered_set<std::uint64_t> keep_nodes;
  keep.reserve(live.size() * 4 + 16);
  keep_nodes.reserve(live.size() * 4 + 16);
  for (const auto &state : live)
    for (auto current = state;
         current && keep.insert(current->handle_id).second;
         current = current->parent) {
      if (current->added)
        keep_nodes.insert(current->added->handle_id);
      if (!current->parent)
        for (const auto &node : current->roots)
          keep_nodes.insert(node->handle_id);
    }
  // Release independent discarded state records on the same bounded planner
  // workers. Extracting nodes first leaves the live bank stable and lets every
  // shared evidence owner outlive all references still used by another state.
  using Discarded = decltype(impl->states)::node_type;
  std::vector<Discarded> discarded;
  discarded.reserve(impl->states.size());
  for (auto state = impl->states.begin(); state != impl->states.end();)
    if (!keep.count(state->first)) {
      auto removed = state++;
      discarded.push_back(impl->states.extract(removed));
    } else
      ++state;
  auto prune_memo = [&](Impl &environment) {
    for (auto memo = environment.survival_memo.begin();
         memo != environment.survival_memo.end();)
      if (std::any_of(memo->second.inputs.begin(), memo->second.inputs.end(),
                      [&](auto id) { return !keep_nodes.count(id); }))
        memo = environment.survival_memo.erase(memo);
      else
        ++memo;
  };
  const auto worker_count = impl->observation_workers.size();
  const auto dispose_workers = std::min(worker_count, discarded.size() / 256);
  std::size_t total_memo = impl->survival_memo.size();
  for (const auto &worker : impl->observation_workers)
    total_memo += worker->survival_memo.size();
  if (dispose_workers < 2 && (worker_count < 2 || total_memo < 256)) {
    prune_memo(*impl);
    for (auto &worker : impl->observation_workers)
      prune_memo(*worker);
    return;
  }
  const auto threads_to_run = std::max(dispose_workers, worker_count >= 2 ? worker_count : std::size_t(0));
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> next_memo{0};
  auto run_work = [&](bool include_main) {
    if (include_main)
      prune_memo(*impl);
    for (;;) {
      const auto m = next_memo.fetch_add(1, std::memory_order_relaxed);
      if (m >= worker_count)
        break;
      prune_memo(*impl->observation_workers[m]);
    }
    for (;;) {
      const auto index = next.fetch_add(64, std::memory_order_relaxed);
      if (index >= discarded.size())
        break;
      const auto limit = std::min(index + 64, discarded.size());
      for (std::size_t i = index; i < limit; ++i)
        discarded[i] = Discarded{};
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(threads_to_run > 1 ? threads_to_run - 1 : 0);
  try {
    for (std::size_t i = 1; i < threads_to_run; ++i)
      threads.emplace_back(run_work, false);
  } catch (...) {
  }
  run_work(true);
  for (auto &thread : threads)
    thread.join();
}
Json ProfileEnvironment::statistics(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->statistics(state, true);
}
Json ProfileEnvironment::full_statistics(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->statistics(state, false);
}
std::map<Id, std::shared_ptr<const Json>>
ProfileEnvironment::statistic_blocks(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::map<Id, Value> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->scalar();
  return result;
}
std::map<Id, std::shared_ptr<const Json>>
ProfileEnvironment::feature_blocks(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::map<Id, Value> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->features();
  return result;
}
const Json *StatisticView::find(const std::string &key) const {
  if (!fields)
    return nullptr;
  const auto &values = *static_cast<const FieldMap *>(fields);
  const auto found = values.find(key);
  return found == values.end() ? nullptr : found->second.get();
}
const Json &StatisticView::get(const std::string &key) const {
  static const Json missing = nullptr;
  const auto value = find(key);
  return value ? *value : missing;
}
std::map<Id, StatisticView>
ProfileEnvironment::feature_views(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::map<Id, StatisticView> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result.emplace(impl->node_index->ids[slot],
                     StatisticView{record.get(), &record->fields});
  return result;
}
StatisticView ProfileEnvironment::feature_view(const StatePtr &state,
                                               Id node_id) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  const auto &record =
      impl->node_record(impl->states.at(state->handle_id), node_id);
  return record ? StatisticView{record.get(), &record->fields}
                : StatisticView{};
}
ProfileEnvironment::BatchViews
ProfileEnvironment::batch_state_views(const std::vector<StatePtr> &states) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  BatchViews batch;
  batch.states.reserve(states.size());
  std::size_t total_nodes = 0;
  std::vector<const StateData *> resolved;
  resolved.reserve(states.size());
  for (const auto &state : states) {
    const auto &data = impl->states.at(state->handle_id);
    resolved.push_back(&data);
    total_nodes += data.nodes.size();
  }
  batch.views.reserve(total_nodes);
  for (const auto *data : resolved) {
    batch.states.push_back({&data->nodes, batch.views.size()});
    for (std::size_t slot : data->node_slots) {
      const auto &record = data->node_records[slot];
      batch.views.push_back(
          record ? StatisticView{record.get(), &record->fields}
                 : StatisticView{});
    }
  }
  return batch;
}
std::vector<StatisticView>
ProfileEnvironment::batch_added_views(const std::vector<StatePtr> &states) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::vector<StatisticView> result;
  result.reserve(states.size());
  for (const auto &state : states) {
    if (!state->added) {
      result.push_back({});
      continue;
    }
    const auto &data = impl->states.at(state->handle_id);
    const auto &record = impl->node_record(data, state->added->node_id);
    result.push_back(record ? StatisticView{record.get(), &record->fields}
                            : StatisticView{});
  }
  return result;
}
std::map<Id, std::map<std::string, std::shared_ptr<const Json>>>
ProfileEnvironment::statistic_fields(const StatePtr &state) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  std::map<Id, std::map<std::string, Value>> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->fields;
  return result;
}
Json ProfileEnvironment::cache(const StatePtr &) const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  Json profile = Json::object(), history = Json::array();
  profile["remaining_selectivity"] = impl->remaining_selectivity_;
  for (const auto *key :
       {"models", "accuracy", "task_accuracy", "request_profiles",
        "task_request_profiles", "relational_profiles", "constraints",
        "system_prefix", "operator_specifications", "cost_seconds_per_unit",
        "operator_time_calibration"})
    if (impl->profile_.contains(key))
      profile[key] = impl->profile_.at(key);
  for (const auto &source : impl->history_) {
    Json row = Json::object();
    for (auto field = source.begin(); field != source.end(); ++field)
      if (field.key() != "summary")
        row[field.key()] = field.value();
    history.push_back(std::move(row));
  }
  return {{"profile", std::move(profile)},
          {"history", std::move(history)},
          {"version", 1}};
}
Json ProfileEnvironment::profile_overlay() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  Json profile = Json::object();
  profile["remaining_selectivity"] = impl->remaining_selectivity_;
  for (const auto *key : {"accuracy", "request_profiles",
                          "operator_time_calibration", "relational_profiles"})
    if (impl->profile_.contains(key))
      profile[key] = impl->profile_.at(key);
  return profile;
}
std::shared_ptr<const Json> ProfileEnvironment::history_shared() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->history_owner;
}
std::shared_ptr<const void> ProfileEnvironment::identity_token() const {
  return impl->identity_token;
}
Json ProfileEnvironment::time_calibration() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->profile_.value("operator_time_calibration", Json::object());
}
Json ProfileEnvironment::profile() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->imported->value("prepared_statistics", false) ? impl->profile_
                                                             : Json::object();
}
Json ProfileEnvironment::history() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->history_;
}
Json ProfileEnvironment::calls() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return impl->imported->value("sampling_calls", Json::object());
}
Json ProfileEnvironment::timings() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  Json result = impl->imported->value("sampling_timings", Json::object());
  for (const auto &metric : impl->metrics)
    result[metric.first] = {{"seconds", metric.second.first},
                            {"calls", metric.second.second}};
  return result;
}
Json ProfileEnvironment::config() const {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  return *impl->imported;
}
void ProfileEnvironment::apply_runtime(const StatePtr &state, Id operator_id,
                                       const Json &entry,
                                       const Json &observation) {
  std::lock_guard<std::recursive_mutex> lock(impl->api_mutex);
  impl->runtime_observed = true;
  impl->observation_workers.clear();
  impl->survival_memo.clear();
  impl->observe(state);
  auto &data = impl->states.at(state->handle_id);
  auto action = std::find_if(
      data.actions.begin(), data.actions.end(),
      [&](const auto &value) { return value->operator_id == operator_id; });
  if (action == data.actions.end())
    throw std::invalid_argument("Runtime observation has no selected operator");
  Record info = *impl->node_record(data, (*action)->node_id);
  auto completed = entry.at("completed").get<double>();
  if (entry.at("status") == "DONE") {
    info.set("input_rows", entry.at("completed"));
    info.set("output_rows", entry.at("emitted"));
    if (completed)
      info.set("selectivity", entry.at("emitted").get<double>() / completed);
  } else {
    if (!observation.is_null() && !observation.empty()) {
      info.set("selectivity", observation.at("selectivity"));
      info.set(
          "effective_samples",
          observation.value("effective_samples", observation.at("completed")));
      auto reliability = info.get("reliability").is_object()
                             ? info.get("reliability")
                             : Json::object();
      auto count =
          observation.value("effective_samples", observation.at("completed"))
              .get<double>();
      reliability["selectivity"] = count / (count + 1);
      info.set("reliability", std::move(reliability));
    }
    if (completed && !info.get("input_rows").is_null()) {
      auto incoming = std::max(completed, info.get("input_rows").get<double>());
      info.set("input_rows", incoming);
      info.set("output_rows",
               info.get("selectivity").is_null()
                   ? Json(nullptr)
                   : Json(entry.at("emitted").get<double>() +
                          (incoming - completed) *
                              info.get("selectivity").get<double>()));
    }
  }
  // Semantic selectivity is shared across implementations. Runtime evidence
  // replaces that estimate too; leaving it unchanged would make the value
  // encoder ignore the newly observed selectivity.
  if (!info.get("selectivity").is_null() &&
      (entry.at("status") == "DONE" || !observation.is_null())) {
    info.set("estimated_selectivity", info.get("selectivity"));
    auto valid =
        info.get("valid").is_object() ? info.get("valid") : Json::object();
    valid["selectivity"] = true;
    info.set("valid", std::move(valid));
    if (number(info.get("selectivity"), true))
      data.runtime_selectivities[(*action)->node_id] =
          info.get("selectivity").get<double>();
  }
  if (observation.is_object() && observation.contains("input_statistics")) {
    const auto &actual = observation.at("input_statistics");
    const auto &summaries = actual.at("input_field_summaries");
    const auto &chosen = **action;
    const auto &op = impl->ops.at(operator_id);
    const auto &layout = chosen.action.at("layout");
    auto attach = [&](const char *order, const char *slot) {
      const auto names = layout.value(order, Json::array());
      if (names.empty()) return;
      auto key = names[0].get<std::string>();
      for (const auto &field : input_schema(op))
        if (field.at("name") == names[0]) key = field.value("value_key", key);
      if (summaries.contains(key)) info.set(slot, summaries.at(key).at("summary"));
    };
    attach(layout.contains("field_order") ? "field_order" : "left_field_order", "summary");
    attach("right_field_order", "right_summary");
    const bool pairs = kind(op).find("JOIN") != std::string::npos;
    info.set("input_field_statistics", actual.at("input_field_statistics"));
    info.set("input_field_summaries", summaries);
    info.set("pair_input", pairs);
    info.set("pair_request_summary", pairs);
    info.set("alpha", 1.0);
    info.set("right_alpha", 1.0);
    for (const auto *key : {"cache_field_origin", "pre", "l_existing_features"}) info.erase(key);
    info.set("runtime_statistics_basis", "actual consumed execution HTTP inputs");
    impl->cache_edges(info, data, data.nodes, impl->indexed_node(data, chosen.node_id), chosen);
    auto actual_valid = info.get("valid").is_object() ? info.get("valid") : Json::object();
    actual_valid["l_self"] = !info.get("l_self").is_null();
    info.set("valid", std::move(actual_valid));
    auto actual_reliability = info.get("reliability").is_object() ? info.get("reliability") : Json::object();
    const auto requests = actual.at("observed_requests").get<double>();
    actual_reliability["l_self"] = requests / (requests + 1.);
    info.set("reliability", std::move(actual_reliability));
  } else if (semantic(impl->ops.at(operator_id)) &&
             info.contains("summary") && !info.get("summary").is_null()) {
    const auto &chosen = **action;
    const auto &node = impl->indexed_node(data, chosen.node_id);
    impl->estimate_survivals(info, data, data, node, chosen);
    impl->cache_edges(info, data, data.nodes, node, chosen);
  }
  data.node_records[impl->node_index->slots.at((*action)->node_id)] =
      std::make_shared<const Record>(std::move(info));
}
}
