#include "../detail.hpp"
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>

namespace semcache::sampling {
bool NativeSampler::Impl::resolve_parallel_groups(
    const Json &op, const SampleInputs &inputs,
    const std::vector<Group> &observed, const std::function<void()> &prepare) {
  PhaseTimer timer(phase_metrics, "selection_parallel_models", trace_timings);
  auto supported = models(op);
  supported.resize(std::min(std::size_t(3), supported.size()));
  const auto concurrency = positive("max_concurrent_requests_per_model", 1);
  if (concurrency > 1024)
    throw std::invalid_argument(
        "Sampling model concurrency must be in [1, 1024]");
  std::set<std::string> claimed;
  if (state_actor)
    for (const auto &group : observed)
      if (!ledger.find(group.key))
        claimed.insert(group.key);
  StateClaim claim(*this, selection_claims, {claimed.begin(), claimed.end()});
  auto available = selection_remaining();
  while (state_actor && !available && pending_selection) {
    const auto before = pending_selection;
    state_actor->task.wait([&] { return pending_selection < before; });
    available = selection_remaining();
  }
  std::vector<const Group *> missing;
  for (const auto &group : observed)
    if (!ledger.find(group.key))
      missing.push_back(&group);
  if (missing.empty())
    return true;
  if (deadline_reached() || !available)
    return false;

  const auto started = std::chrono::steady_clock::now();
  const auto oid = op.at("operator_id").get<Id>();
  // Fill one lane per model first, then use one shared queue. A completion
  // immediately gives the next input to whichever model lane became idle;
  // model latency therefore determines subsequent assignment.
  const auto count = std::min(missing.size(), available);
  for (std::size_t i = 0; i < count; ++i)
    if (missing[i]->unit.empty())
      throw std::logic_error("Parallel selection lost a missing input unit");
  const auto lanes_per_model = std::min(concurrency, count);
  const auto capacity = std::min(count, supported.size() * lanes_per_model);
  if (selection_workers.size() < capacity)
    selection_workers.resize(capacity);
  struct Job {
    const Group *group = nullptr;
    std::size_t model = 0;
    std::future<Json> result;
  };
  std::vector<Job> jobs(capacity);
  std::mutex ready_mutex;
  std::condition_variable ready_changed;
  std::queue<std::size_t> ready;
  std::size_t active = 0, reserved = count, next_group = 0;
  pending_selection += reserved;
  std::exception_ptr error;
  auto launch = [&](std::size_t slot, std::size_t model) {
    if (next_group == count)
      return false;
    auto *group = missing[next_group++];
    auto &job = jobs[slot];
    auto &worker = selection_workers[slot];
    // Only this lane accesses its connection. The owner retains the state
    // turn while requests run, so ledger/history/publication remain serial.
    job.result = std::async(std::launch::async, [&, group, model, slot,
                                                 parent = backend] {
      auto complete = [&] {
        {
          std::lock_guard<std::mutex> lock(ready_mutex);
          ready.push(slot);
        }
        ready_changed.notify_one();
      };
      try {
        if (!worker)
          worker = parent->fork_semantic();
        Json action = {{"model_id", supported[model]},
                       {"layout", op.at("default_layout")},
                       {"llm_request_limit", 1}};
        Json output;
        if (group->unit.size() == 1) {
          Json rows = Json::array({group->unit[0]});
          output = worker->execute(op, {&rows}, action);
        } else
          output = worker->join_pairs(op, Json::array({group->unit}), action);
        complete();
        return output;
      } catch (...) {
        complete();
        throw;
      }
    });
    job.group = group;
    job.model = model;
    ++active;
    counters["selectivity_parallel_peak"] = std::max(
        counters.value("selectivity_parallel_peak", std::size_t(0)), active);
    return true;
  };
  try {
    std::size_t slot = 0;
    for (std::size_t lane = 0; lane < lanes_per_model && next_group < count;
         ++lane)
      for (std::size_t model = 0;
           model < supported.size() && next_group < count; ++model)
        launch(slot++, model);
    if (prepare)
      prepare();
  } catch (...) {
    error = std::current_exception();
  }
  while (active) {
    std::size_t slot;
    {
      std::unique_lock<std::mutex> lock(ready_mutex);
      ready_changed.wait(lock, [&] { return !ready.empty(); });
      slot = ready.front();
      ready.pop();
    }
    auto &job = jobs[slot];
    Json output;
    bool succeeded = false;
    try {
      output = job.result.get();
      succeeded = true;
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
    --active;
    --reserved;
    --pending_selection;
    const auto &worker = selection_workers[slot];
    if (worker)
      try {
        const auto progress = worker->selection_progress();
        const auto sends = progress.at("requests").get<std::size_t>();
        if (sends > 1 || !ledger.charge(sends))
          throw std::logic_error("Parallel selection exceeded request budget");
        selection_requests_by_operator[oid] += sends;
        counters["selectivity"] =
            counters.value("selectivity", std::size_t(0)) + 1;
        counters["sql_stages"] =
            counters.value("sql_stages", std::size_t(0)) + 1;
        auto &model_counts =
            counters["selection_requests_by_model"][supported[job.model]];
        model_counts =
            (model_counts.is_null() ? 0 : model_counts.get<std::size_t>()) +
            sends;
        if (succeeded && progress.at("completed_inputs") == 1) {
          const auto &unit = job.group->unit;
          const auto id =
              unit.size() == 1
                  ? unit[0].at("id").get<std::string>()
                  : digest(Json::array({unit[0].at("id"), unit[1].at("id")}));
          const Json *row = nullptr;
          for (const auto &candidate : output)
            if (candidate.at("id") == id)
              row = &candidate;
          const auto column = output_column(op);
          const Json value = row && !column.empty()
                                 ? row->at("values").at(column)
                                 : Json(nullptr);
          // Shared selectivity and model-specific quality retain separate
          // identities; only the invoked model owns the physical label.
          ledger.publish(job.group->key, row != nullptr, value);
          RequestKeyPlan physical(op, supported[job.model],
                                  op.at("default_layout"), unit.size() == 2);
          ledger.publish(
              physical(unit[0], unit.size() == 2 ? &unit[1] : nullptr),
              row != nullptr, value);
        }
        harvest(worker.get());
      } catch (...) {
        if (!error)
          error = std::current_exception();
      }
    // A completion immediately refills whichever model lane became idle.
    // Errors/deadlines stop admission while every started job drains.
    if (!error && !deadline_reached() && next_group < count)
      try {
        launch(slot, job.model);
      } catch (...) {
        error = std::current_exception();
      }
  }
  pending_selection -= reserved;
  if (state_actor)
    state_actor->task.notify();
  backend_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (error)
    std::rethrow_exception(error);
  for (const auto *group : missing)
    if (!ledger.find(group->key))
      return false;
  return true;
}
}
