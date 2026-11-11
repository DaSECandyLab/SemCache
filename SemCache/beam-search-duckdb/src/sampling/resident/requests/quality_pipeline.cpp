#include "../detail.hpp"
#include "../../common/accuracy.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>

namespace semcache::sampling {
namespace {
struct QualityItem {
  std::shared_ptr<const std::vector<Json>> unit;
  std::string key;
};
struct QualityJob {
  const Json *op;
  std::vector<QualityItem> items;
  std::string model;
  bool judge;
  Json layout;
  std::array<std::int64_t, 2> span{};
};
struct QualityPair {
  const Json *op;
  std::string model, teacher, actual, reference, group;
  Json layout;
  double weight;
};
struct QualityCompletion {
  std::mutex mutex;
  std::condition_variable ready;
  void notify() {
    std::lock_guard<std::mutex> lock(mutex);
    ready.notify_one();
  }
};
}

void NativeSampler::Impl::accuracy_batch(const std::vector<StatePtr> &batch) {
  if (batch.empty() || deadline_reached())
    return;
  constexpr std::size_t packing = 1;
  struct Active {
    std::shared_ptr<QualityJob> job;
    std::future<Json> result;
  };
  std::vector<Active> active;
  std::deque<std::shared_ptr<QualityJob>> pending;
  std::vector<QualityPair> comparisons;
  std::set<std::string> scheduled;
  std::set<std::pair<std::string, std::string>> comparing;
  auto completion = std::make_shared<QualityCompletion>();
  std::exception_ptr error;
  auto fail = [&] {
    if (!error)
      error = std::current_exception();
  };
  auto running = [&] {
    return std::any_of(active.begin(), active.end(),
                       [](const auto &slot) { return bool(slot.job); });
  };
  auto finish = [&](std::size_t i) {
    auto &slot = active[i];
    const auto job = slot.job;
    Json output;
    bool succeeded = false;
    std::size_t completed = 0;
    try {
      output = slot.result.get();
      succeeded = true;
    } catch (const std::exception &exception) {
      const std::string reason = exception.what();
      const bool external_judge =
          job->judge && profile_.contains("teacher_models") &&
          profile_.at("teacher_models").contains(job->model);
      bool transient =
          reason.find("HTTP transport failure") != std::string::npos;
      for (const auto status : {"status=408", "status=429", "status=502",
                                "status=503", "status=504"})
        transient |= reason.find(status) != std::string::npos;
      if (external_judge && transient) {
        // Quality is optional new evidence, never a prerequisite for using the
        // trained prior. Charge the failed send but publish no label/accuracy.
        counters["accuracy_judge_unavailable"] =
            counters.value("accuracy_judge_unavailable", std::size_t(0)) + 1;
        counters["accuracy_judge_last_error"] = reason.substr(0, 2048);
      } else {
        fail();
      }
    } catch (...) {
      fail();
    }
    if (job->span[1] > job->span[0]) {
      const auto seconds = double(job->span[1] - job->span[0]) / 1e9;
      backend_seconds += seconds;
      auto &metric =
          phase_metrics[job->judge ? "accuracy_judge" : "accuracy_model"];
      metric.seconds += seconds;
      ++metric.calls;
      if (trace_timings)
        metric.spans.push_back(job->span);
    }
    // A first-use backend may fail before any transport was attempted. The
    // future joins its initialization before the owning thread reads it.
    if (!quality_workers[i]->backend) {
      slot.job.reset();
      return;
    }
    auto &source = *quality_workers[i]->backend;
    try {
      const auto requests =
          source.selection_progress().at("requests").get<std::size_t>();
      const auto oid = job->op->at("operator_id").get<Id>();
      if (requests > 1 || !ledger.charge(requests, true))
        throw std::logic_error(
            "Accuracy transport exceeded its single HTTP request limit");
      accuracy_request_used.at(oid) += requests;
      const auto name = job->judge ? "judge" : "accuracy";
      counters[name] = counters.value(name, std::size_t(0)) + requests;
      if (succeeded) {
        completed = source.selection_progress()
                        .at("completed_inputs")
                        .get<std::size_t>();
        if (completed > job->items.size())
          throw std::logic_error(
              "Accuracy transport completed too many inputs");
        std::map<std::string, const Json *> by_id;
        for (const auto &row : output)
          by_id.emplace(row.at("id").get<std::string>(), &row);
        const auto column = output_column(*job->op);
        for (std::size_t item = 0; item < completed; ++item) {
          const auto &unit = *job->items[item].unit;
          const auto id =
              unit.size() == 1
                  ? unit[0].at("id").get<std::string>()
                  : digest(Json::array({unit[0].at("id"), unit[1].at("id")}));
          const auto found = by_id.find(id);
          ledger.publish(job->items[item].key, found != by_id.end(),
                         found != by_id.end() && !column.empty()
                             ? found->second->at("values").at(column)
                             : Json(nullptr));
        }
      }
    } catch (...) {
      fail();
    }
    // Harvest successful evidence even when a stage fails. No backend is read
    // until its future is ready, and it cannot be reused before this harvest.
    try {
      harvest(&source);
    } catch (...) {
      fail();
    }
    slot.job.reset();
    // A byte-limited prompt can complete only part of an admitted batch. Keep
    // its paid answers; the suffix still belongs to the same operator batch.
    if (!error && succeeded && completed < job->items.size() &&
        !deadline_reached()) {
      if (!completed) {
        // The backend may stop after an invalid packed response consumes the
        // job's single-send allowance. Publish no labels and do not spin on
        // the same suffix; the existing accuracy prior remains available.
        counters["accuracy_incomplete_batches"] =
            counters.value("accuracy_incomplete_batches", std::size_t(0)) + 1;
        return;
      }
      auto suffix = std::make_shared<QualityJob>();
      suffix->op = job->op;
      suffix->model = job->model;
      suffix->judge = job->judge;
      suffix->layout = job->layout;
      suffix->items.assign(job->items.begin() + completed, job->items.end());
      pending.push_front(std::move(suffix));
    }
  };
  auto reap = [&](bool wait) {
    auto ready = [&] {
      for (auto &slot : active)
        if (slot.job && slot.result.wait_for(std::chrono::seconds(0)) ==
                            std::future_status::ready)
          return true;
      return false;
    };
    if (wait && running()) {
      std::unique_lock<std::mutex> lock(completion->mutex);
      completion->ready.wait(lock, ready);
    }
    for (std::size_t i = 0; i < active.size(); ++i)
      if (active[i].job && active[i].result.wait_for(std::chrono::seconds(0)) ==
                               std::future_status::ready)
        finish(i);
  };
  auto dispatch = [&](bool flush_partial = true) {
    if (error || deadline_reached())
      return;
    for (std::size_t i = 0; !pending.empty(); ++i) {
      if (i < active.size() && active[i].job)
        continue;
      // Beam quality observations are independent. Every active job owns a
      // private SQL backend and uses the unrestricted sampling scheduler, so
      // jobs targeting the same physical model must not wait for one another.
      // The operator batch limit was checked before admission. Request keys
      // deduplicate the shared teacher and repeated parameter evaluations.
      auto next =
          std::find_if(pending.begin(), pending.end(), [&](const auto &job) {
            if (!flush_partial && job->items.size() < packing)
              return false;
            return true;
          });
      if (next == pending.end())
        break;
      if (i == active.size())
        active.emplace_back();
      if (i == quality_workers.size())
        quality_workers.push_back(std::make_unique<QualityWorker>(nullptr));
      auto job = *next;
      auto *worker = quality_workers[i].get();
      auto execute = [worker, parent = backend, job] {
        // Initialize private SQL connections on their workers; reuse them after owner-thread harvest.
        if (!worker->backend)
          worker->backend = parent->fork_semantic();
        const auto &source = worker->backend;
        struct Measure {
          QualityJob &job;
          explicit Measure(QualityJob &j) : job(j) {
            job.span[0] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
          }
          ~Measure() {
            job.span[1] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
          }
        } measure{*job};
        Json action{{"model_id", job->model},
                    {"layout", job->layout},
                    {"llm_request_limit", 1}};
        if (job->items.front().unit->size() == 2) {
          Json pairs = Json::array();
          for (const auto &item : job->items)
            pairs.push_back(*item.unit);
          return source->join_pairs(*job->op, pairs, action);
        }
        Json rows = Json::array();
        for (const auto &item : job->items)
          rows.push_back(item.unit->at(0));
        return source->execute(*job->op, {&rows}, action);
      };
      active[i].result = quality_workers[i]->pipeline.submit(
          std::move(execute), [completion] { completion->notify(); });
      active[i].job = job;
      pending.erase(next);
      counters["accuracy_parallel_peak"] =
          std::max(counters.value("accuracy_parallel_peak", std::size_t(0)),
                   std::size_t(std::count_if(
                       active.begin(), active.end(),
                       [](const auto &slot) { return bool(slot.job); })));
      counters["sql_stages"] = counters["sql_stages"].get<std::size_t>() + 1;
    }
  };
  auto drain = [&] {
    while (running())
      reap(true);
  };
  try {
    for (const auto &state : batch) {
      reap(false);
      if (error)
        std::rethrow_exception(error);
      if (!states.count(state->handle_id)) {
        drain();
        if (error)
          std::rethrow_exception(error);
      }
      // Incomplete output can still have usable inputs. Preparing their SQL
      // expressions borrows the primary backend, so defer dispatch until this
      // state's input preparation finishes; complete observations stay async.
      const auto observed = states.find(state->handle_id);
      const bool preparing = observed == states.end() ||
                             std::any_of(observed->second.observations.begin(),
                                         observed->second.observations.end(),
                                         [](const auto &entry) {
                                           return !entry.second->core->complete;
                                         });
      if (preparing) {
        drain();
        if (error)
          std::rethrow_exception(error);
      }
      prepare_accuracy(state, [&](const Json &op, const Group &group,
                                  const std::vector<std::string> &models,
                                  const std::string &teacher,
                                  const Json &layout) {
        reap(false);
        if (error)
          std::rethrow_exception(error);
        // Every compatible model sees the same selected unit. Share its
        // immutable payload, and schedule the teacher only once per key.
        std::shared_ptr<const std::vector<Json>> unit;
        for (const auto &model : models) {
          const auto actual = request_key(op, group.unit, model, layout);
          const auto &teacher_layout = op.at("default_layout");
          const auto reference =
              request_key(op, group.unit, teacher, teacher_layout);
          const auto identity = std::make_pair(actual, teacher);
          if (quality_seen.count(identity) || comparing.count(identity))
            continue;
          std::map<std::string, std::pair<std::string, bool>> missing;
          for (const auto &entry : {std::make_pair(actual, model),
                                    std::make_pair(reference, teacher)})
            if (!ledger.find(entry.first) && !scheduled.count(entry.first))
              missing.emplace(
                  entry.first,
                  std::make_pair(entry.second, entry.second == teacher));
          auto open_job = [&](const std::string &model_id, bool judge,
                              const Json &job_layout) {
            const auto found = std::find_if(
                pending.rbegin(), pending.rend(), [&](const auto &job) {
                  return job->op == &op && job->model == model_id &&
                         job->layout == job_layout && job->judge == judge &&
                         job->items.size() < packing;
                });
            return found == pending.rend() ? std::shared_ptr<QualityJob>{}
                                           : *found;
          };
          comparisons.push_back({&op, model, teacher, actual, reference, actual,
                                 layout, group.weight});
          comparing.insert(identity);
          if (missing.empty())
            continue;
          if (!backend)
            throw std::runtime_error(
                "Native sampling requires a native SQL backend");
          if (!unit)
            unit = std::make_shared<const std::vector<Json>>(group.unit);
          for (const auto &key : {actual, reference}) {
            const auto entry = missing.find(key);
            if (entry == missing.end())
              continue;
            const auto &model_id = entry->second.first;
            const auto &job_layout = key == reference ? teacher_layout : layout;
            auto job = open_job(model_id, entry->second.second, job_layout);
            if (!job) {
              job = std::make_shared<QualityJob>();
              job->op = &op;
              job->model = model_id;
              job->judge = entry->second.second;
              job->layout = job_layout;
              pending.push_back(job);
            }
            job->items.push_back({unit, key});
            scheduled.insert(key);
            missing.erase(entry);
          }
        }
        if (!preparing)
          dispatch(false);
      });
      dispatch();
    }
    while (!pending.empty() || running()) {
      dispatch();
      if (!running())
        break;
      reap(true);
      if (error)
        break;
    }
  } catch (...) {
    fail();
  }
  // Never release borrowed task data or propagate an error with paid work
  // outstanding. Queued (unstarted) tasks are simply abandoned.
  drain();
  // Requests may finish in a different order. Expose history in actual
  // operator-start order, independently of transport completion order.
  std::stable_sort(
      history_.begin(), history_.end(), [](const auto &a, const auto &b) {
        return a.value("started_at", 0.) < b.value("started_at", 0.);
      });
  for (const auto &pair : comparisons) {
    auto actual = ledger.find(pair.actual),
         reference = ledger.find(pair.reference);
    if (!actual || !reference)
      continue;
    auto observation =
        Json::array({pair.actual, pair.reference, actual->success,
                     actual->value, reference->success, reference->value});
    // A cache-free sampling invocation obtains fresh responses. Repeated
    // comparisons within one invocation still share the same evidence.
    if (!imported.value("prompt_answers_enabled", true))
      observation.push_back(imported.value("sampling_epoch", Json(nullptr)));
    const auto observation_key = digest(observation);
    auto &seen = profile_["accuracy_observations"];
    if (seen.is_null())
      seen = Json::object();
    if (seen.contains(observation_key))
      continue;
    seen[observation_key] = true;
    const auto &op = *pair.op;
    const bool match = actual->success == reference->success &&
                       actual->value == reference->value;
    const auto layout_key = accuracy_layout_key(pair.layout);
    auto &global =
        profile_["accuracy"][pair.model][op.at("kind").get<std::string>()];
    update_quality(global, match, pair.weight, pair.teacher);
    update_quality(global["layouts"][layout_key], match, pair.weight,
                   pair.teacher);
    auto &task = profile_["task_accuracy"][pair.model][task_key(op)];
    update_quality(task, match, pair.weight, pair.teacher);
    update_quality(task["layouts"][layout_key], match, pair.weight,
                   pair.teacher);
    quality_seen.emplace(pair.group, pair.teacher);
  }
  if (error)
    std::rethrow_exception(error);
  for (const auto &state : batch)
    if (states.count(state->handle_id))
      finish_accuracy(state);
}
}
