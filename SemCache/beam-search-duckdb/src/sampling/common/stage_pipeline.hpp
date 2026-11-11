#pragma once
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace semcache::sampling {
// The SQL stage owns its backend until completion. Preparation borrows only
// immutable inputs and publishes no model outcomes.
template <class Result> class StagePipeline {
  std::mutex mutex_;
  std::condition_variable ready_;
  std::packaged_task<Result()> pending_;
  std::function<void()> completed_;
  std::thread worker_;
  bool stop_ = false, busy_ = false;

  void work() {
    for (;;) {
      std::packaged_task<Result()> task;
      std::function<void()> completed;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stop_ || pending_.valid(); });
        if (stop_ && !pending_.valid())
          return;
        task = std::move(pending_);
        completed = std::move(completed_);
      }
      task();
      if (completed)
        completed();
    }
  }

public:
  StagePipeline() = default;
  StagePipeline(const StagePipeline &) = delete;
  StagePipeline &operator=(const StagePipeline &) = delete;
  ~StagePipeline() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable())
      worker_.join();
  }

  // One outstanding stage per connection; only its owner submits work.
  // The future lets that owner prepare other tasks before joining the stage.
  std::future<Result> submit(std::function<Result()> stage,
                             std::function<void()> completed = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_ || stop_)
      throw std::logic_error("SQL stage already pending or stopped");
    if (!worker_.joinable())
      worker_ = std::thread([this] { work(); });
    std::packaged_task<Result()> task([this, stage = std::move(stage)] {
      struct Complete {
        StagePipeline &pipeline;
        ~Complete() {
          std::lock_guard<std::mutex> lock(pipeline.mutex_);
          pipeline.busy_ = false;
        }
      } complete{*this};
      return stage();
    });
    auto result = task.get_future();
    pending_ = std::move(task);
    completed_ = std::move(completed);
    busy_ = true;
    ready_.notify_one();
    return result;
  }

  // Without useful preparation retain the synchronous fast path.
  Result run(const std::function<Result()> &stage,
             const std::function<void()> &prepare = {}) {
    if (!prepare)
      return stage();
    auto result = submit(stage);
    std::exception_ptr preparation_error;
    try {
      prepare();
    } catch (...) {
      preparation_error = std::current_exception();
    }
    // get() also drains a failed stage. The backend's progress/events are
    // visible before the sampler charges actual HTTP submissions.
    auto output = result.get();
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    return output;
  }
};
}
