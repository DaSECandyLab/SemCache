#pragma once
#include "parallel.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace semcache::sampling {
// Whole, stack-owning tasks share a short CPU ownership section. A private
// backend may run outside it; publication rejoins the original task order.
class OrderedTasks {
  ParallelRanges workers_;
  struct Wave {
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t start = 0, commit = 0;
    bool failed = false;
  };

public:
  class Task {
    Wave &wave_;
    std::unique_lock<std::mutex> lock_;
    std::function<void()> leave_, enter_;

  public:
    const std::size_t index, slot;
    Task(Wave &wave, std::size_t i, std::size_t worker)
        : wave_(wave), lock_(wave.mutex), index(i), slot(worker) {
      wave.changed.wait(lock_, [&] { return wave.start == index; });
      ++wave.start;
      wave.changed.notify_all();
    }
    void hooks(std::function<void()> leave = {},
               std::function<void()> enter = {}) {
      leave_ = std::move(leave);
      enter_ = std::move(enter);
    }
    bool failed() const { return wave_.failed; }
    void notify() { wave_.changed.notify_all(); }
    template <class Predicate> void wait(Predicate ready) {
      if (ready())
        return;
      if (leave_)
        leave_();
      wave_.changed.wait(lock_, ready);
      if (enter_)
        enter_();
    }
    void turn() {
      wait([&] { return wave_.commit == index; });
    }
    template <class Operation> auto outside(Operation operation) {
      if (leave_)
        leave_();
      lock_.unlock();
      try {
        auto result = operation();
        lock_.lock();
        if (enter_)
          enter_();
        return result;
      } catch (...) {
        lock_.lock();
        if (enter_)
          enter_();
        throw;
      }
    }
    void finish(bool failed) {
      turn();
      wave_.failed |= failed;
      ++wave_.commit;
      wave_.changed.notify_all();
    }
  };

  explicit OrderedTasks(std::size_t workers) : workers_(workers) {}
  std::size_t size() const { return workers_.size(); }
  std::size_t peak() const { return workers_.peak_size(); }
  void run(std::size_t count, const std::function<void(Task &)> &operation,
           const std::function<void(std::size_t)> &initialize = {}) {
    if (!count)
      return;
    Wave wave;
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(count);
    // Adaptive persistent ranges dispatch from a shared queue; a worker takes
    // multiple states, rather than reserving a contiguous block behind a slow
    // first state. The configured width is only an upper bound.
    workers_.run(
        std::min(count, workers_.size()),
        [&](std::size_t begin, std::size_t) {
          // Opening a private connection must precede ticket
          // allocation. A later task must not acquire a root while
          // an earlier ticket is still booting its worker.
          if (initialize)
            initialize(begin);
          for (;;) {
            const auto index = next.fetch_add(1);
            if (index >= count)
              return;
            Task task(wave, index, begin);
            if (!task.failed())
              try {
                operation(task);
              } catch (...) {
                errors[index] = std::current_exception();
              }
            task.finish(bool(errors[index]));
          }
        },
        RangeWork::General, true);
    for (const auto &error : errors)
      if (error)
        std::rethrow_exception(error);
  }
};
}
