#pragma once
#include "range_policy.hpp"
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace semcache::sampling {
// Search-local independent ranges. The limit is a cap: measured work and
// scheduling cost choose the active width; unused workers are not awakened.
class ParallelRanges {
public:
  explicit ParallelRanges(std::size_t workers);
  ~ParallelRanges();
  ParallelRanges(const ParallelRanges &) = delete;
  ParallelRanges &operator=(const ParallelRanges &) = delete;
  std::size_t size() const { return workers_; }
  std::size_t last_size() const { return active_; }
  std::size_t peak_size() const { return peak_; }
  std::size_t serial_batches() const { return serial_batches_; }
  std::size_t parallel_batches() const { return parallel_batches_; }
  void run(std::size_t count,
           const std::function<void(std::size_t, std::size_t)> &operation,
           RangeWork kind = RangeWork::General, bool force_parallel = false);

private:
  struct Slot {
    std::condition_variable ready;
    std::thread thread;
  };
  void start(std::size_t active);
  void worker(std::size_t rank, Slot &slot);
  void execute(std::size_t rank) noexcept;
  const std::size_t workers_;
  RangePolicy policy_;
  std::vector<std::unique_ptr<Slot>> threads_;
  std::vector<std::exception_ptr> errors_;
  std::mutex mutex_;
  std::condition_variable finished_;
  std::function<void(std::size_t, std::size_t)> operation_;
  std::size_t count_ = 0, generation_ = 0, pending_ = 0, active_ = 1, peak_ = 1,
              serial_batches_ = 0, parallel_batches_ = 0;
  bool stop_ = false;
};
}
