#include "parallel.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace semcache::sampling {
namespace {
using Clock = std::chrono::steady_clock;
double nanoseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::nano>(duration).count();
}
}
ParallelRanges::ParallelRanges(std::size_t workers) : workers_(workers) {
  if (!workers || workers > 64)
    throw std::invalid_argument("CPU workers must be in [1,64]");
  errors_.resize(workers);
  threads_.reserve(workers - 1);
}
ParallelRanges::~ParallelRanges() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  for (auto &slot : threads_)
    slot->ready.notify_one();
  for (auto &slot : threads_)
    slot->thread.join();
}
void ParallelRanges::start(std::size_t active) {
  try {
    while (threads_.size() + 1 < active) {
      auto slot = std::make_unique<Slot>();
      const auto rank = threads_.size() + 1;
      auto *borrowed = slot.get();
      threads_.push_back(std::move(slot));
      borrowed->thread =
          std::thread([this, rank, borrowed] { worker(rank, *borrowed); });
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    for (auto &slot : threads_)
      slot->ready.notify_one();
    for (auto &slot : threads_)
      if (slot->thread.joinable())
        slot->thread.join();
    threads_.clear();
    stop_ = false;
    throw;
  }
}
void ParallelRanges::execute(std::size_t rank) noexcept {
  const auto width = count_ / active_, extra = count_ % active_;
  const auto begin = rank * width + std::min(rank, extra);
  const auto end = begin + width + (rank < extra);
  try {
    operation_(begin, end);
  } catch (...) {
    errors_[rank] = std::current_exception();
  }
}
void ParallelRanges::worker(std::size_t rank, Slot &slot) {
  std::size_t generation = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    slot.ready.wait(lock, [&] { return stop_ || generation != generation_; });
    if (stop_)
      return;
    generation = generation_;
    if (rank >= active_)
      continue;
    lock.unlock();
    execute(rank);
    lock.lock();
    if (!--pending_)
      finished_.notify_one();
  }
}
void ParallelRanges::run(
    std::size_t count,
    const std::function<void(std::size_t, std::size_t)> &operation,
    RangeWork kind, bool force_parallel) {
  if (!count)
    return;
  if (workers_ == 1) {
    ++serial_batches_;
    operation(0, count);
    return;
  }
  const auto chosen = force_parallel ? std::min(count, workers_)
                                     : policy_.choose(kind, count, workers_,
                                                      threads_.size() + 1);
  if (chosen == 1) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ = 1;
    }
    ++serial_batches_;
    const auto start = Clock::now();
    operation(0, count);
    policy_.serial(kind, count, nanoseconds(Clock::now() - start));
    return;
  }
  const auto start_time = Clock::now();
  const auto previous = threads_.size();
  start(chosen);
  const auto startup = nanoseconds(Clock::now() - start_time);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = chosen;
    peak_ = std::max(peak_, active_);
    ++parallel_batches_;
    std::fill(errors_.begin(), errors_.end(), nullptr);
    operation_ = operation;
    count_ = count;
    pending_ = active_ - 1;
    ++generation_;
  }
  for (std::size_t i = 0; i + 1 < chosen; ++i)
    threads_[i]->ready.notify_one();
  execute(0);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [&] { return pending_ == 0; });
    operation_ = {};
  }
  for (std::size_t rank = 0; rank < chosen; ++rank)
    if (errors_[rank])
      std::rethrow_exception(errors_[rank]);
  policy_.parallel(kind, count, chosen, nanoseconds(Clock::now() - start_time),
                   startup, threads_.size() - previous);
}
}
