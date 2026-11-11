#pragma once
#include "../../common/parallel.hpp"
#include "../timing.hpp"
#include <algorithm>

namespace semcache::sampling::resident_detail {
// Read-only keys are speculative; consume them in source order, including
// errors, so a later malformed tuple never overrides an earlier budget exit.
class RequestKeys {
  struct Pending {
    std::string key;
    std::exception_ptr failure;
  };
  ParallelRanges &workers_;
  std::map<std::string, PhaseMetric> &metrics_;
  std::size_t total_, begin_ = 0, end_ = 0;
  bool parallel_, trace_;
  std::vector<Pending> keys_;

public:
  RequestKeys(ParallelRanges &workers,
              std::map<std::string, PhaseMetric> &metrics, bool trace,
              std::size_t total)
      : workers_(workers), metrics_(metrics), total_(total),
        parallel_(workers.size() > 1 && total >= 256), trace_(trace) {}

  template <class Compute, class Recover>
  std::string at(std::size_t index, Compute compute, Recover recover) {
    if (!parallel_)
      return recover(index);
    if (index < begin_ || index >= end_) {
      PhaseTimer timer(metrics_, "parallel_request_keys", trace_);
      begin_ = index;
      end_ = begin_ + std::min(std::size_t(512), total_ - begin_);
      keys_.clear();
      keys_.resize(end_ - begin_);
      workers_.run(
          keys_.size(),
          [&](std::size_t from, std::size_t until) {
            for (auto offset = from; offset < until; ++offset) {
              try {
                keys_[offset].key = compute(begin_ + offset);
              } catch (...) {
                keys_[offset].failure = std::current_exception();
              }
            }
          },
          RangeWork::RequestKeys);
    }
    auto &key = keys_[index - begin_];
    // `at()` may be called more than once for an index when a request is
    // reused. Do not consume the prefetched key on the first lookup.
    return key.failure ? recover(index) : key.key;
  }
};
}
