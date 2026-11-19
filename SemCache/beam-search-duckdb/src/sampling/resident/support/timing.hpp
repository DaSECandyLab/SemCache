#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace semcache::sampling::resident_detail {
struct PhaseMetric {
  double seconds = 0;
  std::size_t calls = 0;
  std::vector<std::array<std::int64_t, 2>> spans;
};
struct PhaseTimer {
  std::map<std::string, PhaseMetric> &metrics;
  std::string name;
  bool trace, active;
  std::chrono::steady_clock::time_point start;
  PhaseTimer(std::map<std::string, PhaseMetric> &m, std::string_view n,
             bool tracing, bool diagnostic_only = false)
      : metrics(m), trace(tracing), active(tracing || !diagnostic_only) {
    if (active) {
      name = n;
      start = std::chrono::steady_clock::now();
    }
  }
  ~PhaseTimer() {
    if (!active)
      return;
    const auto end = std::chrono::steady_clock::now();
    auto &value = metrics[name];
    value.seconds += std::chrono::duration<double>(end - start).count();
    ++value.calls;
    if (trace)
      value.spans.push_back(
          {std::chrono::duration_cast<std::chrono::nanoseconds>(
               start.time_since_epoch())
               .count(),
           std::chrono::duration_cast<std::chrono::nanoseconds>(
               end.time_since_epoch())
               .count()});
  }
};
}
