#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

namespace semcache::sampling {
enum class RangeWork {
  General,
  RequestKeys,
  JoinMetadata,
  JoinProbes,
  JoinRestore,
  Count
};

// Search-local measurements, never sampling evidence. No tuple-count cutoff:
// compare estimated useful work with measured launch and handoff costs.
class RangePolicy {
  struct Profile {
    double unit_ns = 0;
    bool observed = false;
    std::optional<double> handoff_ns;
  };
  std::array<Profile, static_cast<std::size_t>(RangeWork::Count)> profiles_{};
  std::optional<double> handoff_ns_;
  double launch_ns_ = 0;

public:
  std::size_t choose(RangeWork kind, std::size_t count, std::size_t limit,
                     std::size_t live) const {
    const auto cap = std::min(count, limit);
    const auto &profile = profiles_.at(static_cast<std::size_t>(kind));
    if (cap < 2 || !profile.observed)
      return 1;
    // One bounded two-way trial measures the pool's own handoff cost. It does
    // real work once, never repeats samples, and never allocates the full cap.
    if (!handoff_ns_)
      return 2;
    const auto work = profile.unit_ns * count;
    auto best = work;
    std::size_t selected = 1;
    const auto handoff = profile.handoff_ns.value_or(*handoff_ns_);
    for (std::size_t n = 2; n <= cap; ++n) {
      const auto startup = n > live ? launch_ns_ * (n - live) : 0.;
      const auto predicted = work / n + handoff * (n - 1) + startup;
      if (predicted < best) {
        best = predicted;
        selected = n;
      }
    }
    return selected;
  }
  void serial(RangeWork kind, std::size_t count, double ns) {
    if (!count)
      return;
    auto &profile = profiles_.at(static_cast<std::size_t>(kind));
    profile.unit_ns = std::max(0., ns) / count;
    profile.observed = true;
  }
  void parallel(RangeWork kind, std::size_t count, std::size_t workers,
                double ns, double startup_ns, std::size_t added) {
    auto &profile = profiles_.at(static_cast<std::size_t>(kind));
    const auto overhead =
        std::max(0., ns - startup_ns - profile.unit_ns * count / workers) /
        (workers - 1);
    profile.handoff_ns = overhead;
    handoff_ns_ = overhead;
    if (added)
      launch_ns_ = std::max(0., startup_ns) / added;
  }
};
}
