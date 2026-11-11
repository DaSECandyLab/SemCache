#pragma once
#include "../../search/state_core.hpp"
#include "../environment.hpp"
#include "../sql/backend.hpp"
#include "../statistics/statistics.hpp"
#include <memory>
#include <vector>

namespace semcache::sampling {
// Own independent base cohorts during statistics preparation. Beam receives
// only the resulting immutable ProfileEnvironment, never these sample rows.
class NativeSampler : public SearchEnvironment {
public:
  NativeSampler(const Json &config, std::shared_ptr<NativeSQLBackend> backend);
  ~NativeSampler();
  void begin_search(double seconds);
  void end_search();
  void observe(const StatePtr &child);
  void refresh(const std::vector<StatePtr> &states);
  void accuracy(const StatePtr &state);
  void accuracy_batch(const std::vector<StatePtr> &states);
  Json calibrate_time(const StatePtr &state);
  void retain(const std::vector<StatePtr> &live);
  Json statistics(const StatePtr &state) const;
  std::map<Id, SharedSamples>
  statistic_blocks(const StatePtr &state) const override;
  Json full_statistics(const StatePtr &state) const;
  std::map<Id, std::map<std::string, SharedSamples>>
  statistic_fields(const StatePtr &state) const override;
  Json cache(const StatePtr &state) const;
  Json samples(const StatePtr &state) const;
  Json profile() const;
  Json history() const;
  Json calls() const;
  Json timings() const;
  Json config() const;
  Json telemetry() const;
  // Collect independent operator evidence once. The returned environment owns
  // statistics only: no candidate row banks, backend or refill machinery.
  std::shared_ptr<SearchEnvironment>
  prepare_statistics(const StateProblem &problem,
                     const ActionPriorities &priorities,
                     const std::optional<std::string> &fixed_model,
                     bool collect_accuracy, double max_seconds);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  void resample_empty(const StatePtr &state);
};
}
