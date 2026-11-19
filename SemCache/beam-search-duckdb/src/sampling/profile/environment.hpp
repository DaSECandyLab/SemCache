#pragma once
#include "../environment.hpp"
#include <memory>

namespace semcache::sampling {
// Borrowed immutable fields, valid until the owning state's next mutation or
// release. Inference consumes these synchronously while the state is retained.
struct StatisticView {
  const void *identity = nullptr;
  const void *fields = nullptr;
  const Json *find(const std::string &key) const;
  bool contains(const std::string &key) const { return find(key) != nullptr; }
  const Json &get(const std::string &key) const;
};
// Imported profiles are immutable evidence. States retain shared field values
// and independently owned estimate overlays, without Python observation hooks.
class ProfileEnvironment final : public SearchEnvironment {
public:
  explicit ProfileEnvironment(const Json &config);
  ~ProfileEnvironment() override;
  void observe(const StatePtr &state) override;
  void observe_batch(const std::vector<StatePtr> &states, std::size_t workers);
  void refresh(const std::vector<StatePtr> &states) override;
  void accuracy(const StatePtr &state);
  void retain(const std::vector<StatePtr> &live) override;
  Json statistics(const StatePtr &state) const override;
  Json full_statistics(const StatePtr &state) const override;
  std::map<Id, std::shared_ptr<const Json>>
  statistic_blocks(const StatePtr &state) const override;
  // Compact immutable inputs for value-model inference; public statistics
  // retain their complete scalar and evidence exports.
  std::map<Id, std::shared_ptr<const Json>>
  feature_blocks(const StatePtr &state) const;
  std::map<Id, StatisticView> feature_views(const StatePtr &state) const;
  StatisticView feature_view(const StatePtr &state, Id node_id) const;
  // Borrowed node/record views captured under one lock; synchronous inference only.
  struct BatchViews {
    struct Entry {
      const std::vector<NodePtr> *nodes = nullptr;
      std::size_t offset = 0;
    };
    std::vector<Entry> states;
    std::vector<StatisticView> views;
  };
  BatchViews batch_state_views(const std::vector<StatePtr> &states) const;
  std::vector<StatisticView>
  batch_added_views(const std::vector<StatePtr> &states) const;
  std::map<Id, std::map<std::string, std::shared_ptr<const Json>>>
  statistic_fields(const StatePtr &state) const override;
  Json cache(const StatePtr &state) const override;
  Json profile_overlay() const;
  std::shared_ptr<const Json> history_shared() const;
  std::shared_ptr<const void> identity_token() const;
  Json profile() const override;
  Json time_calibration() const override;
  Json history() const override;
  Json calls() const override;
  Json timings() const override;
  Json config() const;
  void apply_runtime(const StatePtr &state, Id operator_id, const Json &entry,
                     const Json &observation = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
