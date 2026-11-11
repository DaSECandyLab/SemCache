#pragma once
#include "../search/state_core.hpp"
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace semcache::sampling {
using Json = nlohmann::ordered_json;
// Search consumes measured/profile environments through the same native
// ownership boundary. Configuration and final exports stay outside the loop.
class SearchEnvironment {
public:
  virtual ~SearchEnvironment() = default;
  virtual void observe(const StatePtr &state) = 0;
  virtual void refresh(const std::vector<StatePtr> &states) = 0;
  virtual void retain(const std::vector<StatePtr> &live) = 0;
  virtual Json statistics(const StatePtr &state) const = 0;
  // Diagnostic snapshots borrow immutable scalar blocks where available.
  // Profile-only environments can freeze their compact public statistics.
  virtual std::map<Id, std::shared_ptr<const Json>>
  statistic_blocks(const StatePtr &state) const {
    std::map<Id, std::shared_ptr<const Json>> result;
    const auto values = statistics(state);
    for (auto it = values.begin(); it != values.end(); ++it)
      result.emplace(std::stoll(it.key()),
                     std::make_shared<const Json>(it.value()));
    return result;
  }
  virtual Json full_statistics(const StatePtr &state) const = 0;
  virtual std::map<Id, std::map<std::string, std::shared_ptr<const Json>>>
  statistic_fields(const StatePtr &state) const {
    std::map<Id, std::map<std::string, std::shared_ptr<const Json>>> result;
    for (const auto &block : statistic_blocks(state))
      for (auto field = block.second->begin(); field != block.second->end();
           ++field)
        result[block.first][field.key()] =
            std::shared_ptr<const Json>(block.second, &field.value());
    return result;
  }
  virtual Json cache(const StatePtr &state) const = 0;
  virtual Json time_calibration() const { return Json::object(); }
  virtual Json profile() const = 0;
  virtual Json history() const = 0;
  virtual Json calls() const = 0;
  virtual Json timings() const = 0;
};
}
