#pragma once
#include "../statistics/statistics.hpp"

namespace semcache::sampling {
inline Json accuracy_layouts(const Json &op) {
  const auto orders =
      op.value("layout_domain", Json::object())
          .value(
              "block_orders",
              Json::array({Json::array({"OPERATOR_SPECIFICATION",
                                        "TASK_DESCRIPTION", "DATA_BLOCK"}),
                           Json::array({"OPERATOR_SPECIFICATION", "DATA_BLOCK",
                                        "TASK_DESCRIPTION"}),
                           Json::array({"DATA_BLOCK", "OPERATOR_SPECIFICATION",
                                        "TASK_DESCRIPTION"})}));
  Json result = Json::array();
  for (const auto &order : orders) {
    auto layout = op.at("default_layout");
    layout["block_order"] = order;
    result.push_back(std::move(layout));
  }
  return result;
}

inline std::string accuracy_layout_key(const Json &layout) {
  std::string result;
  for (const auto &block : layout.at("block_order")) {
    if (!result.empty())
      result += ':';
    result += block.get<std::string>();
  }
  return result;
}
inline const Json &accuracy_evidence(const Json &estimate, const Json &layout) {
  if (estimate.is_object() && estimate.contains("layouts")) {
    const auto &layouts = estimate.at("layouts");
    const auto found = layouts.find(accuracy_layout_key(layout));
    static const Json unknown;
    return found == layouts.end() ? unknown : *found;
  }
  return estimate;
}
}
