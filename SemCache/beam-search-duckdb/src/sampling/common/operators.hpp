#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace semcache::sampling {
inline bool fixed_relation_filter(const nlohmann::ordered_json &op) {
  const auto kind = op.value("kind", op.value("type", std::string()));
  return kind == "FILTER" && op.value("sampling_row_local", false) &&
         op.contains("sampling_required_fields") &&
         op["sampling_required_fields"].is_array() &&
         op.value("sampling_row_local_sql", std::string()) ==
             op.value("sql", op.value("sampling_sql", std::string()));
}
inline bool semantic_operator(const nlohmann::ordered_json &op) {
  const auto flag = op.find("semantic");
  if (flag != op.end())
    return flag->get<bool>();
  const auto kind = op.find("kind");
  return kind != op.end() &&
         kind->get_ref<const std::string &>().rfind("SEMANTIC", 0) == 0;
}
}
