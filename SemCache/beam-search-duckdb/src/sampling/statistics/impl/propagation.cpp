#include "impl_internal.hpp"
#include <regex>

namespace semcache::sampling {

using namespace stats_detail;

std::string prefix_namespace(const Json &op, const Json &action,
                             const std::string &system,
                             const std::string &spec) {
  const auto &layout = action.at("layout");
  Json fields = layout.value("field_order", Json::array());
  if (fields.empty()) {
    fields = layout.value("left_field_order", Json::array());
    for (auto &f : layout.value("right_field_order", Json::array()))
      fields.push_back(f);
  }
  Json inputs = op.value("input_fields", Json::array());
  if (inputs.empty()) {
    inputs = op.value("left_input_fields", Json::array());
    for (auto &f : op.value("right_input_fields", Json::array()))
      inputs.push_back(f);
  }
  std::map<std::size_t, std::size_t> inverse;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    auto pos = std::find(inputs.begin(), inputs.end(), fields[i]);
    if (pos == inputs.end())
      throw std::invalid_argument("Unknown layout field");
    inverse[pos - inputs.begin()] = i;
  }
  std::string task = op.at("task").get<std::string>(), translated;
  static const std::regex pattern("\\{([0-9]+)\\}");
  std::size_t previous = 0;
  for (std::sregex_iterator it(task.begin(), task.end(), pattern), end;
       it != end; ++it) {
    translated += task.substr(previous, it->position() - previous) + "{" +
                  std::to_string(inverse.at(std::stoull((*it)[1]))) + "}";
    previous = it->position() + it->length();
  }
  translated += task.substr(previous);
  std::string parts;
  for (auto &raw : layout.at("block_order")) {
    std::string block = raw;
    if (block == "DATA_BLOCK")
      break;
    if (!parts.empty())
      parts += "\n\n";
    parts += "<" + block + ">\n" +
             (block == "TASK_DESCRIPTION" ? translated : spec) + "\n</" +
             block + ">";
  }
  if (!parts.empty())
    parts += "\n\n";
  return canonical(
      Json::array({"prompt-v4", system, parts + "<DATA_BLOCK>\n{0}: "}));
}
Json prefix_length(const std::string &prefix) {
  const auto parts = Json::parse(prefix, nullptr, false);
  if (!parts.is_array() || parts.size() != 3 || !parts.at(1).is_string() ||
      !parts.at(2).is_string())
    return nullptr;
  const auto &system = parts.at(1).get_ref<const std::string &>();
  const auto &user = parts.at(2).get_ref<const std::string &>();
  return character_length(system) +
         character_length(user.substr(0, user.find("<DATA_BLOCK>")));
}
RequestKeyPlan::RequestKeyPlan(const Json &op, const std::string &model,
                               const Json &layout, bool two_inputs) {
  auto fixed =
      Json::array({op.at("kind"), op.at("task"),
                   op.value("output_schema", Json(nullptr)), model, layout});
  prefix_ = canonical(fixed);
  prefix_.back() = ',';
  Json names = op.value("input_fields", Json::array());
  if (names.empty()) {
    names = op.value("left_input_fields", Json::array());
    for (auto &n : op.value("right_input_fields", Json::array()))
      names.push_back(n);
  }
  std::map<std::string, std::string> descriptors;
  for (auto &d : op.value("input_field_schema", Json::array())) {
    std::string name = d.at("name");
    descriptors[name] = d.value("value_key", name);
  }
  auto right = op.value("right_input_fields", Json::array());
  for (auto &raw : names) {
    std::string name = raw;
    slots_.push_back({descriptors.count(name) ? descriptors.at(name) : name,
                      two_inputs && std::find(right.begin(), right.end(),
                                              raw) != right.end()});
  }
}
std::string RequestKeyPlan::operator()(const Json &left,
                                       const Json *right) const {
  return from_values(
      [&](const std::string &name, bool is_right) -> const Json & {
        if (is_right && !right)
          throw std::invalid_argument("Missing right semantic input");
        return (is_right ? *right : left).at("values").at(name);
      });
}
std::string request_key(const Json &op, const std::vector<Json> &unit,
                        const std::string &model, const Json &layout) {
  return RequestKeyPlan(op, model, layout, unit.size() == 2)(
      unit.at(0), unit.size() == 2 ? &unit[1] : nullptr);
}
Json propagate(const Json &op, const Json &prior,
               const std::vector<Json> &sizes) {
  Json out = prior;
  propagate_inplace(op, out, sizes);
  return out;
}
void propagate_inplace(const Json &op, Json &out,
                       const std::vector<Json> &sizes) {
  double incoming = 1;
  for (auto &n : sizes) {
    if (!number(n)) {
      for (auto &key : {"input_rows", "output_rows", "input_side_rows",
                        "relational_cost", "relational_seconds"})
        out.erase(key);
      return;
    }
    incoming *= n.get<double>();
  }
  if (sizes.empty())
    return;
  const bool group_prior = out.value("group_count_estimated", false);
  out.erase("group_count_estimated");
  Json rate = group_prior ? Json(nullptr)
                          : out.value("selectivity", Json(nullptr)),
       old = out.value("input_rows", Json(nullptr));
  std::string kind = op.value("kind", op.value("type", std::string()));
  std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
  const bool uncertain = out.value("uncertain", false) ||
                         out.value("survival_estimated", false) || group_prior;
  if (rate.is_null() && !uncertain && number(old) && old.get<double>() &&
      number(out.value("output_rows", Json(nullptr))))
    rate = out["output_rows"].get<double>() / old.get<double>();
  if (kind.find("PROJECTION") != std::string::npos || kind == "CROSS_PRODUCT" ||
      kind == "ORDER_BY" || kind == "CTE_SCAN")
    rate = 1.;
  if ((out.contains("selectivity_fusion") ||
       (out.value("survival_estimated", false) &&
        !out.value("survival_estimation_basis", std::string()).empty())) &&
      number(out.value("estimated_selectivity", Json(nullptr))))
    rate = out.at("estimated_selectivity");
  const auto duckdb = out.value("duckdb_selectivity",
                                op.value("duckdb_selectivity", Json(nullptr)));
  if (kind.rfind("SEMANTIC", 0) != 0 && rate.is_null() && number(duckdb) &&
      (kind.find("FILTER") != std::string::npos ||
       kind.find("JOIN") != std::string::npos)) {
    rate = duckdb;
    out["duckdb_selectivity"] = duckdb;
    out["estimated_selectivity"] = rate;
    out["survival_estimated"] = true;
    out["selectivity_fusion"] = {{"weights", {{"duckdb", 1.}}},
                                 {"repair", true}};
    out["survival_estimation_basis"] =
        "weak DuckDB selectivity prior; not observed";
  }
  out["input_rows"] = incoming;
  out["input_side_rows"] = sizes;
  if (old != Json(incoming))
    for (auto &key : {"output_rows", "relational_cost", "relational_seconds"})
      out.erase(key);
  if (op.value("join_type", std::string()) == "MARK" && sizes.size() == 2) {
    out["output_rows"] = sizes[0];
    out["selectivity"] =
        incoming ? Json(sizes[0].get<double>() / incoming) : Json(nullptr);
    out["cardinality_basis"] = "MARK join preserves each left input row";
  } else if (op.contains("sampling_passthrough_input")) {
    const auto side = op.at("sampling_passthrough_input").get<std::size_t>();
    if (side >= sizes.size())
      throw std::invalid_argument("Invalid materialized CTE output boundary");
    out["output_rows"] = sizes[side];
    out["selectivity"] =
        incoming ? Json(sizes[side].get<double>() / incoming) : Json(nullptr);
    out["cardinality_basis"] = "materialized CTE consumer output";
  } else if (op.value("scalar_aggregate", false)) {
    out["output_rows"] = 1.;
    out["selectivity"] = incoming ? Json(1 / incoming) : Json(nullptr);
  } else if (kind.find("LIMIT") != std::string::npos &&
             number(op.value("limit", Json(nullptr)))) {
    double n = std::min(std::max(0., incoming - op.value("offset", 0.)),
                        op["limit"].get<double>());
    out["output_rows"] = n;
    out["selectivity"] = incoming ? Json(n / incoming) : Json(nullptr);
  } else if (kind.find("AGGREGATE") != std::string::npos && rate.is_null() &&
             number(op.value("estimated_rows", Json(nullptr))) &&
             op.at("estimated_rows").get<double>() > 0) {
    // A logical group count is a coarse prior, not a conditional rate. In
    // particular an expected fractional input must not produce many groups.
    const double n = std::min(incoming, op.at("estimated_rows").get<double>());
    out["output_rows"] = n;
    out["selectivity"] = incoming ? Json(n / incoming) : Json(nullptr);
    out["group_count_estimated"] = true;
    out["cardinality_basis"] =
        "current upstream rows capped by DuckDB logical group-count prior";
  } else if (uncertain && rate.is_null()) {
    for (auto &key : {"output_rows", "relational_cost", "relational_seconds"})
      out.erase(key);
    out["selectivity"] = nullptr;
  } else if (!incoming)
    out["output_rows"] = 0.;
  else if (number(rate)) {
    if (!out.contains("selectivity_fusion"))
      out["selectivity"] = rate;
    out["output_rows"] = incoming * rate.get<double>();
  }
  if (kind.find("JOIN") != std::string::npos &&
      op.value("join_type", std::string()) == "LEFT" && sizes.size() == 2) {
    // Null extension preserves the left input even when the right is empty.
    Json output = out.value("output_rows", Json(nullptr));
    if (sizes[0] == 0 || sizes[1] == 0)
      output = sizes[0];
    else if (number(output))
      output = std::max(sizes[0].get<double>(), output.get<double>());
    if (number(output)) {
      out["output_rows"] = output;
      out["selectivity"] =
          incoming ? Json(output.get<double>() / incoming) : Json(nullptr);
    }
  }
  if (!out.contains("cardinality_basis"))
    out["cardinality_basis"] = "upstream estimates × local selectivity";
}
bool RequestLedger::charge(std::size_t requests, bool accuracy) {
  // Accuracy admission is bounded by per-operator sampling batches. Keep its
  // physical HTTP accounting independent from the ordinary selection budget.
  if (!accuracy && requests > remaining())
    return false;
  (accuracy ? accuracy_used_ : selection_used_) += requests;
  return true;
}
void RequestLedger::publish(const std::string &key, bool success,
                            const Json &value) {
  values_.insert_or_assign(key, RequestResult{success, value});
}
const RequestResult *RequestLedger::find(const std::string &key) const {
  auto it = values_.find(key);
  return it == values_.end() ? nullptr : &it->second;
}
}
