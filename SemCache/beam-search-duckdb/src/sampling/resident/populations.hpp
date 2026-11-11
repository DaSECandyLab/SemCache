#pragma once
#include "../common/row_metadata.hpp"
#include "../statistics/statistics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace semcache::sampling::resident_detail {
using Counts = std::map<std::pair<std::string, std::size_t>, std::size_t>;
using Origins = std::map<std::string, std::set<std::string>>;
inline bool product(const Json &rows) {
  return rows.is_object() && rows.contains("product");
}
inline std::size_t row_count(const Json &rows) {
  if (!product(rows))
    return rows.size();
  auto a = row_count(rows["product"][0]), b = row_count(rows["product"][1]);
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("Cartesian population overflow");
  return a * b;
}
inline std::string relkey(const Json &rel) {
  return rel.is_string() ? rel.get<std::string>() : rel.dump();
}
inline std::string node_relkey(const std::string &rel) {
  return relkey(Json::parse(rel));
}
inline Json combine(const Json &left, const Json &right,
                    bool with_values = true) {
  Json values = with_values ? left.at("values") : Json::object();
  if (with_values)
    values.update(right.at("values"));
  std::map<std::string, Json> origins;
  for (auto *r : {&left, &right})
    for (auto &p : r->at("lineage"))
      origins[canonical(p)] = p;
  Json lineage = Json::array();
  for (auto &p : origins)
    lineage.push_back(p.second);
  Json result = {{"id", digest(Json::array({left.at("id"), right.at("id")}))},
                 {"values", std::move(values)},
                 {"lineage", std::move(lineage)}};
  if (left.contains("_sample_weight_factor") ||
      right.contains("_sample_weight_factor"))
    sample_factor(result, sample_factor(left) * sample_factor(right));
  merge_sample_origins(result, left);
  merge_sample_origins(result, right);
  return result;
}
inline Json row_at(const Json &rows, std::size_t index) {
  if (!product(rows))
    return rows.at(index);
  auto &right = rows.at("product").at(1);
  auto n = row_count(right);
  return combine(row_at(rows["product"][0], index / n),
                 row_at(right, index % n));
}
// Probe payloads need task/consumer cells and all source evidence, not every
// unused field in a Cartesian context. Keep the same recursive ID/lineage and
// weight operations as row_at; successful rows are restored from coordinates.
inline Json projected_row_at(const Json &rows, std::size_t index,
                             const std::set<std::string> &fields) {
  if (product(rows)) {
    const auto &parts = rows.at("product");
    const auto n = row_count(parts[1]);
    return combine(projected_row_at(parts[0], index / n, fields),
                   projected_row_at(parts[1], index % n, fields));
  }
  const auto &source = rows.at(index);
  Json result = Json::object();
  for (auto it = source.begin(); it != source.end(); ++it)
    if (it.key() == "values") {
      Json values = Json::object();
      for (auto cell = it.value().begin(); cell != it.value().end(); ++cell)
        if (fields.count(cell.key()))
          values[cell.key()] = cell.value();
      result["values"] = std::move(values);
    } else
      result[it.key()] = it.value();
  return result;
}
// Look up only a requested cell in an exact product. Right-side replacement
// matches combine(); null cells remain distinguishable from missing fields.
inline const Json *find_value(const Json &rows, std::size_t index,
                              const std::string &name) {
  if (!product(rows)) {
    const auto &values = rows.at(index).at("values");
    auto found = values.find(name);
    return found == values.end() ? nullptr : &found.value();
  }
  const auto &parts = rows.at("product");
  const auto count = row_count(parts[1]);
  if (auto found = find_value(parts[1], index % count, name))
    return found;
  return find_value(parts[0], index / count, name);
}

inline std::set<std::string> columns(const Json &rows) {
  std::set<std::string> out;
  if (product(rows)) {
    for (auto &r : rows["product"]) {
      auto c = columns(r);
      out.insert(c.begin(), c.end());
    }
  } else
    for (auto &row : rows)
      for (auto it = row.at("values").begin(); it != row.at("values").end();
           ++it)
        out.insert(it.key());
  return out;
}
inline Origins origins(const Json &rows) {
  Origins result;
  if (!row_count(rows))
    return result;
  if (product(rows)) {
    for (auto &r : rows["product"])
      for (auto &entry : origins(r))
        result[entry.first].insert(entry.second.begin(), entry.second.end());
  } else
    for (auto &row : rows)
      for (auto &pair : row.at("lineage"))
        result[relkey(pair[0])].insert(pair[1].get<std::string>());
  return result;
}
inline std::optional<std::vector<int>>
leaf_for(const Json &rows, const std::set<std::string> &keys) {
  if (!product(rows)) {
    auto c = columns(rows);
    if (std::includes(c.begin(), c.end(), keys.begin(), keys.end()))
      return std::vector<int>{};
    return {};
  }
  for (int i = 0; i < 2; ++i)
    if (auto p = leaf_for(rows["product"][i], keys)) {
      p->insert(p->begin(), i);
      return p;
    }
  return {};
}
inline const Json &leaf_at(const Json &rows, const std::vector<int> &path) {
  auto *p = &rows;
  for (int side : path)
    p = &p->at("product").at(side);
  return *p;
}
inline Json replace_leaf(const Json &rows, const std::vector<int> &path,
                         const Json &replacement, std::size_t depth = 0) {
  if (depth == path.size())
    return replacement;
  Json out = rows;
  out["product"][path[depth]] =
      replace_leaf(rows["product"][path[depth]], path, replacement, depth + 1);
  return out;
}
inline Json representative(const Json &rows, const std::vector<int> &path,
                           const Json &row, std::size_t depth = 0) {
  if (depth == path.size())
    return row;
  auto &parts = rows.at("product");
  return path[depth] == 0
             ? combine(representative(parts[0], path, row, depth + 1),
                       row_at(parts[1], 0))
             : combine(row_at(parts[0], 0),
                       representative(parts[1], path, row, depth + 1));
}
inline std::set<std::string> input_keys(const Json &op) {
  Json descriptors = Json::object();
  for (auto &d : op.value("input_field_schema", Json::array()))
    descriptors[d.at("name").get<std::string>()] = d;
  Json names = op.value("input_fields", Json::array());
  if (names.empty()) {
    names = op.value("left_input_fields", Json::array());
    for (auto &f : op.value("right_input_fields", Json::array()))
      names.push_back(f);
  }
  std::set<std::string> out;
  for (auto &f : names) {
    std::string name = f;
    out.insert(descriptors.contains(name)
                   ? descriptors[name].value("value_key", name)
                   : name);
  }
  return out;
}
inline bool independent_leaf_identities(const Json &rows) {
  if (product(rows))
    return independent_leaf_identities(rows.at("product")[0]) &&
           independent_leaf_identities(rows.at("product")[1]);
  std::set<std::vector<std::string>> seen;
  for (auto &row : rows) {
    std::vector<std::string> origin;
    for (auto &pair : row.at("lineage"))
      origin.push_back(canonical(pair));
    std::sort(origin.begin(), origin.end());
    if (!seen.insert(std::move(origin)).second)
      return false;
  }
  return true;
}
inline std::string sample_column(const Json &f) {
  if (f.is_string())
    return f.get<std::string>();
  if (!f.is_object())
    return canonical(f);
  if (f.contains("name"))
    return f.at("name").get<std::string>();
  if (f.contains("binding"))
    return canonical(f.at("binding"));
  if (f.contains("identity"))
    return canonical(f.at("identity"));
  if (f.contains("expression"))
    return canonical(f.at("expression"));
  return canonical(f);
}
inline std::string field_key(const Json &f) {
  if (!f.is_object())
    return canonical(f);
  for (auto *name : {"binding", "identity", "name", "expression"})
    if (f.contains(name))
      return canonical(f.at(name));
  return canonical(f);
}
inline std::string output_column(const Json &op) {
  if (op.value("kind", std::string()) != "SEMANTIC_PROJECTION")
    return {};
  std::string sql = op.value("sql", op.value("sampling_sql", std::string()));
  auto as = sql.rfind(" AS ");
  if (as == std::string::npos)
    throw std::invalid_argument(
        "Semantic projection requires bound output alias");
  std::string field = sql.substr(as + 4);
  field = field.substr(0, field.find(" FROM "));
  auto first = field.find_first_not_of(" \t\n"),
       last = field.find_last_not_of(" \t\n");
  field = field.substr(first, last - first + 1);
  if (field.size() > 1 && field.front() == '"' && field.back() == '"') {
    field = field.substr(1, field.size() - 2);
    for (std::size_t pos = 0;
         (pos = field.find("\"\"", pos)) != std::string::npos;)
      field.replace(pos, 2, "\"");
  }
  return field;
}
inline Json restore_row(const Json &op, Json row, const RequestResult &result) {
  if (!result.success)
    return nullptr;
  for (auto it = row["values"].begin(); it != row["values"].end();)
    if (it.key().rfind("__semantic_input:", 0) == 0)
      it = row["values"].erase(it);
    else
      ++it;
  auto name = output_column(op);
  if (!name.empty())
    row["values"][name] = result.value;
  return row;
}
inline Json restore(const Json &op, const std::vector<Json> &unit,
                    const RequestResult &result) {
  if (!result.success)
    return nullptr;
  return restore_row(op, unit.size() == 1 ? unit[0] : combine(unit[0], unit[1]),
                     result);
}
inline std::string task_key(const Json &op) {
  Json schema = Json::array();
  for (auto &f : op.value("input_field_schema", Json::array()))
    schema.push_back(
        Json::array({f.at("name"), f.value("type", Json(nullptr))}));
  return digest(Json::array({op.at("kind"), op.at("task"), schema,
                             op.value("output_schema", Json::array())}));
}
inline bool finite_number(const Json &v) {
  return v.is_number() && std::isfinite(v.get<double>()) &&
         v.get<double>() >= 0;
}
}
