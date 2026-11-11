#pragma once
#include "../../search/state_core.hpp"
#include "operators.hpp"
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace semcache::sampling {
// Cost-only mirror of planning.cost.physical_work. Never changes cardinalities
// or planner legality. Unsupported bound SQL deliberately keeps ordinary costs.
struct PhysicalWork {
  std::string cost_kind;
  Ids input_nodes;
  Id fusion_owner;
  std::string input_units_basis;
};
namespace physical_work_detail {
using Json = nlohmann::ordered_json;
inline std::string kind(const Json &op) {
  auto result = op.value("kind", op.value("type", std::string()));
  for (auto &c : result)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return result;
}
inline std::string sql(const Json &op) {
  return op.value("sql", op.value("sampling_sql", std::string()));
}
inline bool trusted_filter(const Json &op) {
  return kind(op) == "FILTER" && !op.value("semantic", false) &&
         op.value("sampling_row_local", false) &&
         op.contains("sampling_required_fields") &&
         op.at("sampling_required_fields").is_array() &&
         op.value("sampling_row_local_sql", std::string()) == sql(op);
}
struct Parser {
  const std::string &text;
  std::size_t pos = 0;
  void space() {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos])))
      ++pos;
  }
  bool token(const std::string &value) {
    space();
    if (text.compare(pos, value.size(), value) != 0)
      return false;
    pos += value.size();
    return true;
  }
  bool quoted(std::string &value) {
    if (!token("\""))
      return false;
    value.clear();
    while (pos < text.size()) {
      const char c = text[pos++];
      if (c != '"') {
        value += c;
      } else if (pos < text.size() && text[pos] == '"') {
        value += '"';
        ++pos;
      } else {
        return true;
      }
    }
    return false;
  }
  bool atom(std::vector<std::pair<std::string, std::string>> &pairs,
            std::size_t depth) {
    if (depth > 128)
      return false;
    if (token("("))
      return expression(pairs, depth + 1) && token(")");
    std::string left, right;
    if (!quoted(left) || !token("=") || !quoted(right))
      return false;
    pairs.emplace_back(std::move(left), std::move(right));
    return true;
  }
  bool expression(std::vector<std::pair<std::string, std::string>> &pairs,
                  std::size_t depth = 0) {
    if (!atom(pairs, depth))
      return false;
    while (token("AND"))
      if (!atom(pairs, depth))
        return false;
    return true;
  }
  bool done() {
    space();
    return pos == text.size();
  }
};
inline std::vector<std::pair<std::string, std::string>>
equalities(const std::string &text) {
  const std::string prefix = "SELECT * FROM $INPUT WHERE ";
  if (text.rfind(prefix, 0) != 0)
    return {};
  Parser parser{text, prefix.size()};
  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parser.expression(pairs) || !parser.done())
    return {};
  return pairs;
}
inline bool identity_projection(const std::string &text,
                                const std::set<std::string> &required) {
  const std::string prefix = "SELECT ", suffix = " FROM $INPUT";
  if (text.rfind(prefix, 0) != 0 ||
      text.size() < prefix.size() + suffix.size() ||
      text.compare(text.size() - suffix.size(), suffix.size(), suffix) != 0)
    return false;
  const auto body =
      text.substr(prefix.size(), text.size() - prefix.size() - suffix.size());
  Parser parser{body};
  std::set<std::string> columns;
  do {
    std::string source, target;
    if (!parser.token("__sample_id")) {
      if (!parser.quoted(source) || !parser.token("AS") ||
          !parser.quoted(target) || source != target)
        return false;
      columns.insert(source);
    }
  } while (parser.token(","));
  if (!parser.done())
    return false;
  for (const auto &field : required)
    if (!columns.count(field))
      return false;
  return true;
}
inline std::string name(const Json &field) {
  if (field.is_string())
    return field.get<std::string>();
  if (field.is_object()) {
    if (field.contains("binding"))
      return field.at("binding").dump();
    return field.value("name", std::string());
  }
  return field.dump();
}
}

inline std::map<Id, PhysicalWork>
physical_work(const std::vector<NodePtr> &nodes,
              const std::map<Id, nlohmann::ordered_json> &operators,
              const std::map<Id, nlohmann::ordered_json> &logical,
              const nlohmann::ordered_json &field_registry) {
  using namespace physical_work_detail;
  std::map<Id, NodePtr> by_id;
  std::map<Id, std::size_t> consumers;
  std::map<Id, Json> descriptions;
  std::map<Id, std::set<std::string>> fields;
  for (const auto &node : nodes) {
    by_id.emplace(node->node_id, node);
    for (const auto child : node->children)
      ++consumers[child];
    descriptions[node->node_id] = node->operator_id
                                      ? operators.at(*node->operator_id)
                                      : logical.at(node->node_id);
    for (const auto id : node->fields)
      fields[node->node_id].insert(name(field_registry.at(id)));
    // Empty schemas cannot certify a two-sided equality.
  }
  std::map<Id, PhysicalWork> result;
  for (const auto &owner : nodes) {
    const auto &op = descriptions.at(owner->node_id);
    const auto pairs = trusted_filter(op)
                           ? equalities(sql(op))
                           : std::vector<std::pair<std::string, std::string>>{};
    if (owner->children.size() != 1 || pairs.empty())
      continue;
    std::set<std::string> required;
    for (const auto &[a, b] : pairs) {
      required.insert(a);
      required.insert(b);
    }
    auto current = by_id.at(owner->children[0]);
    Ids intermediates;
    while (consumers[current->node_id] == 1) {
      const auto &description = descriptions.at(current->node_id);
      const auto label = kind(description);
      if (label == "CROSS_PRODUCT" && current->children.size() == 2) {
        const auto &left = fields[current->children[0]],
                   &right = fields[current->children[1]];
        bool valid = true;
        for (const auto &field : left)
          valid &= !right.count(field);
        for (const auto &[a, b] : pairs)
          valid &= (left.count(a) && right.count(b)) ||
                   (right.count(a) && left.count(b));
        if (!valid || result.count(current->node_id))
          break;
        result.emplace(
            current->node_id,
            PhysicalWork{
                "CROSS_PRODUCT",
                {},
                owner->node_id,
                "fused into certified equijoin; no separate Cartesian work"});
        for (const auto intermediate : intermediates)
          result.emplace(
              intermediate,
              PhysicalWork{kind(descriptions.at(intermediate)),
                           {},
                           owner->node_id,
                           "trusted residual predicate or identity projection "
                           "fused into certified equijoin"});
        result.emplace(owner->node_id,
                       PhysicalWork{"COMPARISON_JOIN", current->children,
                                    owner->node_id,
                                    "sum of certified equijoin child rows"});
        break;
      }
      if (current->children.size() != 1 ||
          !(trusted_filter(description) ||
            (label == "PROJECTION" && !description.value("semantic", false) &&
             identity_projection(sql(description), required))))
        break;
      intermediates.push_back(current->node_id);
      current = by_id.at(current->children[0]);
    }
  }
  return result;
}
}
