#pragma once
#include "../statistics/statistics.hpp"
#include <algorithm>
#include <stdexcept>

namespace semcache::sampling {
// Only the exported, single semantic Join call is rewritten. Argument 0 (the
// task literal, including encoded tasks) and user SQL string contents survive.
inline std::string prepared_join_arguments(std::string sql, const Json &op,
                                           bool two_inputs) {
  if (op.value("kind", std::string()).find("JOIN") == std::string::npos)
    return sql;
  bool generated = false;
  for (const auto &field : op.value("input_field_schema", Json::array()))
    generated |=
        field.value("value_key", std::string()).rfind("__semantic_input:", 0) ==
        0;
  if (!generated)
    return sql;
  auto call = sql.find("semantic_join(");
  if (call == std::string::npos)
    call = sql.find("semantic_join_encoded(");
  if (call == std::string::npos)
    throw std::invalid_argument("Prepared Join requires bound semantic call");
  const auto open = sql.find('(', call);
  std::vector<std::pair<std::size_t, std::size_t>> arguments;
  std::size_t begin = open + 1, depth = 0, lists = 0, structs = 0;
  char quote = 0;
  for (std::size_t p = begin; p < sql.size(); ++p) {
    const auto c = sql[p];
    if (quote) {
      if (c == quote) {
        if (p + 1 < sql.size() && sql[p + 1] == quote)
          ++p;
        else
          quote = 0;
      }
    } else if (c == '\'' || c == '"')
      quote = c;
    else if (c == '(')
      ++depth;
    else if (c == '[')
      ++lists;
    else if (c == ']' && lists)
      --lists;
    else if (c == '{')
      ++structs;
    else if (c == '}' && structs)
      --structs;
    else if (c == ')' && depth)
      --depth;
    else if (c == ',' || c == ')') {
      if (depth || lists || structs)
        continue;
      arguments.emplace_back(begin, p - begin);
      begin = p + 1;
      if (c == ')')
        break;
    }
  }
  auto names = op.value("left_input_fields", Json::array());
  const auto right = op.value("right_input_fields", Json::array());
  for (const auto &name : right)
    names.push_back(name);
  if (arguments.size() != names.size() + 1)
    throw std::invalid_argument("Prepared Join argument arity changed");
  for (std::size_t i = names.size(); i; --i) {
    for (const auto &field : op.at("input_field_schema")) {
      if (field.at("name") != names[i - 1])
        continue;
      auto key = field.value("value_key", std::string());
      if (key.rfind("__semantic_input:", 0) != 0)
        continue;
      std::string quoted = "\"";
      for (auto c : key)
        quoted += c == '"' ? "\"\"" : std::string(1, c);
      quoted += '"';
      if (two_inputs)
        quoted =
            (std::find(right.begin(), right.end(), names[i - 1]) != right.end()
                 ? "r."
                 : "l.") +
            quoted;
      sql.replace(arguments[i].first, arguments[i].second, quoted);
    }
  }
  return sql;
}
}
