#pragma once
#include <set>
#include <string>

namespace semcache::sampling {
// Sampling expressions come from the bound SQL exporter. String literals do
// not introduce dependencies; quoted identifiers retain their exact binding.
inline std::set<std::string>
expression_bindings(const std::string &expression) {
  std::set<std::string> result;
  for (std::size_t i = 0; i < expression.size(); ++i) {
    const auto quote = expression[i];
    if (quote != '\'' && quote != '"')
      continue;
    std::string value;
    for (++i; i < expression.size(); ++i) {
      if (expression[i] == quote) {
        if (i + 1 < expression.size() && expression[i + 1] == quote) {
          value += quote;
          ++i;
        } else
          break;
      } else
        value += expression[i];
    }
    if (quote == '"')
      result.insert(std::move(value));
  }
  return result;
}
}
