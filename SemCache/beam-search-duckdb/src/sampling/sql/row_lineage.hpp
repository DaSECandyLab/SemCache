#pragma once
#include "../common/inputs.hpp"
#include <algorithm>
#include <unordered_map>

namespace semcache::sampling {
// One SQL call borrows immutable current rows. Normalize their origin keys once
// rather than once per match; this directory never survives the call.
class SQLRowLineages {
  struct Entry {
    std::string key;
    const Json *value;
  };
  std::unordered_map<const Json *, std::vector<Entry>> rows_;
  std::size_t &measurements_;

  const std::vector<Entry> &get(const Json &row) {
    if (auto found = rows_.find(&row); found != rows_.end())
      return found->second;
    std::vector<Entry> entries;
    for (const auto &origin : row.at("lineage")) {
      ++measurements_;
      entries.push_back({canonical(origin), &origin});
    }
    std::stable_sort(
        entries.begin(), entries.end(),
        [](const Entry &a, const Entry &b) { return a.key < b.key; });
    entries.erase(std::unique(entries.begin(), entries.end(),
                              [](const Entry &a, const Entry &b) {
                                return a.key == b.key;
                              }),
                  entries.end());
    return rows_.emplace(&row, std::move(entries)).first->second;
  }

public:
  explicit SQLRowLineages(std::size_t &measurements)
      : measurements_(measurements) {}

  Json pair(const Json &left, const Json &right) {
    const auto &a = get(left), &b = get(right);
    auto i = a.begin(), j = b.begin();
    Json result = Json::array();
    result.get_ref<Json::array_t &>().reserve(a.size() + b.size());
    while (i != a.end() || j != b.end()) {
      if (j == b.end() || (i != a.end() && i->key < j->key))
        result.push_back(*i++->value);
      else if (i == a.end() || j->key < i->key)
        result.push_back(*j++->value);
      else {
        result.push_back(*i++->value); // First source wins equal keys.
        ++j;
      }
    }
    return result;
  }

  Json many(const std::vector<const Json *> &sources) {
    std::vector<const Entry *> entries;
    for (const auto *row : sources)
      for (const auto &entry : get(*row))
        entries.push_back(&entry);
    std::stable_sort(
        entries.begin(), entries.end(),
        [](const Entry *a, const Entry *b) { return a->key < b->key; });
    Json result = Json::array();
    result.get_ref<Json::array_t &>().reserve(entries.size());
    const Entry *previous = nullptr;
    for (const auto *entry : entries) {
      if (!previous || previous->key != entry->key)
        result.push_back(*entry->value);
      previous = entry;
    }
    return result;
  }
};
}
