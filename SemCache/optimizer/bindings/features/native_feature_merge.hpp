#pragma once

// CPU-only batch pieces. Builders own disjoint, contiguous ranges of complete
// inference chunks; this stable reduction restores the serial first-occurrence
// order before any Python objects, tensors or model activations are created.
#include "sampling/environment.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace semcache::native_features {
using Json = sampling::Json;
inline constexpr std::size_t kNodeNumericWidth = 105;
inline constexpr std::size_t kHistoryWidth = 18;

struct Row {
  Json value;
  std::array<double, kNodeNumericWidth> numeric{};
  std::array<double, kHistoryWidth> history{};
  bool has_history = false;
  // Ordered canonical JSON of fields/description/first_fields only. Local
  // template ordinals cannot cross a shard boundary.
  std::string template_key;
};
struct Link {
  std::size_t source = 0; // Position within its graph, never a unique-row ID.
  std::array<double, 8> relation{};
  double reuse = 0.;
  bool present = false;
};
struct Graph {
  std::vector<std::int64_t> rows;
  std::vector<Link> links;
  std::vector<std::size_t> link_offsets{0};
  std::vector<bool> roots;
  std::size_t work = 0;
  double size = 0.;
};
struct Description {
  Id nid;
  Json value;
};
struct Shard {
  std::vector<Row> rows;
  std::vector<Graph> graphs;
  // A remaining operator has the same immutable input in every candidate of
  // this scoring call. Groups retain its ID instead of copying its tokens and
  // 214 numeric coordinates for every remaining-set combination.
  std::vector<std::vector<Id>> work;
  std::unordered_map<Id, Json> work_operators;
  std::vector<std::string> work_keys;
  std::vector<std::vector<Description>> descriptions;
  std::vector<std::size_t> offsets{0};
  std::size_t max_nodes = 1, max_degree = 1;
};

inline std::string row_key(std::size_t ordinal, const Row &row) {
  // Keep the exact serial key, including the separate absent-history marker
  // and the original double bit patterns. Inference chunk grouping depends on
  // which rows are first seen, even when their eventual float32 values agree.
  std::string key = std::to_string(ordinal);
  key.append(reinterpret_cast<const char *>(row.numeric.data()),
             sizeof(row.numeric));
  key.push_back(row.has_history ? '1' : '0');
  key.append(reinterpret_cast<const char *>(row.history.data()),
             sizeof(row.history));
  return key;
}

// Destructively consume shards in their input-range order. The caller must not
// pass ranges in worker-completion order: that would change model batch sizes.
inline Shard merge_shards(std::vector<Shard> &shards) {
  Shard result;
  std::size_t graph_count = 0, chunk_count = 0;
  for (const auto &shard : shards) {
    if (shard.work.size() != shard.work_keys.size() || shard.offsets.empty() ||
        shard.offsets.front() != 0 ||
        shard.offsets.back() != shard.rows.size() ||
        shard.descriptions.size() + 1 != shard.offsets.size() ||
        !std::is_sorted(shard.offsets.begin(), shard.offsets.end()))
      throw std::invalid_argument("Inconsistent native feature shard metadata");
    graph_count += shard.graphs.size();
    chunk_count += shard.descriptions.size();
  }
  result.graphs.reserve(graph_count);
  result.descriptions.reserve(chunk_count);
  result.offsets.reserve(chunk_count + 1);
  std::unordered_map<std::string, std::size_t> templates, unique_rows,
      unique_work;
  std::unordered_set<Id> described;

  for (auto &shard : shards) {
    result.max_nodes = std::max(result.max_nodes, shard.max_nodes);
    result.max_degree = std::max(result.max_degree, shard.max_degree);
    for (auto &entry : shard.work_operators)
      result.work_operators.try_emplace(entry.first, std::move(entry.second));
    std::vector<std::size_t> rows(shard.rows.size());
    // prefix[n] is the global row count after the first n local unique rows.
    // A later shard can discover that every one of those rows already exists.
    std::vector<std::size_t> prefix(shard.rows.size() + 1, result.rows.size());
    for (std::size_t i = 0; i < shard.rows.size(); ++i) {
      auto &row = shard.rows[i];
      const auto template_id =
          templates.emplace(row.template_key, templates.size()).first->second;
      auto key = row_key(template_id, row);
      const auto inserted =
          unique_rows.emplace(std::move(key), result.rows.size());
      rows[i] = inserted.first->second;
      if (inserted.second)
        result.rows.push_back(std::move(row));
      prefix[i + 1] = result.rows.size();
    }
    std::vector<std::size_t> work(shard.work.size());
    for (std::size_t i = 0; i < shard.work.size(); ++i) {
      const auto inserted =
          unique_work.emplace(shard.work_keys[i], result.work.size());
      work[i] = inserted.first->second;
      if (inserted.second) {
        result.work.push_back(std::move(shard.work[i]));
        result.work_keys.push_back(std::move(shard.work_keys[i]));
      }
    }
    for (auto &graph : shard.graphs) {
      if (graph.work >= work.size() ||
          graph.roots.size() != graph.rows.size() ||
          graph.link_offsets.size() != graph.rows.size() + 1 ||
          graph.link_offsets.front() != 0 ||
          graph.link_offsets.back() != graph.links.size())
        throw std::invalid_argument("Inconsistent native feature shard graph");
      for (auto &row : graph.rows) {
        if (row < 0 || static_cast<std::size_t>(row) >= rows.size())
          throw std::invalid_argument(
              "Native feature row ID outside its shard");
        row = static_cast<std::int64_t>(rows[static_cast<std::size_t>(row)]);
      }
      graph.work = work[graph.work];
      result.max_nodes = std::max(result.max_nodes, graph.rows.size());
      result.graphs.push_back(std::move(graph));
    }
    for (auto &chunk : shard.descriptions) {
      std::vector<Description> values;
      for (auto &description : chunk)
        if (described.insert(description.nid).second)
          values.push_back(std::move(description));
      result.descriptions.push_back(std::move(values));
    }
    for (std::size_t i = 1; i < shard.offsets.size(); ++i) {
      const auto offset = shard.offsets[i];
      if (offset >= prefix.size())
        throw std::invalid_argument(
            "Native feature chunk offset outside shard");
      result.offsets.push_back(prefix[offset]);
    }
    shard = Shard{};
  }
  return result;
}
}
