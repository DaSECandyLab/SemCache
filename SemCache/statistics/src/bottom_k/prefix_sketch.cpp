// Character-domain coordinated bottom-k construction uses SHA-256 ranks and frequency counts.
#include "mbedtls/sha256.h"
#include "nlohmann/json.hpp"
#include "semcache/prefix_sketch.h"
#include "semcache/prefix_entries.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <thread>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {
using Json = nlohmann::json;
using Hash = std::array<unsigned char, 32>;
thread_local std::string error;
Hash Rank(const std::string &prefix) {
  Hash result;
  if (mbedtls_sha256(reinterpret_cast<const unsigned char *>(prefix.data()),
                     prefix.size(), result.data(), 0))
    throw std::runtime_error("SHA-256 failed");
  return result;
}
// One running digest serves all increasing character-depth anchors of a value.
// Only the new byte interval is hashed; finalization works on a cloned context.
class PrefixHash {
public:
  PrefixHash() {
    mbedtls_sha256_init(&digest);
    if (mbedtls_sha256_starts(&digest, 0)) {
      mbedtls_sha256_free(&digest);
      throw std::runtime_error("SHA-256 initialization failed");
    }
  }
  ~PrefixHash() { mbedtls_sha256_free(&digest); }
  PrefixHash(const PrefixHash &) = delete;
  PrefixHash &operator=(const PrefixHash &) = delete;
  Hash At(const char *data, size_t offset) {
    if (mbedtls_sha256_update(
            &digest, reinterpret_cast<const unsigned char *>(data + previous),
            offset - previous))
      throw std::runtime_error("SHA-256 update failed");
    previous = offset;
    mbedtls_sha256_context copy;
    mbedtls_sha256_init(&copy);
    mbedtls_sha256_clone(&copy, &digest);
    Hash hash;
    const auto status = mbedtls_sha256_finish(&copy, hash.data());
    mbedtls_sha256_free(&copy);
    if (status)
      throw std::runtime_error("SHA-256 finalization failed");
    return hash;
  }

private:
  mbedtls_sha256_context digest;
  size_t previous = 0;
};
void Increment(uint64_t &value, uint64_t amount) {
  if (amount > std::numeric_limits<uint64_t>::max() - value)
    throw std::overflow_error("BottomKSketch counter overflow");
  value += amount;
}
uint64_t Multiply(uint64_t value, uint64_t frequency) {
  if (frequency && value > std::numeric_limits<uint64_t>::max() / frequency)
    throw std::overflow_error("BottomKSketch counter overflow");
  return value * frequency;
}
struct RepeatedBatch {
  std::vector<size_t> representatives;
  std::vector<std::pair<size_t, uint64_t>> groups;
};
RepeatedBatch GroupBatch(const char *const *data, const size_t *sizes,
                         size_t count) {
  RepeatedBatch result;
  // Callback/C ABI buffers live for this batch, so deduplication needs no copy.
  std::unordered_map<std::string_view, size_t> seen;
  seen.reserve(count);
  result.representatives.reserve(count);
  result.groups.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!data[i] && sizes[i])
      throw std::invalid_argument("Null string pointer");
    auto found =
        seen.emplace(std::string_view(data[i] ? data[i] : "", sizes[i]),
                     result.groups.size());
    if (found.second)
      result.groups.emplace_back(i, 0);
    auto &group = result.groups[found.first->second];
    ++group.second;
    result.representatives.push_back(group.first);
  }
  return result;
}
using RankedRow = std::pair<std::uint64_t, std::size_t>;
using RowSample = std::vector<RankedRow>;
// A separate, bijective row-ordinal permutation coordinates row sampling across
// fields/depths without reusing content SHA ranks or allocating an N-row bank.
std::uint64_t RowRank(std::size_t row) {
  auto x = static_cast<std::uint64_t>(row) + 0x9e3779b97f4a7c15ULL + 104729;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
void KeepRow(RowSample &sample, const RankedRow &row, std::size_t limit) {
  if (!limit)
    return;
  if (sample.size() == limit) {
    if (row >= sample.front())
      return;
    std::pop_heap(sample.begin(), sample.end());
    sample.pop_back();
  }
  sample.push_back(row);
  std::push_heap(sample.begin(), sample.end());
}
struct BottomKSketch {
  struct Count {
    std::uint64_t frequency;
    RowSample rows;
    Hash hash;
  };
  struct Entry {
    Hash hash;
    const std::string *prefix;
    bool operator<(const Entry &other) const {
      return hash < other.hash ||
             (hash == other.hash && *prefix < *other.prefix);
    }
  };
  explicit BottomKSketch(size_t capacity, size_t row_limit = 0)
      : capacity(capacity), row_limit(row_limit) {}
  BottomKSketch(const BottomKSketch &) = delete;
  BottomKSketch &operator=(const BottomKSketch &) = delete;
  BottomKSketch(BottomKSketch &&) noexcept = default;
  BottomKSketch &operator=(BottomKSketch &&) noexcept = default;
  size_t capacity, row_limit;
  bool complete = true;
  std::unordered_map<std::string, Count> counts;
  // unordered_map keeps key addresses stable across rehashes. The heap borrows
  // those keys so each retained prefix owns only one string, at every depth.
  std::priority_queue<Entry> heap;
  void AddHashed(const char *data, size_t bytes, const Hash &hash,
                 uint64_t frequency = 1, const RowSample *rows = nullptr) {
    // A hash above the current threshold cannot belong to a retained prefix.
    // Reject it before allocating a string or hashing its bytes a second time.
    if (counts.size() == capacity && hash > heap.top().hash) {
      complete = false;
      return;
    }
    std::string prefix(data, bytes);
    auto found = counts.find(prefix);
    if (found != counts.end()) {
      Increment(found->second.frequency, frequency);
      if (rows)
        for (const auto &row : *rows)
          KeepRow(found->second.rows, row, row_limit);
      return;
    }
    // 最大堆顶是当前最差的保留项；阈值只会下降，已淘汰前缀不会重新入选，
    // 因此最终保留项的频次不需要为历史淘汰项回放输入。
    if (counts.size() == capacity) {
      complete = false;
      if (hash == heap.top().hash && prefix >= *heap.top().prefix)
        return;
      const auto *evicted = heap.top().prefix;
      heap.pop();
      counts.erase(*evicted);
    }
    auto inserted = counts.emplace(std::move(prefix), Count{frequency, {}, hash});
    if (rows)
      for (const auto &row : *rows)
        KeepRow(inserted.first->second.rows, row, row_limit);
    heap.push({hash, &inserted.first->first});
  }
  Json Serialize() const {
    Json entries = Json::object(), prefixes = Json::object();
    for (const auto &entry : counts) {
      const auto hash = semcache::statistics::encode_prefix_hash(entry.second.hash);
      if (entries.contains(hash))
        throw std::invalid_argument("Duplicate prefix hash");
      entries[hash] = entry.second.frequency;
      // Only the offline cooccurrence reservoir needs text for starts-with.
      // Request-history and ordinary field sketches retain hashes/counts only.
      if (row_limit)
        prefixes[hash] = entry.first;
    }
    Json result = {{"capacity", capacity},
                   {"complete", complete},
                   {"entry_encoding", semcache::statistics::prefix_entry_encoding},
                   {"entries", std::move(entries)}};
    if (row_limit)
      result["prefixes"] = std::move(prefixes);
    return result;
  }
};
// One UTF-8 walker serves both pilot tuning and full-stream construction.
// Validate the entire value, including characters beyond the tracked ceiling.
template <class Visitor>
size_t VisitCharacters(const char *data, size_t bytes, Visitor visit) {
  if (!data && bytes)
    throw std::invalid_argument("Null string pointer");
  // Validate UTF-8 once and visit anchors at Unicode code-point boundaries,
  // matching Python len/slicing, including supplementary characters/NUL.
  size_t offset = 0, characters = 0;
  while (offset < bytes) {
    const auto first = static_cast<unsigned char>(data[offset]);
    const size_t length = first < 0x80                     ? 1
                          : first >= 0xC2 && first <= 0xDF ? 2
                          : first >= 0xE0 && first <= 0xEF ? 3
                          : first >= 0xF0 && first <= 0xF4 ? 4
                                                           : 0;
    if (!length || length > bytes - offset)
      throw std::invalid_argument("Invalid UTF-8");
    uint32_t code = first & (length == 1   ? 0x7F
                             : length == 2 ? 0x1F
                             : length == 3 ? 0x0F
                                           : 0x07);
    for (size_t j = 1; j < length; ++j) {
      auto byte = static_cast<unsigned char>(data[offset + j]);
      if ((byte & 0xC0) != 0x80)
        throw std::invalid_argument("Invalid UTF-8 continuation");
      code = (code << 6) | (byte & 0x3F);
    }
    if ((length == 2 && code < 0x80) || (length == 3 && code < 0x800) ||
        (length == 4 && code < 0x10000) || code > 0x10FFFF ||
        (code >= 0xD800 && code <= 0xDFFF))
      throw std::invalid_argument("Noncanonical UTF-8");
    offset += length;
    ++characters;
    visit(characters, offset);
  }
  return characters;
}
struct PrefixSummary {
  PrefixSummary(size_t capacity, size_t maximum, const size_t *depths,
                size_t count, size_t row_limit = 0)
      : capacity(capacity), maximum(maximum) {
    if (capacity < 2 || !maximum || !count || !depths || depths[0] != 1 ||
        depths[count - 1] != maximum)
      throw std::invalid_argument("Invalid sketch capacity/anchors");
    sketches.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (i && depths[i] <= depths[i - 1])
        throw std::invalid_argument("Anchors must strictly increase");
      sketches.emplace_back(depths[i], BottomKSketch(capacity, row_limit));
    }
  }
  size_t capacity, maximum;
  uint64_t count = 0, total_length = 0, truncated = 0;
  std::vector<std::pair<size_t, BottomKSketch>> sketches;
  std::string serialized;
  bool failed = false;
  size_t Add(const char *data, size_t bytes, uint64_t frequency = 1,
             const RowSample *rows = nullptr) {
    if (!frequency)
      throw std::invalid_argument("Frequency must be positive");
    auto anchor = sketches.begin();
    PrefixHash digest;
    const auto characters =
        VisitCharacters(data, bytes, [&](size_t count, size_t offset) {
          if (anchor != sketches.end() && anchor->first == count) {
            anchor->second.AddHashed(data, offset, digest.At(data, offset),
                                     frequency, rows);
            ++anchor;
          }
        });
    Increment(count, frequency);
    Increment(total_length, Multiply(std::min(characters, maximum), frequency));
    Increment(truncated, characters > maximum ? frequency : 0);
    return characters;
  }
  Json Snapshot() const {
    Json depths = Json::array(), values = Json::object();
    for (const auto &entry : sketches) {
      depths.push_back(entry.first);
      values[std::to_string(entry.first)] = entry.second.Serialize();
    }
    return Json({{"metadata",
                  {{"hash_version", "sha256-prefix-v1"},
                   {"serializer", "prompt-v4-json-unicode"},
                   {"domain", "characters"},
                   {"capacity", capacity},
                   {"max_length", maximum},
                   {"anchors", depths}}},
                 {"count", count},
                 {"total_length", total_length},
                 {"truncated", truncated},
                 {"sketches", values}});
  }
  const char *Serialize() {
    serialized = Snapshot().dump();
    return serialized.c_str();
  }
};
#include "adaptive_sketch.hpp"
}

// C ABI 不允许异常跨越 ctypes 边界；失败返回值配合线程局部错误文本传递原因。
extern "C" {
int sem_prefix_version() { return 1; }
const char *sem_prefix_error() { return error.c_str(); }
void *sem_adaptive_build(const char *config, const char *names,
                         const size_t *counts, size_t fields,
                         AdaptiveFetch fetch, void *context) {
  try {
    if (!config || !names || !fetch || (fields && !counts))
      throw std::invalid_argument("Invalid adaptive build arguments");
    return new std::string(BuildAdaptive(Json::parse(config),
                                         Json::parse(names), counts, fields,
                                         fetch, context)
                               .dump());
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
const char *sem_adaptive_json(void *result) {
  return result ? static_cast<std::string *>(result)->c_str() : nullptr;
}
void *sem_table_build(const char *config, const char *names,
                      const size_t *counts, size_t fields, AdaptiveFetch fetch,
                      void *context) {
  try {
    if (!config || !names || !fetch || (fields && !counts))
      throw std::invalid_argument("Invalid offline table build arguments");
    return new std::string(BuildAdaptive(Json::parse(config),
                                         Json::parse(names), counts, fields,
                                         fetch, context, true)
                               .dump());
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void sem_adaptive_destroy(void *result) {
  delete static_cast<std::string *>(result);
}
void *sem_prefix_create(size_t capacity, size_t maximum, const size_t *anchors,
                        size_t size) {
  try {
    return new PrefixSummary(capacity, maximum, anchors, size);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
int sem_prefix_add(void *handle, const char *const *strings,
                   const size_t *sizes, size_t count) {
  auto summary = static_cast<PrefixSummary *>(handle);
  try {
    if (!summary || summary->failed || (count && (!strings || !sizes)))
      throw std::invalid_argument("Invalid or failed native summary");
    for (size_t start = 0; start < count;) {
      const size_t size = std::min<size_t>(1024, count - start);
      auto repeated = GroupBatch(strings + start, sizes + start, size);
      for (const auto &group : repeated.groups)
        summary->Add(strings[start + group.first], sizes[start + group.first],
                     group.second);
      start += size;
    }
    return 0;
  } catch (const std::exception &e) {
    if (summary)
      summary->failed = true; // Never export a partially applied batch.
    error = e.what();
    return -1;
  }
}
const char *sem_prefix_json(void *handle) {
  try {
    auto summary = static_cast<PrefixSummary *>(handle);
    if (!summary || summary->failed)
      throw std::invalid_argument("Invalid or failed native summary");
    return summary->Serialize();
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void sem_prefix_destroy(void *handle) {
  delete static_cast<PrefixSummary *>(handle);
}
}
