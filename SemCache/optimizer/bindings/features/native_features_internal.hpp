#pragma once
#include "native_features.hpp"
#include "native_feature_merge.hpp"
#include "../bridge.hpp"
#include "sampling/profile/environment.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace semcache::python_features_detail {
namespace py = pybind11;
using semcache::Id;
using semcache::StatePtr;
using semcache::native_features::Graph;
using semcache::native_features::Link;
using semcache::native_features::merge_shards;
using semcache::native_features::Row;
using semcache::native_features::Shard;
using semcache::sampling::from_python;
using semcache::sampling::Json;
using semcache::sampling::ProfileEnvironment;
using semcache::sampling::StatisticView;
using semcache::sampling::to_python;
constexpr std::size_t numeric_width = 105, history_width = 18;

const Json &get(const Json &object, const char *key);
// Field values remain owned by immutable native records. Only calibration and
// scan defaults add values; no per-candidate statistic dictionary is copied.
struct StatOverlay {
  StatisticView source{};
  Json overlay = Json::object();
  std::map<std::string, const Json *> defaults;
  explicit StatOverlay(StatisticView view = {}) : source(view) {}
  const Json &read(const std::string &key) const {
    const auto changed = overlay.find(key);
    if (changed != overlay.end())
      return *changed;
    if (const auto *value = source.find(key))
      return *value;
    const auto fallback = defaults.find(key);
    if (fallback != defaults.end())
      return *fallback->second;
    static const Json missing = nullptr;
    return missing;
  }
  bool contains(const std::string &key) const {
    return overlay.contains(key) || source.find(key) || defaults.count(key);
  }
  const Json &at(const std::string &key) const { return read(key); }
  Json &operator[](const std::string &key) {
    if (!overlay.contains(key)) {
      Json value = read(key);
      overlay[key] = std::move(value);
    }
    return overlay[key];
  }
  void erase(const std::string &key) { overlay[key] = nullptr; }
};
const Json &get(const StatOverlay &stat, const char *key);
bool known(const Json &x);
void add_input_time(StatOverlay &stat, const Json &profile,
                    const Json &descriptor, const Json *variant);
bool truth(const Json &x);
double log_value(const Json &x);
bool valid(const StatOverlay &stat, const char *key);
double confidence(const StatOverlay &stat, const char *key, double value);
std::array<double, 3> estimated(const StatOverlay &stat, const char *key,
                                bool ratio, bool probability, bool applicable);
std::size_t characters(const std::string &s);
Json prefix_length(const Json &pre);
Json quality(const Json &profile, const Json &variant);
double precise_sum(const std::vector<double> &values);
struct HistoryGroup {
  bool unknown = false;
  std::map<std::string, std::vector<const Json *>> by_prefix;
  std::array<double, 3> scales{};
};
struct HistoryIndex {
  std::map<std::string, HistoryGroup> models;
  explicit HistoryIndex(const Json &history);
};
std::array<double, history_width> history_features(const HistoryIndex &history,
                                                   const Json &variant,
                                                   const StatOverlay &stat);
struct RecordKey {
  std::uint64_t node;
  const void *statistics;
  bool operator==(const RecordKey &other) const {
    return node == other.node && statistics == other.statistics;
  }
};
struct RecordKeyHash {
  std::size_t operator()(const RecordKey &key) const {
    return std::hash<std::uint64_t>{}(key.node) ^
           (std::hash<const void *>{}(key.statistics) << 1);
  }
};
struct StaticInputs {
  std::size_t ordinal = 0;
  std::vector<int> fields;
};
using StaticInputMap = std::unordered_map<std::string, StaticInputs>;
struct CachedLink {
  Id source;
  std::array<double, 8> relation{};
  double reuse = 0.;
  bool present = false;
};
struct FeatureBatch {
  py::array numeric, history, node_indices, source_indices, relations, reuse,
      reuse_present, edge_mask, node_mask, root_mask, work_indices, sizes;
  py::list templates, work, descriptions_by_chunk;
  std::vector<std::size_t> chunk_offsets;
  bool has_work = false;
  bool cache_features_enabled = true;
  std::size_t chunk_size = 0;
  std::vector<std::pair<std::size_t, std::size_t>> chunk_shapes;
};
template <class T>
py::array array(std::vector<T> &&values,
                const std::vector<py::ssize_t> &shape) {
  auto *owner = new std::vector<T>(std::move(values));
  py::capsule capsule(owner,
                      [](void *p) { delete static_cast<std::vector<T> *>(p); });
  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t stride = sizeof(T);
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return py::array(py::dtype::of<T>(), shape, strides, owner->data(), capsule);
}
struct NativeFeatureContext {
  Json config;
  std::size_t chunk_size;
  bool cache_features_enabled;
  std::size_t worker_limit = 1;
  std::shared_ptr<const HistoryIndex> initial_history;
  std::unordered_map<Id, const Json *> descriptors;
  std::unordered_map<Id, const Json *> variants;
  std::unordered_map<Id, const Json *> initial_node_statistics;
  std::vector<int> field_buckets;
  mutable std::mutex history_mutex;
  mutable std::weak_ptr<const void> cached_history_token;
  mutable std::shared_ptr<const HistoryIndex> cached_history;
  explicit NativeFeatureContext(py::dict value);
  Shard build_range(const std::vector<StatePtr> &states,
                    const ProfileEnvironment::BatchViews &state_views,
                    const Json &profile,
                    const HistoryIndex &history_index, std::size_t begin,
                    std::size_t end) const;
  std::shared_ptr<const HistoryIndex>
  history_for(const ProfileEnvironment &environment) const;
  FeatureBatch pack(const std::vector<StatePtr> &states,
                    const ProfileEnvironment &environment) const;
};
} // namespace semcache::python_features_detail
