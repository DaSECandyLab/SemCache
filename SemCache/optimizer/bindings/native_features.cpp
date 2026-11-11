// Frozen-profile GAT inputs. Python defines text/field identities once; every
// candidate's estimates, graph links and dense storage remain native.
#include "native_features.hpp"
#include "native_feature_merge.hpp"
#include "bridge.hpp"
#include "sampling/profile/environment.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <thread>
#include <exception>

namespace py = pybind11;
namespace {
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
using semcache::sampling::to_python;
constexpr std::size_t numeric_width = 105, history_width = 18;
const Json &get(const Json &object, const char *key) {
  static const Json missing = nullptr;
  const auto found = object.is_object() ? object.find(key) : object.end();
  return found == object.end() ? missing : *found;
}
using semcache::sampling::StatisticView;
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
const Json &get(const StatOverlay &stat, const char *key) {
  return stat.read(key);
}
bool known(const Json &x) {
  return x.is_number() && std::isfinite(x.get<double>());
}
void add_input_time(StatOverlay &stat, const Json &profile,
                    const Json &descriptor, const Json *variant) {
  const bool semantic = descriptor.at("semantic").get<bool>();
  const Json *bank = nullptr;
  const Json *value = nullptr;
  if (semantic) {
    if (variant) {
      bank = &get(get(get(profile, "request_profiles"),
                      variant->at("model").get<std::string>().c_str()),
                  variant->at("kind").get<std::string>().c_str());
      value = &get(*bank, "seconds_per_request");
    }
  } else {
    const auto &label =
        descriptor.at("timing_kind").get_ref<const std::string &>();
    bank = &get(get(get(profile, "operator_time_calibration"), "relational"),
                label.c_str());
    value = &get(*bank, "seconds_per_input");
    if (!known(*value) || value->get<double>() < 0.) {
      bank = &get(get(profile, "relational_profiles"), label.c_str());
      value = &get(*bank, "seconds_per_input");
    }
  }
  stat["seconds_per_input"] = value ? *value : Json();
  double confidence = 0.;
  if (bank) {
    const auto &count = bank->contains("samples") ? get(*bank, "samples")
                                                  : get(*bank, "repeats");
    if (known(count) && count.get<double>() > 0.)
      confidence = count.get<double>() / (count.get<double>() + 1.);
    const auto &cv = get(*bank, "latency_cv");
    if (known(cv) && cv.get<double>() > 0.)
      confidence /= 1. + cv.get<double>();
  }
  if (!get(stat, "reliability").is_object())
    stat["reliability"] = Json::object();
  stat["reliability"]["seconds_per_input"] = confidence;
}
bool truth(const Json &x) {
  if (x.is_null())
    return false;
  if (x.is_boolean())
    return x.get<bool>();
  if (x.is_number())
    return x.get<double>() != 0;
  return !x.empty();
}
double log_value(const Json &x) {
  return known(x) ? std::log1p(std::max(0., x.get<double>())) : 0.;
}
bool valid(const StatOverlay &stat, const char *key) {
  const auto &values = get(stat, "valid");
  return !values.is_object() || !values.contains(key) || truth(values.at(key));
}
double confidence(const StatOverlay &stat, const char *key, double value) {
  const auto &explicit_value = get(get(stat, "reliability"), key);
  double result = 0.;
  if (known(explicit_value))
    result = explicit_value.get<double>();
  else {
    const auto &samples = get(stat, "effective_samples");
    if (known(samples) && samples.get<double>() > 0)
      result = samples.get<double>() / (samples.get<double>() + 1.);
    const auto &bounds = get(get(stat, "intervals"), key);
    if (bounds.is_array() && !bounds.empty() &&
        std::all_of(bounds.begin(), bounds.end(), known)) {
      if (bounds.size() < 2)
        throw std::invalid_argument("Feature interval needs two bounds");
      result /=
          1. + std::max(0., bounds[1].get<double>() - bounds[0].get<double>()) /
                   std::max(1., std::abs(value));
    }
  }
  // Explicit per-metric evidence is independent of generic cardinality flags.
  if (!known(explicit_value) &&
      (truth(get(stat, "uncertain")) ||
       truth(get(stat, "low_survival_uncertain"))))
    result = 0.;
  return std::min(1., std::max(0., result));
}
std::array<double, 3> estimated(const StatOverlay &stat, const char *key,
                                bool ratio, bool probability, bool applicable) {
  const Json *value = &get(stat, key);
  if (std::strcmp(key, "selectivity") == 0 &&
      truth(get(stat, "shared_selectivity")) &&
      stat.contains("estimated_selectivity"))
    value = &stat.at("estimated_selectivity");
  if (!applicable || !known(*value) || value->get<double>() < 0 ||
      !valid(stat, key))
    return {0., 0., 0.};
  double number = value->get<double>();
  if (ratio && probability && number > 1.) {
    if (std::abs(number - 1.) > 1e-12 * std::max(number, 1.))
      throw std::invalid_argument(
          std::string("Probability feature outside [0,1]: ") + key);
    number = 1.;
  }
  return {ratio ? number : std::log1p(std::max(0., number)),
          confidence(stat, key, number), 1.};
}
std::size_t characters(const std::string &s) {
  return std::count_if(s.begin(), s.end(),
                       [](unsigned char c) { return (c & 0xc0) != 0x80; });
}
Json prefix_length(const Json &pre) {
  if (!pre.is_string())
    return nullptr;
  try {
    const auto parts = Json::parse(pre.get<std::string>());
    if (!parts.is_array() || parts.size() != 3 ||
        !std::all_of(parts.begin(), parts.end(),
                     [](const Json &p) { return p.is_string(); }))
      return nullptr;
    auto tail = parts[2].get<std::string>();
    const auto marker = tail.find("<DATA_BLOCK>");
    if (marker != std::string::npos)
      tail.resize(marker);
    return characters(parts[1].get<std::string>()) + characters(tail);
  } catch (const Json::exception &) {
    return nullptr;
  }
}
Json quality(const Json &profile, const Json &variant) {
  auto value = get(get(get(profile, "accuracy"),
                       variant.at("model").get<std::string>().c_str()),
                   variant.at("kind").get<std::string>().c_str());
  if (!value.is_object())
    return value;
  const auto &layouts = get(value, "layouts");
  if (layouts.is_object())
    return get(layouts, variant.at("layout_key").get<std::string>().c_str());
  return value.contains("value") ? value : Json(nullptr);
}
// Expansion summation and the final half-even adjustment match math.fsum.
double precise_sum(const std::vector<double> &values) {
  std::vector<double> partials;
  for (double x : values) {
    std::size_t count = 0;
    for (double y : partials) {
      if (std::abs(x) < std::abs(y))
        std::swap(x, y);
      const double hi = x + y, lo = y - (hi - x);
      if (!std::isfinite(hi))
        throw std::overflow_error("intermediate overflow in fsum");
      if (lo != 0.)
        partials[count++] = lo;
      x = hi;
    }
    partials.resize(count);
    if (x != 0.)
      partials.push_back(x);
  }
  double hi = 0., lo = 0.;
  std::size_t n = partials.size();
  if (n) {
    hi = partials[--n];
    while (n) {
      const double x = hi, y = partials[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo != 0.)
        break;
    }
    if (n && ((lo < 0. && partials[n - 1] < 0.) ||
              (lo > 0. && partials[n - 1] > 0.))) {
      const double y = lo * 2., x = hi + y;
      if (y == x - hi)
        hi = x;
    }
  }
  return hi;
}
struct HistoryGroup {
  bool unknown = false;
  std::map<std::string, std::vector<const Json *>> by_prefix;
  std::array<double, 3> scales{};
};
struct HistoryIndex {
  std::map<std::string, HistoryGroup> models;
  explicit HistoryIndex(const Json &history) {
    const char *distances[] = {"age_seconds", "requests_since",
                               "characters_since"};
    for (const auto &entry : history) {
      auto &group = models[entry.at("model_id").get<std::string>()];
      const auto &pre = get(entry, "pre");
      if (pre.is_null())
        group.unknown = true;
      else
        group
            .by_prefix[pre.is_string() ? pre.get_ref<const std::string &>()
                                       : pre.dump()]
            .push_back(&entry);
      for (std::size_t i = 0; i < 3; ++i) {
        const auto &v = get(entry, distances[i]);
        if (known(v) && v.get<double>() >= 0.)
          group.scales[i] = std::max(group.scales[i], v.get<double>());
      }
    }
  }
};
std::array<double, history_width> history_features(const HistoryIndex &history,
                                                   const Json &variant,
                                                   const StatOverlay &stat) {
  std::array<double, history_width> out{};
  const auto &packed = get(stat, "l_existing_features");
  if (packed.is_array() && packed.size() == 12 &&
      std::all_of(packed.begin(), packed.end(), known) && packed[8] == 1.) {
    for (std::size_t i = 0; i < 6; ++i) {
      out[i] = packed[i].get<double>();
      out[12 + i] = packed[6 + i].get<double>();
    }
  } else {
    const auto &pre = get(stat, "pre");
    if (pre.is_null())
      return out;
    const auto found =
        history.models.find(variant.at("model").get<std::string>());
    const HistoryGroup *group =
        found == history.models.end() ? nullptr : &found->second;
    if (group && group->unknown)
      return out;
    static const std::vector<const Json *> empty;
    const std::vector<const Json *> *entries = &empty;
    if (group) {
      const auto key =
          pre.is_string() ? pre.get_ref<const std::string &>() : pre.dump();
      const auto compatible = group->by_prefix.find(key);
      if (compatible != group->by_prefix.end())
        entries = &compatible->second;
    }
    const auto &matches = get(stat, "l_existing");
    out[2] = std::log1p(static_cast<double>(entries->size()));
    out[14] = 1.;
    std::vector<double> values;
    const Json *strongest = nullptr;
    double maximum = 0.;
    bool complete = true;
    for (const auto *entry : *entries) {
      const auto &v = get(matches, entry->at("id").get<std::string>().c_str());
      if (!known(v) || v.get<double>() < 0.) {
        complete = false;
        break;
      }
      const double number = v.get<double>();
      values.push_back(number);
      if (!strongest || number > maximum ||
          (number == maximum &&
           entry->at("id").get_ref<const std::string &>() <
               strongest->at("id").get_ref<const std::string &>())) {
        strongest = entry;
        maximum = number;
      }
    }
    out[12] = out[13] = complete ? 1. : 0.;
    if (complete && !entries->empty()) {
      out[0] =
          std::log1p(precise_sum(values) / static_cast<double>(values.size()));
      out[1] = std::log1p(maximum);
      const char *distances[] = {"age_seconds", "requests_since",
                                 "characters_since"};
      for (std::size_t i = 0; i < 3; ++i) {
        const auto &v = get(*strongest, distances[i]);
        if (known(v) && v.get<double>() >= 0.) {
          out[3 + i] =
              group->scales[i] ? v.get<double>() / group->scales[i] : 0.;
          out[15 + i] = 1.;
        }
      }
    }
  }
  if (!valid(stat, "l_existing"))
    for (std::size_t i : {0, 1, 3, 4, 5})
      out[i] = out[12 + i] = 0.;
  const double c =
      out[14] && out[2] == 0. ? 1. : confidence(stat, "l_existing", out[0]);
  out[6] = c * out[12];
  out[7] = c * out[13];
  for (std::size_t i = 2; i < 6; ++i)
    out[6 + i] = out[12 + i];
  return out;
}

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
  explicit NativeFeatureContext(py::dict value)
      : config(from_python(value)), chunk_size(config.at("chunk_size")),
        cache_features_enabled(config.value("use_cache_features", true)) {
    if (!chunk_size)
      throw std::invalid_argument(
          "Native inference chunk size must be positive");
    if (config.contains("workers")) {
      const auto &workers = config.at("workers");
      if (!workers.is_number_integer() ||
          (workers.is_number_unsigned() ? workers.get<std::uint64_t>() == 0
                                        : workers.get<std::int64_t>() <= 0))
        throw std::invalid_argument(
            "Native inference workers must be positive integers");
      worker_limit = std::min(std::size_t(32), workers.get<std::size_t>());
    }
  }
  Shard build_range(const std::vector<StatePtr> &states,
                    const ProfileEnvironment &environment, const Json &profile,
                    const Json &node_statistics,
                    const HistoryIndex &history_index, std::size_t begin,
                    std::size_t end) const {
    Shard shard;
    auto &rows = shard.rows;
    auto &graphs = shard.graphs;
    auto &work = shard.work;
    auto &descriptions = shard.descriptions;
    auto &offsets = shard.offsets;
    std::unordered_map<std::string, std::size_t> unique_rows, unique_work;
    std::unordered_map<RecordKey, std::size_t, RecordKeyHash> record_rows;
    std::unordered_map<RecordKey, std::size_t, RecordKeyHash> node_rows;
    StaticInputMap static_inputs;
    std::unordered_map<std::uint64_t, const StaticInputMap::value_type *>
        node_static_inputs;
    std::unordered_map<RecordKey, std::vector<CachedLink>, RecordKeyHash>
        node_topology;
    std::unordered_map<std::uint64_t, const Json *> node_variants;
    std::set<Id> described;
    std::unordered_map<std::string, std::pair<std::size_t, Json>>
        static_templates;
    std::unordered_map<std::string, std::size_t> template_ids;
    auto &max_nodes = shard.max_nodes;
    auto &max_degree = shard.max_degree;
    const auto &descriptors = config.at("descriptors"),
               &variants = config.at("variants");
    const auto &field_buckets = config.at("field_buckets");
    for (std::size_t gi = begin; gi < end; ++gi) {
      if (gi % chunk_size == 0)
        descriptions.emplace_back();
      auto describe = [&](Id nid) {
        if (described.insert(nid).second)
          descriptions.back().push_back(
              {nid, descriptors.at(std::to_string(nid)).at("description")});
      };
      const auto &state = states[gi];
      for (auto current = state; current && current->parent;
           current = current->parent) {
        if (node_variants.count(current->added->handle_id))
          break;
        const auto *variant =
            &variants.at(std::to_string(current->action.operator_id))
                 .at(current->action.domain_index);
        node_variants.emplace(current->added->handle_id, variant);
      }
      const auto nodes = state->nodes();
      std::vector<StatisticView> views;
      views.reserve(nodes.size());
      for (const auto &node : nodes)
        views.push_back(environment.feature_view(state, node->node_id));
      std::vector<const Json *> choices;
      choices.reserve(nodes.size());
      for (const auto &node : nodes) {
        const Json *variant = nullptr;
        if (node->operator_id) {
          const auto found = node_variants.find(node->handle_id);
          if (found != node_variants.end())
            variant = found->second;
        }
        choices.push_back(variant);
      }
      max_nodes = std::max(max_nodes, nodes.size());
      Graph graph;
      graph.rows.reserve(nodes.size());
      graph.roots.reserve(nodes.size());
      graph.links.reserve(nodes.size() * 3);
      graph.link_offsets.reserve(nodes.size() + 1);
      std::map<Id, std::size_t> node_indices, by_operator;
      std::set<Id> children;
      for (const auto &node : nodes)
        for (auto child : node->children)
          children.insert(child);
      for (std::size_t ni = 0; ni < nodes.size(); ++ni) {
        const auto &node = nodes[ni];
        const auto nid = node->node_id;
        node_indices[nid] = ni;
        graph.roots.push_back(!children.count(nid));
        const auto &descriptor = descriptors.at(std::to_string(nid));
        const Json *variant = choices[ni];
        const StatisticView source_view = views[ni];
        const void *source = source_view.identity;
        const RecordKey node_key{node->handle_id, source};
        const auto previous_node = node_rows.find(node_key);
        if (previous_node != node_rows.end()) {
          graph.rows.push_back(previous_node->second);
          describe(nid);
          if (variant && descriptor.at("semantic").get<bool>() &&
              node->operator_id)
            by_operator[*node->operator_id] = ni;
          continue;
        }
        auto static_input = node_static_inputs.find(node->handle_id);
        if (static_input == node_static_inputs.end()) {
          std::vector<int> fields;
          fields.reserve(node->fields.size());
          for (auto field : node->fields)
            fields.push_back(field_buckets.at(field).get<int>());
          std::sort(fields.begin(), fields.end());
          fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
          auto static_key =
              std::to_string(nid) + ":" +
              (variant ? std::to_string(variant->at("index").get<std::size_t>())
                       : "-") +
              ":" + Json(fields).dump();
          const auto inserted =
              static_inputs.try_emplace(std::move(static_key));
          if (inserted.second) {
            inserted.first->second.ordinal = static_inputs.size() - 1;
            inserted.first->second.fields = std::move(fields);
          }
          // unordered_map preserves element addresses when it rehashes.
          static_input =
              node_static_inputs.emplace(node->handle_id, &*inserted.first)
                  .first;
        }
        const auto &static_key = static_input->second->first;
        const auto &fields = static_input->second->second.fields;
        const RecordKey record_key{static_input->second->second.ordinal,
                                   source};
        auto found_record = record_rows.find(record_key);
        std::size_t row_id;
        if (found_record != record_rows.end())
          row_id = found_record->second;
        else {
          Row row;
          StatOverlay stat(source_view);
          const bool semantic = descriptor.at("semantic").get<bool>();
          if (!variant) {
            stat.defaults["input_rows"] = &descriptor.at("estimated_rows");
            stat.defaults["output_rows"] = &descriptor.at("estimated_rows");
            const auto &priors =
                get(node_statistics, std::to_string(nid).c_str());
            if (priors.is_object())
              for (auto i = priors.begin(); i != priors.end(); ++i)
                stat.defaults[i.key()] = &i.value();
            if (get(stat, "input_width").is_null())
              stat["input_width"] = get(stat, "output_width");
          }
          if (semantic && variant) {
            if (get(stat, "pre").is_null() && !variant->at("pre").is_null()) {
              stat["pre"] = variant->at("pre");
              // The prior may have sampled a different layout before this
              // implementation was bound. Its old length is not transferable.
              stat["pre_length"] = variant->at("pre_length");
            }
            if (get(stat, "pre_length").is_null()) {
              stat["pre_length"] = prefix_length(get(stat, "pre"));
              if (stat["pre_length"].is_null())
                stat["pre_length"] = variant->at("pre_length");
            }
            const auto q = quality(profile, *variant);
            if (q.is_object())
              stat["accuracy"] = get(q, "value");
            else if (known(q))
              stat["accuracy"] = q;
            else
              stat.erase("accuracy");
            if (!get(stat, "reliability").is_object())
              stat["reliability"] = Json::object();
            auto reliability = [&](const char *name, const Json &count,
                                   double uncertainty) {
              if (known(count) && count.get<double>() > 0.)
                stat["reliability"][name] = count.get<double>() /
                                            (count.get<double>() + 1.) /
                                            (1. + uncertainty);
            };
            reliability("accuracy", get(q, "effective_samples"), 0.);
          }
          add_input_time(stat, profile, descriptor, variant);
          std::size_t index = 0;
          for (const auto &v : descriptor.at("kind"))
            row.numeric[index++] = v.get<double>();
          if (variant)
            for (const auto &v : variant->at("implementation"))
              row.numeric[index++] = v.get<double>();
          else
            index += 71;
          if (variant && variant->at("has_model").get<bool>()) {
            const auto &length = get(stat, "pre_length");
            row.numeric[86] = log_value(
                length.is_null() ? prefix_length(get(stat, "pre")) : length);
          }
          const char *names[] = {"input_rows", "input_width", "selectivity",
                                 "accuracy", "seconds_per_input"};
          std::array<std::array<double, 3>, 5> triples;
          for (std::size_t j = 0; j < 5; ++j)
            triples[j] = estimated(stat, names[j], j == 2 || j == 3,
                                   semantic || j != 2, semantic || j != 3);
          for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
            for (const auto &t : triples)
              row.numeric[index++] = t[coordinate];
          for (auto v : estimated(stat, "l_self", false, true, semantic))
            row.numeric[index++] = cache_features_enabled ? v : 0.;
          if (index != numeric_width)
            throw std::logic_error("Native feature width mismatch");
          row.has_history = cache_features_enabled && semantic && variant;
          if (row.has_history)
            row.history = history_features(history_index, *variant, stat);
          auto input_template = static_templates.find(static_key);
          if (input_template == static_templates.end()) {
            Json value = {
                {"fields",
                 Json::array({descriptor.at("fields").at(0), Json(fields)})},
                {"description", descriptor.at("description")},
                {"first_fields",
                 variant ? variant->at("first_fields")
                         : Json::array({Json::array(), Json::array()})}};
            auto inserted =
                template_ids.emplace(value.dump(), template_ids.size());
            input_template =
                static_templates
                    .emplace(static_key, std::make_pair(inserted.first->second,
                                                        std::move(value)))
                    .first;
          }

          std::string key = std::to_string(input_template->second.first);
          key.append(reinterpret_cast<const char *>(row.numeric.data()),
                     sizeof(row.numeric));
          key.push_back(row.has_history ? '1' : '0');
          key.append(reinterpret_cast<const char *>(row.history.data()),
                     sizeof(row.history));
          // Prefix/previous-match data affect links, not node projection.
          // Keep estimates separate below if otherwise identical inputs
          // coalesce.
          auto found = unique_rows.find(key);
          if (found == unique_rows.end()) {
            row_id = rows.size();
            unique_rows.emplace(std::move(key), row_id);
            row.value = input_template->second.second;
            row.template_key = row.value.dump();
            rows.push_back(std::move(row));
          } else
            row_id = found->second;
          record_rows.emplace(record_key, row_id);
        }
        node_rows.emplace(node_key, row_id);
        graph.rows.push_back(row_id);
        describe(nid);
        if (variant && descriptor.at("semantic").get<bool>() &&
            node->operator_id)
          by_operator[*node->operator_id] = ni;
      }
      // A retained PlanNode's construction prefix and subtree never change.
      // Its ancestors' model/prefix choices are frozen; runtime feedback
      // only replaces statistics (not the bound prompt). This node's record
      // identity covers changed l_previous strengths/validity. Cache source
      // node IDs, since graph-local positions belong to the current batch.
      for (const auto &node : nodes) {
        const auto index = node_indices.at(node->node_id);
        const StatisticView view = views[index];
        const RecordKey key{node->handle_id, view.identity};
        auto topology = node_topology.find(key);
        if (topology == node_topology.end()) {
          StatOverlay stat(view);
          const Json *variant = choices[index];
          const auto &raw_prefix = get(stat, "pre");
          const Json &prefix =
              variant && raw_prefix.is_null() ? variant->at("pre") : raw_prefix;
          std::map<Id, std::optional<double>> cache;
          if (cache_features_enabled && variant &&
              descriptors.at(std::to_string(node->node_id))
                  .at("semantic")
                  .get<bool>() &&
              !prefix.is_null()) {
            std::set<Id> descendants;
            std::vector<Id> pending = node->children;
            while (!pending.empty()) {
              const auto id = pending.back();
              pending.pop_back();
              if (!descendants.insert(id).second)
                continue;
              const auto &below = nodes.at(node_indices.at(id));
              pending.insert(pending.end(), below->children.begin(),
                             below->children.end());
            }
            for (const auto &[oid, source_index] : by_operator) {
              const auto source_id = nodes[source_index]->node_id;
              if (!descendants.count(source_id))
                continue;
              const auto *source_variant = choices[source_index];
              if (source_variant->at("model") != variant->at("model"))
                continue;
              const StatOverlay source_stat(views[source_index]);
              const auto &raw_source_prefix = get(source_stat, "pre");
              const auto &source_prefix = raw_source_prefix.is_null()
                                              ? source_variant->at("pre")
                                              : raw_source_prefix;
              if (source_prefix != prefix)
                continue;
              const auto &match =
                  get(get(stat, "l_previous"), std::to_string(oid).c_str());
              cache[source_id] = known(match) && match.get<double>() >= 0 &&
                                         valid(stat, "l_previous")
                                     ? std::optional<double>(log_value(match))
                                     : std::nullopt;
            }
          }
          std::set<Id> sources(node->children.begin(), node->children.end());
          for (const auto &cached : cache)
            sources.insert(cached.first);
          sources.insert(node->node_id);
          std::vector<CachedLink> links;
          for (auto source : sources) {
            const auto child =
                std::find(node->children.begin(), node->children.end(), source);
            const bool plan = child != node->children.end();
            const bool cached = source != node->node_id && cache.count(source);
            const int type = source == node->node_id ? 3
                             : plan && cached        ? 2
                             : plan                  ? 0
                                                     : 1;
            const int position =
                plan ? (node->children.size() == 1
                            ? 0
                            : 1 + std::distance(node->children.begin(), child))
                     : 3;
            CachedLink link;
            link.source = source;
            link.relation[type] = 1.;
            if (position < 4)
              link.relation[4 + position] = 1.;
            if (cached && cache.at(source)) {
              link.reuse = *cache.at(source);
              link.present = true;
            }
            links.push_back(link);
          }
          topology = node_topology.emplace(key, std::move(links)).first;
        }
        const auto &cached = topology->second;
        max_degree = std::max(max_degree, cached.size());

        for (const auto &source : cached) {
          Link link;
          link.source = node_indices.at(source.source);
          link.relation = source.relation;
          link.reuse = source.reuse;
          link.present = source.present;
          graph.links.push_back(link);
        }
        graph.link_offsets.push_back(graph.links.size());
      }
      auto remaining = state->remaining();
      auto work_key = Json(remaining).dump();
      auto w = unique_work.find(work_key);
      if (w == unique_work.end()) {
        std::vector<Id> group;
        group.reserve(remaining.size());
        for (auto nid : remaining) {
          group.push_back(nid);
          if (shard.work_operators.count(nid))
            continue;
          const auto &d = descriptors.at(std::to_string(nid));
          Json numeric = d.at("kind");
          const auto &live = get(get(profile, "remaining_selectivity"),
                                 std::to_string(nid).c_str());
          const auto &source =
              live.is_object() ? live : d.at("remaining_statistics");
          StatOverlay stat;
          stat.overlay = source;
          for (auto value : estimated(stat, "selectivity", true,
                                      d.at("semantic").get<bool>(), true))
            numeric.push_back(value);
          add_input_time(stat, profile, d, nullptr);
          for (auto value :
               estimated(stat, "seconds_per_input", false, true, true))
            numeric.push_back(value);
          // Unselected semantic implementations keep separate measured means
          // at the same model slots used by selected implementation features.
          std::array<double, 64 * 3> model_times{};
          for (const auto &model : d.at("remaining_models")) {
            const auto slot = model.at("slot").get<std::size_t>();
            if (slot >= 64)
              throw std::invalid_argument("Unfinished model timing slot out of range");
            Json variant = {{"model", model.at("model_id")},
                            {"kind", d.at("timing_kind")}};
            StatOverlay timing;
            add_input_time(timing, profile, d, &variant);
            const auto values = estimated(timing, "seconds_per_input", false, true, true);
            std::copy(values.begin(), values.end(), model_times.begin() + 3 * slot);
          }
          for (auto value : model_times)
            numeric.push_back(value);
          shard.work_operators.emplace(
              nid, Json{{"fields", d.at("fields")},
                        {"description", d.at("description")},
                        {"numeric", std::move(numeric)}});
        }
        graph.work = work.size();
        shard.work_keys.push_back(work_key);
        unique_work.emplace(std::move(work_key), graph.work);
        work.push_back(std::move(group));
      } else
        graph.work = w->second;
      for (auto nid : remaining)
        describe(nid);
      graph.size = std::log1p(static_cast<double>(remaining.size()));
      graphs.push_back(std::move(graph));
      if ((gi + 1) % chunk_size == 0 || gi + 1 == end)
        offsets.push_back(rows.size());
    }
    return shard;
  }
  FeatureBatch pack(const std::vector<StatePtr> &states,
                    const ProfileEnvironment &environment) const {
    Shard merged;
    std::vector<std::pair<std::size_t, std::size_t>> chunk_shapes;
    std::size_t B = states.size(), N = 1, D = 1, U = 0;
    std::vector<double> numeric, hist, relations, reuse, sizes;
    std::vector<std::int64_t> ids, sources, work_indices;
    std::vector<std::uint8_t> present, mask, node_mask, root_mask;
    {
      py::gil_scoped_release release;
      const auto &initial_profile = config.at("profile");
      const auto &node_statistics = get(initial_profile, "node_statistics");
      // Large immutable scan priors stay borrowed; only calibration banks and
      // remaining-operator evidence need a current per-call overlay.
      Json profile = Json::object(), history;
      for (const auto *key :
           {"accuracy", "request_profiles", "remaining_selectivity", "operator_time_calibration", "relational_profiles"})
        if (initial_profile.contains(key))
          profile[key] = initial_profile.at(key);
      if (!states.empty()) {
        auto context = environment.cache(states.front());
        for (const auto *key :
             {"accuracy", "request_profiles", "remaining_selectivity", "operator_time_calibration", "relational_profiles"})
          if (context.at("profile").contains(key))
            profile[key] = std::move(context.at("profile").at(key));
        history = std::move(context.at("history"));
      } else
        history = config.at("history");
      const HistoryIndex history_index(history);
      const auto chunks = states.size() / chunk_size +
                          std::size_t(states.size() % chunk_size != 0);
      const auto workers = std::min(worker_limit, chunks);
      if (workers <= 1) {
        merged = build_range(states, environment, profile, node_statistics,
                             history_index, 0, states.size());
      } else {
        std::vector<Shard> shards(workers);
        std::vector<std::exception_ptr> errors(workers);
        const auto run = [&](std::size_t worker) {
          try {
            const auto begin = (chunks * worker / workers) * chunk_size;
            const auto end = std::min(
                states.size(), (chunks * (worker + 1) / workers) * chunk_size);
            shards[worker] =
                build_range(states, environment, profile, node_statistics,
                            history_index, begin, end);
          } catch (...) {
            errors[worker] = std::current_exception();
          }
        };
        std::vector<std::thread> threads;
        threads.reserve(workers - 1);
        try {
          for (std::size_t worker = 1; worker < workers; ++worker)
            threads.emplace_back(run, worker);
        } catch (...) {
          for (auto &thread : threads)
            thread.join();
          throw;
        }
        run(0);
        for (auto &thread : threads)
          thread.join();
        for (const auto &error : errors)
          if (error)
            std::rethrow_exception(error);
        merged = merge_shards(shards);
      }
      const auto &graphs = merged.graphs;
      auto &rows = merged.rows;
      N = merged.max_nodes;
      D = merged.max_degree;
      U = rows.size();
      for (std::size_t start = 0; start < graphs.size(); start += chunk_size) {
        std::size_t n = 1, d = 1;
        for (std::size_t i = start;
             i < std::min(graphs.size(), start + chunk_size); ++i) {
          n = std::max(n, graphs[i].rows.size());
          const auto &offsets = graphs[i].link_offsets;
          for (std::size_t j = 0; j + 1 < offsets.size(); ++j)
            d = std::max(d, offsets[j + 1] - offsets[j]);
        }
        chunk_shapes.emplace_back(n, d);
      }
      numeric.resize(U * numeric_width);
      hist.resize(U * history_width);
      relations.resize(B * N * D * 8);
      reuse.resize(B * N * D);
      sizes.resize(B);
      ids.resize(B * N);
      sources.resize(B * N * D);
      work_indices.resize(B);
      present.resize(B * N * D);
      mask.resize(B * N * D);
      node_mask.resize(B * N);
      root_mask.resize(B * N);
      for (std::size_t i = 0; i < U; ++i) {
        auto &r = rows[i];
        std::copy(r.numeric.begin(), r.numeric.end(),
                  numeric.begin() + i * numeric_width);
        std::copy(r.history.begin(), r.history.end(),
                  hist.begin() + i * history_width);
      }
      for (std::size_t i = 0; i < B; ++i) {
        const auto &g = graphs[i];
        work_indices[i] = g.work;
        sizes[i] = g.size;
        for (std::size_t j = 0; j < N; ++j) {
          const auto base = (i * N + j) * D;
          if (j >= g.rows.size()) {
            std::fill(mask.begin() + base, mask.begin() + base + D, 1);
            continue;
          }
          ids[i * N + j] = g.rows[j];
          node_mask[i * N + j] = 1;
          root_mask[i * N + j] = g.roots[j];
          for (std::size_t k = 0; k < g.link_offsets[j + 1] - g.link_offsets[j];
               ++k) {
            const auto &link = g.links[g.link_offsets[j] + k];
            sources[base + k] = link.source;
            reuse[base + k] = link.reuse;
            present[base + k] = link.present;
            mask[base + k] = 1;
            std::copy(link.relation.begin(), link.relation.end(),
                      relations.begin() + (base + k) * 8);
          }
        }
      }
    }
    FeatureBatch batch;
    batch.cache_features_enabled = cache_features_enabled;
    batch.chunk_offsets = std::move(merged.offsets);
    batch.chunk_size = chunk_size;
    batch.chunk_shapes = std::move(chunk_shapes);
    // Numeric/history arrays already own these values. Only immutable task
    // and field metadata cross Python here, shared across identical templates.
    std::unordered_map<std::string, py::object> templates;
    for (auto &row : merged.rows) {
      auto found = templates.find(row.template_key);
      if (found == templates.end())
        found = templates.emplace(row.template_key, to_python(row.value)).first;
      batch.templates.append(found->second);
    }
    std::unordered_map<Id, py::object> work_rows;
    for (auto &group : merged.work) {
      batch.has_work = batch.has_work || !group.empty();
      py::list rows;
      for (auto nid : group) {
        auto found = work_rows.find(nid);
        if (found == work_rows.end())
          found = work_rows
                      .emplace(nid, to_python(merged.work_operators.at(nid)))
                      .first;
        rows.append(found->second);
      }
      batch.work.append(rows);
    }
    for (auto &chunk : merged.descriptions) {
      py::list values;
      for (auto &description : chunk)
        values.append(to_python(description.value));
      batch.descriptions_by_chunk.append(values);
    }
    const auto b = static_cast<py::ssize_t>(B), n = static_cast<py::ssize_t>(N),
               d = static_cast<py::ssize_t>(D), u = static_cast<py::ssize_t>(U);
    batch.numeric = array(std::move(numeric), {u, numeric_width});
    batch.history = array(std::move(hist), {u, history_width});
    batch.node_indices = array(std::move(ids), {b, n});
    batch.source_indices = array(std::move(sources), {b, n, d});
    batch.relations = array(std::move(relations), {b, n, d, 8});
    batch.reuse = array(std::move(reuse), {b, n, d});
    batch.reuse_present = array(std::move(present), {b, n, d});
    batch.edge_mask = array(std::move(mask), {b, n, d});
    batch.node_mask = array(std::move(node_mask), {b, n});
    batch.root_mask = array(std::move(root_mask), {b, n});
    batch.work_indices = array(std::move(work_indices), {b});
    batch.sizes = array(std::move(sizes), {b, 1});
    {
      py::gil_scoped_release release;
      merged = Shard{};
    }
    return batch;
  }
};
}
void bind_native_features(py::module_ &module) {
  module.attr("SUPPORTS_CACHE_FEATURE_ABLATION") = true;
  py::class_<FeatureBatch>(module, "NativeFeatureBatch")
      .def_readonly("cache_features_enabled", &FeatureBatch::cache_features_enabled)
      .def_readonly("numeric", &FeatureBatch::numeric)
      .def_readonly("history", &FeatureBatch::history)
      .def_readonly("templates", &FeatureBatch::templates)
      .def_readonly("work", &FeatureBatch::work)
      .def_readonly("node_indices", &FeatureBatch::node_indices)
      .def_readonly("source_indices", &FeatureBatch::source_indices)
      .def_readonly("relations", &FeatureBatch::relations)
      .def_readonly("reuse", &FeatureBatch::reuse)
      .def_readonly("reuse_present", &FeatureBatch::reuse_present)
      .def_readonly("edge_mask", &FeatureBatch::edge_mask)
      .def_readonly("node_mask", &FeatureBatch::node_mask)
      .def_readonly("root_mask", &FeatureBatch::root_mask)
      .def_readonly("work_indices", &FeatureBatch::work_indices)
      .def_readonly("sizes", &FeatureBatch::sizes)
      .def_readonly("chunk_offsets", &FeatureBatch::chunk_offsets)
      .def_readonly("descriptions_by_chunk",
                    &FeatureBatch::descriptions_by_chunk)
      .def_readonly("has_work", &FeatureBatch::has_work)
      .def_readonly("chunk_size", &FeatureBatch::chunk_size)
      .def_readonly("chunk_shapes", &FeatureBatch::chunk_shapes);
  py::class_<NativeFeatureContext>(module, "NativeFeatureContext")
      .def(py::init<py::dict>())
      .def("pack", &NativeFeatureContext::pack);
}
py::object
native_feature_batch(py::handle context, const std::vector<StatePtr> &states,
                     const semcache::sampling::SearchEnvironment &environment) {
  const auto *profile = dynamic_cast<const ProfileEnvironment *>(&environment);
  if (!profile)
    return py::none();
  return py::cast(
      context.cast<const NativeFeatureContext &>().pack(states, *profile));
}
