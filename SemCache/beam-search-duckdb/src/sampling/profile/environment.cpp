#include "environment.hpp"
#include "../common/accuracy.hpp"
#include "../common/expression_bindings.hpp"
#include "../common/exact_integer.hpp"
#include "../common/prefix_selectivity.hpp"
#include "../statistics/statistics.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace semcache::sampling {
namespace {
using Value = std::shared_ptr<const Json>;
using FieldMap = std::map<std::string, Value>;
struct Record {
  FieldMap fields;
  mutable Value scalar_statistics;
  mutable Value feature_statistics;
  bool contains(const std::string &key) const { return fields.count(key); }
  const Json &get(const std::string &key) const {
    static const Json missing = nullptr;
    auto found = fields.find(key);
    return found == fields.end() ? missing : *found->second;
  }
  Value reference(const std::string &key) const {
    auto found = fields.find(key);
    return found == fields.end() ? Value{} : found->second;
  }
  void set(const std::string &key, Json value) {
    scalar_statistics.reset();
    feature_statistics.reset();
    fields[key] = std::make_shared<const Json>(std::move(value));
  }
  void share(const std::string &key, Value value) {
    if (value) {
      scalar_statistics.reset();
      feature_statistics.reset();
      fields[key] = std::move(value);
    }
  }
  void merge(const Record &other) {
    scalar_statistics.reset();
    feature_statistics.reset();
    for (const auto &field : other.fields)
      fields[field.first] = field.second;
  }
  void erase(const std::string &key) {
    scalar_statistics.reset();
    feature_statistics.reset();
    fields.erase(key);
  }
  Json json(bool raw = false) const {
    static const std::set<std::string> evidence{
        "summary",
        "right_summary",
        "conditional_sketch_plan",
        "adaptive",
        "input_field_summaries",
        "input_field_statistics",
        "output_field_statistics",
        "depth_survival_statistics",
        "depth_selectivity_statistics",
        "prefix_selectivity_statistics"};
    Json out = Json::object();
    for (const auto &field : fields)
      if (!raw || !evidence.count(field.first))
        out[field.first] = *field.second;
    return out;
  }
  Value scalar() const {
    if (!scalar_statistics)
      scalar_statistics = std::make_shared<const Json>(json(true));
    return scalar_statistics;
  }
  Json feature_json() const {
    static const std::set<std::string> consumed{"input_rows",
                                                "input_width",
                                                "output_width",
                                                "output_rows",
                                                "selectivity",
                                                "accuracy",
                                                "seconds_per_request",
                                                "l_self",
                                                "l_existing",
                                                "l_existing_features",
                                                "l_previous",
                                                "pre",
                                                "pre_length",
                                                "reliability",
                                                "effective_samples",
                                                "intervals",
                                                "uncertain",
                                                "low_survival_uncertain",
                                                "valid",
                                                "shared_selectivity",
                                                "estimated_selectivity"};
    Json out = Json::object();
    for (const auto &field : fields)
      if (consumed.count(field.first))
        out[field.first] = *field.second;
    return out;
  }
  Value features() const {
    if (!feature_statistics)
      feature_statistics = std::make_shared<const Json>(feature_json());
    return feature_statistics;
  }
};
using Evidence = std::shared_ptr<const Record>;
using Bank = std::map<std::string, Evidence>;
Evidence empty_record() {
  static const auto empty = std::make_shared<const Record>();
  return empty;
}
Evidence find(const Bank &bank, const std::string &key) {
  auto found = bank.find(key);
  return found == bank.end() ? empty_record() : found->second;
}
Record import_record(Value owner, const Json &raw) {
  Record result;
  if (!raw.is_object())
    throw std::invalid_argument("Profile statistics must be an object");
  for (auto field = raw.begin(); field != raw.end(); ++field)
    result.fields[field.key()] = Value(owner, &field.value());
  return result;
}
const Json &value_reference(const Json &object, const std::string &key,
                            const Json &fallback) {
  if (!object.is_object())
    object.value(key, Json(nullptr));
  const auto found = object.find(key);
  return found == object.end() ? fallback : *found;
}
const Json &input_schema(const Json &op) {
  static const Json empty = Json::array();
  return value_reference(op, "input_field_schema", empty);
}
Bank import_bank(Value owner, const Json &raw) {
  Bank result;
  if (!raw.is_object())
    throw std::invalid_argument("Profile statistics bank must be an object");
  for (auto row = raw.begin(); row != raw.end(); ++row)
    result[row.key()] =
        std::make_shared<const Record>(import_record(owner, row.value()));
  return result;
}
bool number(const Json &value, bool nonnegative = false) {
  if (!value.is_number())
    return false;
  const auto real = value.get<double>();
  return std::isfinite(real) && (!nonnegative || real >= 0);
}
bool flag(const Record &record, const std::string &name) {
  const auto &value = record.get(name);
  if (value.is_null())
    return false;
  if (value.is_boolean())
    return value.get<bool>();
  if (value.is_number())
    return value.get<double>() != 0;
  return !value.empty();
}
std::string kind(const Json &op) {
  auto value = op.value("kind", op.value("type", std::string()));
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}
bool semantic(const Json &op) { return kind(op).rfind("SEMANTIC", 0) == 0; }
std::string field_key(const Json &field) {
  if (field.is_object())
    for (const auto *name : {"binding", "identity", "name", "expression"})
      if (field.contains(name))
        return canonical(field.at(name));
  return canonical(field);
}
Json width(const NodePtr &node, const Json &registry, const Json &summaries) {
  if (!summaries.is_object())
    return node->fields.empty() ? Json(0) : Json(nullptr);
  double total = 0;
  for (auto index : node->fields) {
    const auto &field = registry.at(index);
    auto key = field_key(field);
    auto found = summaries.find(key);
    if (found == summaries.end()) {
      auto fallback =
          field.is_object() ? field.value("name", Json(nullptr)) : field;
      if (fallback.is_string())
        found = summaries.find(fallback.get<std::string>());
    }
    auto bytes = found == summaries.end()
                     ? Json(nullptr)
                     : found->value("average_bytes", Json(nullptr));
    if (!number(bytes, true))
      return nullptr;
    total += bytes.get<double>();
  }
  return total;
}
bool compatible(const Json &a, const Json &b) {
  if (a.at("metadata").at("serializer") != b.at("metadata").at("serializer"))
    return false;
  for (auto key = a.at("sketches").begin(); key != a.at("sketches").end();
       ++key)
    if (b.at("sketches").contains(key.key()))
      return true;
  return false;
}
struct Choice {
  Id operator_id, node_id;
  Json action;
  std::string key;
};
struct NodeIndex {
  std::vector<Id> ids;
  std::unordered_map<Id, std::size_t> slots;
  explicit NodeIndex(const std::set<Id> &values)
      : ids(values.begin(), values.end()) {
    for (std::size_t i = 0; i < ids.size(); ++i)
      slots.emplace(ids[i], i);
  }
};
struct StateData {
  StatePtr state;
  std::vector<Evidence> node_records;
  std::vector<std::shared_ptr<const Choice>> actions;
  std::vector<NodePtr> nodes;
  std::vector<NodePtr> node_by_slot;
};
}

struct ProfileEnvironment::Impl {
  Value imported;
  const Json &profile_, &query, &fields;
  Json history_;
  Json remaining_selectivity_ = Json::object();
  std::map<Id, Json> ops, logical;
  // Owner and observation workers share one immutable ordering. StateData
  // contains vectors only, never a borrowed pointer into a worker's index.
  std::shared_ptr<const NodeIndex> node_index;
  Bank initial, node_priors, operator_priors;
  mutable std::map<Id, Evidence> scan_children;
  std::unordered_map<
      Id, std::unordered_map<std::string, std::shared_ptr<const Choice>>>
      choice_cache;
  std::size_t choice_cache_size = 0;
  bool sample_depth_selectivity = true;
  bool runtime_observed = false;
  std::vector<Record> sources;
  struct SourceGroup {
    std::unordered_map<std::string, std::vector<const Record *>> prefixes;
    bool unknown_prefix = false;
  };
  std::unordered_map<std::string, SourceGroup> source_index;
  bool source_index_compatible = true;
  std::unordered_map<std::uint64_t, StateData> states;
  ReuseCache reuse;
  std::map<std::pair<Id, std::string>, Value> prefixes;
  std::map<std::string, Value> prefix_lengths;
  struct SurvivalMemo {
    std::vector<std::uint64_t> inputs;
    Value summary, value;
  };
  mutable std::unordered_map<std::string, SurvivalMemo> survival_memo;
  std::map<std::string, Value> prefix_cooccurrences;
  std::map<std::string, std::pair<double, std::size_t>> metrics;
  std::vector<std::unique_ptr<Impl>> observation_workers;

  explicit Impl(const Json &config)
      : imported(std::make_shared<const Json>(config)),
        profile_(imported->at("profile")), query(imported->at("query")),
        fields(imported->at("fields")),
        history_(
            config.value("history", profile_.value("history", Json::array()))) {
    for (const auto &op : config.at("operators"))
      ops[op.at("operator_id").get<Id>()] = op;
    for (const auto &node : query.at("operators"))
      logical[node.at("node_id").get<Id>()] = node;
    if (imported->contains("initial_statistics"))
      initial = import_bank(imported, imported->at("initial_statistics"));
    else if (profile_.contains("statistics"))
      initial = import_bank(imported, profile_.at("statistics"));
    if (profile_.value("sampling_strategy", std::string()) ==
            "independent_base" &&
        profile_.contains("source_statistics")) {
      const auto sources =
          import_bank(imported, profile_.at("source_statistics"));
      for (const auto &[id, source] : sources) {
        Record record = *find(initial, "scan:" + id);
        record.merge(*source);
        initial["scan:" + id] =
            std::make_shared<const Record>(std::move(record));
      }
    }
    if (profile_.contains("node_statistics"))
      node_priors = import_bank(imported, profile_.at("node_statistics"));
    if (profile_.contains("prefix_cooccurrences"))
      for (auto entry = profile_.at("prefix_cooccurrences").begin();
           entry != profile_.at("prefix_cooccurrences").end(); ++entry)
        prefix_cooccurrences.emplace(entry.key(), Value(imported, &entry.value()));
    sample_depth_selectivity = config.value("sampling", Json::object())
                                   .value("sample_depth_selectivity", true);
    if (profile_.value("sampling_strategy", std::string()) ==
            "independent_base" &&
        profile_.contains("operator_statistics"))
      operator_priors =
          import_bank(imported, profile_.at("operator_statistics"));
    std::set<Id> node_ids;
    for (const auto &[id, node] : logical)
      node_ids.insert(id);
    for (const auto &[_, op] : ops)
      if (op.contains("node_id") && op.at("node_id").is_number_integer())
        node_ids.insert(op.at("node_id").get<Id>());
    // Logical-query compatibility profiles may declare source scans only in
    // their ingress bank. Ignore unrelated, nonnumeric statistic identities.
    for (const auto &[key, record] : initial)
      if (key.compare(0, 5, "scan:") == 0)
        try {
          std::size_t parsed = 0;
          const auto id = std::stoll(key.substr(5), &parsed);
          if (parsed == key.size() - 5)
            node_ids.insert(id);
        } catch (const std::exception &) {
        }
    node_index = std::make_shared<const NodeIndex>(node_ids);
    for (const auto &[oid, op] : ops) {
      const auto nid = op.at("node_id").get<Id>();
      const auto record = semantic(op)
                              ? find(operator_priors, std::to_string(oid))
                              : find(node_priors, std::to_string(nid));
      Json evidence = Json::object();
      for (const auto *key :
           {"selectivity", "estimated_selectivity", "shared_selectivity",
            "effective_samples", "valid", "reliability", "intervals",
            "uncertain", "low_survival_uncertain"})
        if (record->contains(key))
          evidence[key] = record->get(key);
      if (!number(evidence.value("selectivity", Json()))) {
        if (kind(op).find("PROJECTION") != std::string::npos)
          evidence["selectivity"] = 1.;
        else if (number(record->get("duckdb_selectivity")))
          evidence["selectivity"] = record->get("duckdb_selectivity");
      }
      remaining_selectivity_[std::to_string(nid)] = std::move(evidence);
    }
    // History owns a single imported catalogue; alias summary values in place.
    Value history_owner = std::make_shared<const Json>(history_);
    for (const auto &source : *history_owner)
      sources.push_back(import_record(history_owner, source));
    index_sources();
  }
  // Workers share immutable imported facts, but own every mutable formula
  // cache. Their history catalogue is already retained by source aliases.
  explicit Impl(const Impl &source)
      : imported(source.imported), profile_(source.profile_),
        query(source.query), fields(source.fields),
        remaining_selectivity_(source.remaining_selectivity_), ops(source.ops),
        logical(source.logical), node_index(source.node_index),
        initial(source.initial), node_priors(source.node_priors),
        operator_priors(source.operator_priors),
        prefix_cooccurrences(source.prefix_cooccurrences),
        sample_depth_selectivity(source.sample_depth_selectivity),
        sources(source.sources) {
    index_sources();
  }
  void index_sources() {
    // Build pointers only after the vector has reached its immutable size.
    for (const auto &source : sources) {
      const auto &model = source.get("model_id");
      const auto &prefix = source.get("pre");
      if (!model.is_string() || (!prefix.is_null() && !prefix.is_string())) {
        source_index_compatible = false;
        continue;
      }
      auto &group = source_index[model.get_ref<const std::string &>()];
      if (prefix.is_null())
        group.unknown_prefix = true;
      else
        group.prefixes[prefix.get_ref<const std::string &>()].push_back(
            &source);
    }
  }
  std::set<Id> descendants(const std::vector<NodePtr> &nodes, Id target) const {
    std::map<Id, std::set<Id>> paths;
    for (const auto &node : nodes) {
      auto &path = paths[node->node_id];
      for (auto child : node->children) {
        path.insert(child);
        path.insert(paths[child].begin(), paths[child].end());
      }
    }
    return paths.at(target);
  }
  const NodePtr &indexed_node(const StateData &data, Id id) const {
    const auto &node = data.node_by_slot.at(node_index->slots.at(id));
    if (!node)
      throw std::out_of_range("Missing state node");
    return node;
  }
  const Evidence &node_record(const StateData &data, Id id) const {
    static const Evidence missing;
    const auto slot = node_index->slots.find(id);
    return slot == node_index->slots.end() ? missing
                                           : data.node_records.at(slot->second);
  }
  const Record &child(const StateData &data, const NodePtr &node) const {
    if (!node->operator_id) {
      auto found = scan_children.find(node->node_id);
      if (found == scan_children.end()) {
        // Source priors and ingress scan observations are immutable, including
        // after runtime feedback updates a selected operator's node record.
        Record result = *find(node_priors, std::to_string(node->node_id));
        result.merge(*find(initial, "scan:" + std::to_string(node->node_id)));
        found = scan_children
                    .emplace(node->node_id,
                             std::make_shared<const Record>(std::move(result)))
                    .first;
      }
      return *found->second;
    }
    const auto &record = node_record(data, node->node_id);
    return record ? *record : *empty_record();
  }
  CrossDependence cache_dependence(const Record &source, const Record &target) const {
    const auto &a = source.get("cache_field_origin"),
               &b = target.get("cache_field_origin");
    if (!a.is_array() || !b.is_array() || a.size() != 2 || b.size() != 2 ||
        a[0] != b[0])
      return {};
    if (a[1] == b[1])
      return {true, {}};
    const auto key = canonical(Json::array({a[0], std::min(a[1], b[1]),
                                                  std::max(a[1], b[1])}));
    const auto found = prefix_cooccurrences.find(key);
    if (found == prefix_cooccurrences.end())
      throw std::invalid_argument(
          "Missing frozen same-table prefix cooccurrence statistics: " + key);
    return {false, found->second};
  }
  void cache_edges(Record &result, const StateData &context,
                   const std::vector<NodePtr> &nodes, const NodePtr &node,
                   const Choice &chosen) {
    const auto &op = ops.at(chosen.operator_id);
    if (node->children.size() == 1 &&
        kind(op).find("JOIN") != std::string::npos &&
        !flag(result, "pair_request_summary")) {
      result.set("l_self", nullptr);
      result.set("l_existing", Json::object());
      result.set("l_previous", Json::object());
      result.set("cache_sources_complete", false);
      result.set("pair_input", true);
      return;
    }
    auto summary = result.reference("summary");
    if (!summary || summary->is_null()) {
      result.set("cache_sources_complete", false);
      return;
    }
    if (summary->at("metadata").at("serializer") !=
        "prompt-v4-first-field-value-v1")
      throw std::invalid_argument(
          "Legacy input summary: regenerate conditional profiles from "
          "execution telemetry");
    if (result.get("pre").is_null() && op.contains("default_layout")) {
      const auto key = std::make_pair(chosen.operator_id,
                                      canonical(chosen.action.at("layout")));
      auto found = prefixes.find(key);
      if (found == prefixes.end()) {
        static const Json empty = Json::object();
        const auto &specifications =
            value_reference(profile_, "operator_specifications", empty);
        const auto label = op.at("kind").get<std::string>();
        auto prefix = prefix_namespace(
            op, chosen.action,
            profile_.value("system_prefix", std::string("semantic-global-v1")),
            specifications.value(label, label));
        found =
            prefixes
                .emplace(key, std::make_shared<const Json>(std::move(prefix)))
                .first;
      }
      result.share("pre", found->second);
    }
    static const Json one = 1.;
    const auto &alpha = result.contains("alpha") ? result.get("alpha") : one;
    const auto &right_alpha =
        result.contains("right_alpha") ? result.get("right_alpha") : one;
    if (!result.get("pre").is_null()) {
      const auto &prefix = result.get("pre").get_ref<const std::string &>();
      auto found = prefix_lengths.find(prefix);
      if (found == prefix_lengths.end())
        found =
            prefix_lengths
                .emplace(prefix,
                         std::make_shared<const Json>(prefix_length(prefix)))
                .first;
      result.share("pre_length", found->second);
    }
    auto right = flag(result, "pair_input") ? Value{}
                                            : result.reference("right_summary");
    if (right && right->is_null())
      right.reset();
    if (right && !compatible(*right, *summary))
      throw std::invalid_argument(
          "Join field summaries must use compatible coordinates");
    const auto requests = result.get("input_rows");
    const auto &sides = result.get("input_side_rows");
    const Json left_requests = right && sides.is_array() && sides.size() == 2
                                   ? sides[0]
                                   : Json(nullptr),
               right_requests = right && sides.is_array() && sides.size() == 2
                                    ? sides[1]
                                    : Json(nullptr);
    result.set("l_self",
               right ? reuse.join(summary, right,
                                  result.get("left_total_length"), alpha,
                                  right_alpha, left_requests, right_requests)
                     : reuse.self(summary, alpha, requests));
    Json existing = Json::object(), previous = Json::object();
    auto right_rate = right ? survival_rate(right_alpha) : Json(nullptr);
    result.set("l_existing", existing);
    result.set("l_previous", previous);
    result.set("cache_sources_complete", !result.get("pre").is_null());
    if (right && right_rate.is_null()) {
      result.set("cache_sources_complete", false);
      return;
    }
    const auto right_count =
        right ? right->at("count").get<double>() * right_rate.get<double>()
              : 1.;
    auto edge = [&](const Record &source, const Json &source_alpha,
                    const CrossDependence &dependence = CrossDependence{}) {
      auto bank = source.reference("summary");
      if (!bank || bank->is_null() || !compatible(*bank, *summary))
        return Json(nullptr);
      auto value =
          reuse.cross(bank, summary, source_alpha, alpha, dependence, requests);
      if (!value.is_null() && right && requests.is_null())
        value =
            right_count ? Json(value.get<double>() / right_count) : Json(0.);
      return value;
    };
    const auto &model = chosen.action.at("model_id");
    const auto &prefix = result.get("pre");
    if (source_index_compatible && model.is_string() &&
        (prefix.is_string() || prefix.is_null())) {
      const auto group =
          source_index.find(model.get_ref<const std::string &>());
      if (group != source_index.end()) {
        if (group->second.unknown_prefix)
          result.set("cache_sources_complete", false);
        if (prefix.is_string()) {
          const auto matches = group->second.prefixes.find(
              prefix.get_ref<const std::string &>());
          if (matches != group->second.prefixes.end())
            for (const auto *source : matches->second) {
              auto value = edge(*source, one);
              existing[source->get("id").get<std::string>()] = std::move(value);
            }
        }
      }
    } else
      for (const auto &source : sources) {
        if (source.get("model_id") != model)
          continue;
        if (source.get("pre").is_null())
          result.set("cache_sources_complete", false);
        else if (source.get("pre") == prefix) {
          auto value = edge(source, one);
          existing[source.get("id").get<std::string>()] = std::move(value);
        }
      }
    auto ancestors = descendants(nodes, node->node_id);
    for (std::size_t index = 0; index < context.actions.size(); ++index) {
      const auto &old = *context.actions[index];
      if (!semantic(ops.at(old.operator_id)) || !ancestors.count(old.node_id) ||
          old.action.at("model_id") != chosen.action.at("model_id"))
        continue;
      const auto &source = node_record(context, old.node_id);
      if (source->get("pre").is_null())
        result.set("cache_sources_complete", false);
      else if (!result.get("pre").is_null() &&
               source->get("pre") == result.get("pre")) {
        const auto dependence = cache_dependence(*source, result);
        auto value = edge(
            *source, source->contains("alpha") ? source->get("alpha") : one,
            dependence);
        previous[std::to_string(old.operator_id)] = std::move(value);
      }
    }
    result.set("l_existing", std::move(existing));
    result.set("l_previous", std::move(previous));
  }
  bool has_field(const NodePtr &node, const std::string &name) const {
    for (const auto id : node->fields) {
      const auto &field = fields.at(id);
      if ((field.is_string() && field.get<std::string>() == name) ||
          (field.is_object() && field.value("name", std::string()) == name))
        return true;
    }
    return false;
  }
  Json predecessor_survival(const StateData &context,
                            const StateData &structure, const NodePtr &node,
                            const std::string &name, const std::string &depth,
                            const std::string &expression = "",
                            const std::string &expression_type = "") const {
    if (!node->operator_id)
      return child(context, node).get("output_rows") == 0 ? 0. : 1.;
    const auto &op = ops.at(*node->operator_id);
    if (prefix_population_boundary(kind(op)))
      return child(context, node).get("output_rows") == 0 ? 0. : 1.;
    std::string source_name = name;
    if (!expression.empty())
      for (const auto &descriptor : input_schema(op))
        if (descriptor.value("sample_expression", std::string()) ==
                expression &&
            descriptor.value("type", std::string()) == expression_type) {
          source_name = descriptor.value("value_key", name);
          break;
        }
    if (kind(op) == "SEMANTIC_PROJECTION") {
      std::set<Id> existing;
      for (const auto child_id : node->children)
        for (const auto id : indexed_node(structure, child_id)->fields)
          existing.insert(id);
      for (const auto id : node->fields)
        if (!existing.count(id)) {
          const auto &field = fields.at(id);
          const auto generated = field.is_string() ? field.get<std::string>()
                                 : field.contains("name")
                                     ? field.at("name").get<std::string>()
                                     : canonical(field.at("binding"));
          if (generated == name ||
              (!expression.empty() &&
               expression.find('"' + generated + '"') != std::string::npos))
            return 1.;
        }
    }
    const auto &record = child(context, node);
    auto local = record.get("selectivity");
    const auto label = kind(op);
    if (prefix_preserving_operator(label))
      local = 1.;
    if (label.find("JOIN") != std::string::npos) {
      const auto bindings = expression_bindings(expression);
      for (std::size_t side = 0; side < node->children.size(); ++side) {
        const auto &source = indexed_node(structure, node->children[side]);
        bool contains = has_field(source, name);
        for (const auto id : source->fields) {
          const auto &field = fields.at(id);
          const auto column = field.is_string() ? field.get<std::string>()
                              : field.contains("name")
                                  ? field.at("name").get<std::string>()
                                  : canonical(field.at("binding"));
          contains |= bindings.count(column) != 0;
        }
        const auto side_rate =
            record.get(side ? "right_selectivity" : "left_selectivity");
        if (contains && number(side_rate, true))
          local = side_rate;
      }
    }
    bool measured = false;
    // Frozen local estimates retain the original field identity. Prefer their
    // complete grid; raw observations remain available as evidence only.
    const auto &dense = record.get("prefix_selectivity_statistics");
    const auto &raw = record.get("depth_selectivity_statistics");
    auto matching = [&](const std::string &key) {
      const auto suffix = ':' + source_name;
      return key.size() >= suffix.size() &&
             key.compare(key.size() - suffix.size(), suffix.size(), suffix) ==
                 0;
    };
    if (dense.is_object())
      for (auto field = dense.begin(); field != dense.end(); ++field) {
        if (!matching(field.key()))
          continue;
        if (field.value().contains(depth)) {
          const auto &entry = field.value().at(depth);
          const auto value = entry.value("rate", Json(nullptr));
          const auto source = entry.value("source", std::string());
          if (number(value, true) &&
              (entry.value("observed", std::size_t(0)) > 0 ||
               source == "exact" || source == "empty_population")) {
            local = value;
            measured = true;
            break;
          }
        }
        // Recompute missing depths and historical interpolation from original
        // measurements and the operator's whole-field rate.
        const auto evidence = raw.is_object()
                                  ? raw.value(field.key(), Json::object())
                                  : Json::object();
        auto field_scalar = local;
        if (field.value().contains("0")) {
          const auto &zero = field.value().at("0");
          const auto source = zero.value("source", std::string());
          if (zero.value("observed", std::size_t(0)) > 0 ||
              source == "operator_scalar" || source == "exact" ||
              source == "empty_population")
            field_scalar = zero.value("rate", local);
        }
        const auto expanded =
            complete_prefix_selectivity({{depth, true}}, evidence, field_scalar,
                                        prefix_preserving_operator(label));
        const auto value = expanded.at(depth).at("rate");
        if (number(value, true)) {
          local = value;
          measured = true;
          break;
        }
      }
    if (!measured && raw.is_object())
      for (auto field = raw.begin(); field != raw.end(); ++field)
        if (matching(field.key())) {
          const auto expanded =
              complete_prefix_selectivity({{depth, true}}, field.value(), local,
                                          prefix_preserving_operator(label));
          const auto value = expanded.at(depth).at("rate");
          if (number(value, true)) {
            local = value;
            measured = true;
            break;
          }
        }
    Json incoming = 1.;
    for (std::size_t side = 0; side < node->children.size(); ++side) {
      const auto &source = indexed_node(structure, node->children[side]);
      bool contains = has_field(source, name);
      const auto bindings = expression_bindings(expression);
      for (const auto id : source->fields) {
        const auto &field = fields.at(id);
        const auto column = field.is_string() ? field.get<std::string>()
                            : field.contains("name")
                                ? field.at("name").get<std::string>()
                                : canonical(field.at("binding"));
        contains |= bindings.count(column) != 0;
      }
      if (!contains && (expression.empty() || node->children.size() != 1))
        continue;
      const auto inherited = predecessor_survival(
          context, structure, source, name, depth, expression, expression_type);
      incoming = incoming == 0 || inherited == 0 ? Json(0.)
                 : number(incoming, true) && number(inherited, true)
                     ? Json(incoming.get<double>() * inherited.get<double>())
                     : Json(nullptr);
      if (!measured && label.find("JOIN") != std::string::npos) {
        const auto side_rate =
            record.get(side ? "right_selectivity" : "left_selectivity");
        if (number(side_rate, true))
          local = side_rate;
      }
    }
    return incoming == 0 || local == 0 ? Json(0.)
           : number(incoming, true) && number(local, true)
               ? Json(incoming.get<double>() * local.get<double>())
               : Json(nullptr);
  }
  Value incoming_survival(const StateData &context, const StateData &structure,
                          const NodePtr &node, const std::string &name,
                          Value summary_owner) const {
    static const Value missing = std::make_shared<const Json>(nullptr);
    if (!summary_owner)
      summary_owner = missing;
    const auto &summary = *summary_owner;
    std::string expression, expression_type;
    if (node->operator_id)
      for (const auto &descriptor : input_schema(ops.at(*node->operator_id)))
        if (descriptor.value("value_key", std::string()) == name) {
          expression = descriptor.value("sample_expression", std::string());
          expression_type = descriptor.value("type", std::string());
          break;
        }
    std::vector<std::uint64_t> inputs;
    for (const auto child : node->children)
      inputs.push_back(indexed_node(structure, child)->handle_id);
    const auto identity = canonical(
        Json::array({inputs, name, expression, expression_type,
                     reinterpret_cast<std::uintptr_t>(summary_owner.get())}));
    if (!runtime_observed) {
      const auto found = survival_memo.find(identity);
      if (found != survival_memo.end())
        return found->second.value;
    }
    std::vector<NodePtr> sources;
    const auto bindings = expression_bindings(expression);
    for (const auto child_id : node->children) {
      const auto &source = indexed_node(structure, child_id);
      bool contains = has_field(source, name);
      for (const auto id : source->fields) {
        const auto &field = fields.at(id);
        const auto column = field.is_string() ? field.get<std::string>()
                            : field.contains("name")
                                ? field.at("name").get<std::string>()
                                : canonical(field.at("binding"));
        contains |= bindings.count(column) != 0;
      }
      if (contains || (!expression.empty() && node->children.size() == 1))
        sources.push_back(source);
    }
    auto rate = [&](const std::string &depth) {
      Json value = 1.;
      for (const auto &source : sources) {
        const auto factor =
            predecessor_survival(context, structure, source, name, depth,
                                 expression, expression_type);
        value = value == 0 || factor == 0 ? Json(0.)
                : number(value, true) && number(factor, true)
                    ? Json(value.get<double>() * factor.get<double>())
                    : Json(nullptr);
      }
      return value;
    };
    Json depths = Json::object();
    if (summary.is_object() && summary.contains("sketches"))
      for (auto depth = summary.at("sketches").begin();
           depth != summary.at("sketches").end(); ++depth)
        if (depth.key() != "0")
          depths[depth.key()] = depth.value().at("entries").empty()
                                    ? Json(0.)
                                    : rate(depth.key());
    auto value = std::make_shared<const Json>(
        Json{{"total", rate("0")}, {"depths", std::move(depths)}});
    if (!runtime_observed)
      survival_memo.emplace(
          identity,
          SurvivalMemo{std::move(inputs), std::move(summary_owner), value});
    return value;
  }
  void estimate_survivals(Record &result, const StateData &context,
                          const StateData &structure, const NodePtr &node,
                          const Choice &chosen) const {
    if (!sample_depth_selectivity || !semantic(ops.at(chosen.operator_id)))
      return;
    const auto &layout = chosen.action.at("layout");
    const auto &op = ops.at(chosen.operator_id);
    auto describe = [&](const std::string &name) {
      for (const auto &field : input_schema(op))
        if (field.at("name") == name)
          return field.value("value_key", name);
      return name;
    };
    auto first = layout.value("field_order",
                              layout.value("left_field_order", Json::array()));
    if (!first.empty())
      result.share("alpha",
                   incoming_survival(context, structure, node,
                                     describe(first[0].get<std::string>()),
                                     result.reference("summary")));
    const auto right = layout.value("right_field_order", Json::array());
    if (!right.empty())
      result.share("right_alpha",
                   incoming_survival(context, structure, node,
                                     describe(right[0].get<std::string>()),
                                     result.reference("right_summary")));
  }

  Record estimate(const StateData &context, const StateData &structure,
                  const NodePtr &node, const Choice &chosen) {
    const auto &op = ops.at(chosen.operator_id);
    Record result = *find(node_priors, std::to_string(chosen.node_id));
    if (semantic(op)) {
      result = Record{};
      const auto source =
          find(operator_priors, std::to_string(chosen.operator_id));
      const auto &implementations = source->get("implementations");
      if (source->get("selectivity_scope") == "operator")
        result = *source;
      else if (implementations.is_object()) {
        const auto implementation =
            canonical(Json{{"operator_id", chosen.operator_id},
                           {"model_id", chosen.action.at("model_id")},
                           {"layout", chosen.action.at("layout")}});
        const auto measured = implementations.find(implementation);
        if (measured != implementations.end())
          result = import_record(imported, *measured);
      } else if (source->get("sample_model") == chosen.action.at("model_id") &&
                 source->get("sample_layout") == chosen.action.at("layout"))
        result = *source;
    }
    if (semantic(op)) {
      // A construction adds each logical operator once. An existing node is
      // its selected/runtime-updated record; before construction only the
      // immutable ingress bank can contain an observation for this action.
      const auto &selected = node_record(context, chosen.node_id);
      const auto own = selected ? selected : find(initial, chosen.key);
      if (own->get("sampling_strategy") == "independent_base")
        result.merge(*own);
      static const Json empty = Json::object(), unknown = nullptr;
      const auto &bank = value_reference(profile_, "accuracy", empty);
      const auto &model = value_reference(
          bank, chosen.action.at("model_id").get<std::string>(), empty);
      const auto &task =
          value_reference(model, op.at("kind").get<std::string>(), unknown);
      const auto &evidence =
          accuracy_evidence(task, chosen.action.at("layout"));
      const auto &accuracy = evidence.is_object()
                                 ? value_reference(evidence, "value", unknown)
                                 : evidence;
      if (number(accuracy))
        result.set("accuracy", accuracy);
      else
        result.erase("accuracy");
    }
    result.set("sampling_strategy", "independent_base");
    std::vector<const Record *> children;
    Json sizes = Json::array();
    bool known = !node->children.empty();
    for (auto id : node->children) {
      const auto &info = child(context, indexed_node(structure, id));
      auto size = info.contains("output_rows")
                      ? info.get("output_rows")
                      : logical.at(id).value("estimated_rows", Json(nullptr));
      known &= number(size, true);
      sizes.push_back(size);
      children.push_back(&info);
    }
    if (known) {
      // Cardinality propagation touches scalar facts only; distribution/sketch
      // fields keep their immutable aliases when no new merged bank is needed.
      Json scalar = Json::object();
      for (const auto *key :
           {"input_rows", "output_rows", "input_side_rows", "selectivity",
            "relational_cost", "relational_seconds", "cardinality_basis",
            "sampled_selectivity", "sampling_selectivity",
            "sampled_selectivity_complete", "uncertain",
            "survival_estimated", "group_count_estimated", "duckdb_selectivity",
            "estimated_selectivity", "selectivity_fusion",
            "survival_estimation_basis"})
        if (result.contains(key))
          scalar[key] = result.get(key);
      auto propagated = propagate(op, scalar, sizes.get<std::vector<Json>>());
      for (const auto *key :
           {"input_rows", "output_rows", "input_side_rows", "selectivity",
            "relational_cost", "relational_seconds", "cardinality_basis",
            "group_count_estimated", "duckdb_selectivity", "sampling_selectivity",
            "estimated_selectivity", "selectivity_fusion", "survival_estimated",
            "survival_estimation_basis"}) {
        if (propagated.contains(key))
          result.set(key, propagated.at(key));
        else
          result.erase(key);
      }
      Json widths = Json::array();
      double input_width = 0;
      bool width_known = true;
      for (const auto &info : children) {
        widths.push_back(info->get("output_width"));
        width_known &= number(info->get("output_width"));
        if (number(info->get("output_width")))
          input_width += info->get("output_width").get<double>();
      }
      if (width_known) {
        result.set("input_width", input_width);
        result.set("input_side_widths", widths);
      }
      const bool needs_input = !result.contains("input_field_statistics");
      const bool needs_output =
          kind(op).find("PROJECTION") == std::string::npos &&
          !result.contains("output_field_statistics");
      if (needs_input || needs_output) {
        Value field_statistics;
        if (children.size() == 1)
          field_statistics = children[0]->reference("output_field_statistics");
        if (!field_statistics) {
          Json merged = Json::object();
          for (const auto &info : children)
            if (info->get("output_field_statistics").is_object())
              merged.update(info->get("output_field_statistics"));
          field_statistics = std::make_shared<const Json>(std::move(merged));
        }
        if (needs_input)
          result.share("input_field_statistics", field_statistics);
        if (needs_output)
          result.share("output_field_statistics", field_statistics);
      }
      auto output_width =
          width(node, fields, result.get("output_field_statistics"));
      if (!output_width.is_null())
        result.set("output_width", std::move(output_width));
    } else if (!semantic(op)) {
      for (const auto *key : {"input_rows", "output_rows", "relational_cost",
                              "relational_seconds"})
        result.erase(key);
    }
    if (semantic(op) && sample_depth_selectivity &&
        result.get("input_field_summaries").is_object()) {
      const auto &layout = chosen.action.at("layout");
      auto attach = [&](const char *order, const char *slot) {
        const auto names = layout.value(order, Json::array());
        if (names.empty())
          return;
        auto key = names[0].get<std::string>();
        for (const auto &field : input_schema(op))
          if (field.at("name") == names[0])
            key = field.value("value_key", key);
        const auto owner = result.reference("input_field_summaries");
        const auto &summaries = *owner;
        if (summaries.contains(key) && summaries.at(key).contains("summary")) {
          result.share(slot, Value(owner, &summaries.at(key).at("summary")));
          if (std::string(slot) == "summary") {
            if (summaries.at(key).contains("origin"))
              result.share("cache_field_origin", Value(owner, &summaries.at(key).at("origin")));
            else
              result.erase("cache_field_origin");
          }
        }
      };
      attach(layout.contains("field_order") ? "field_order"
                                            : "left_field_order",
             "summary");
      attach("right_field_order", "right_summary");
    }
    if (semantic(op) && sample_depth_selectivity) {
      estimate_survivals(result, context, structure, node, chosen);
      cache_edges(result, context, structure.nodes, node, chosen);
    }
    return result;
  }
  std::shared_ptr<const Choice> choice(const StatePtr &state) {
    const auto node_id = state->added->node_id;
    auto group = choice_cache.find(node_id);
    if (group != choice_cache.end()) {
      const auto found = group->second.find(state->action_json);
      if (found != group->second.end())
        return found->second;
    }
    auto action = Json::parse(state->action_json);
    auto key = digest(action);
    auto result = std::make_shared<const Choice>(Choice{
        state->action.operator_id, node_id, std::move(action), std::move(key)});
    // Candidate orders commonly share the exact immutable action. Keep this
    // worker-local catalogue bounded; retained states independently own entries
    // even when their cache references are discarded.
    if (choice_cache_size >= 32768) {
      choice_cache.clear();
      choice_cache_size = 0;
    }
    choice_cache[node_id].emplace(state->action_json, result);
    ++choice_cache_size;
    return result;
  }
  StateData build_state(const StatePtr &state, const StateData *parent) {
    StateData data;
    data.state = state;
    data.nodes = state->nodes();
    if (!state->parent) {
      data.node_by_slot.resize(node_index->ids.size());
      data.node_records.resize(node_index->ids.size());
      for (const auto &node : data.nodes) {
        const auto slot = node_index->slots.at(node->node_id);
        data.node_by_slot[slot] = node;
        data.node_records[slot] =
            find(initial, "scan:" + std::to_string(node->node_id));
      }
    } else {
      data.node_by_slot = parent->node_by_slot;
      data.node_records = parent->node_records;
      data.actions = parent->actions;
      auto chosen = choice(state);
      const auto slot = node_index->slots.at(chosen->node_id);
      data.node_by_slot[slot] = state->added;
      data.node_records[slot] = std::make_shared<const Record>(
          estimate(*parent, data, state->added, *chosen));
      data.actions.push_back(std::move(chosen));
    }
    return data;
  }
  void observe(const StatePtr &state) {
    if (states.count(state->handle_id))
      return;
    if (state->parent)
      observe(state->parent);
    else if (states.empty()) {
      auto ids = std::set<Id>(node_index->ids.begin(), node_index->ids.end());
      for (const auto &node : state->nodes())
        ids.insert(node->node_id);
      if (ids.size() != node_index->ids.size()) {
        observation_workers.clear();
        node_index = std::make_shared<const NodeIndex>(ids);
      }
    }
    const auto *parent =
        state->parent ? &states.at(state->parent->handle_id) : nullptr;
    states.emplace(state->handle_id, build_state(state, parent));
  }
  Json statistics(const StatePtr &state, bool raw) const {
    const auto &data = states.at(state->handle_id);
    Json result = Json::object();
    for (std::size_t slot = 0; slot < data.node_records.size(); ++slot)
      if (const auto &record = data.node_records[slot])
        result[std::to_string(node_index->ids[slot])] = record->json(raw);
    return result;
  }
};

ProfileEnvironment::ProfileEnvironment(const Json &config)
    : impl(std::make_unique<Impl>(config)) {}
ProfileEnvironment::~ProfileEnvironment() = default;
void ProfileEnvironment::observe(const StatePtr &state) {
  auto started = std::chrono::steady_clock::now();
  impl->observe(state);
  auto &metric = impl->metrics["profile_observe"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  ++metric.second;
}
void ProfileEnvironment::observe_batch(const std::vector<StatePtr> &states,
                                       std::size_t workers) {
  auto serial = [&] {
    for (const auto &state : states)
      observe(state);
  };
  if (workers <= 1 || impl->runtime_observed) {
    serial();
    return;
  }
  const auto started = std::chrono::steady_clock::now();
  std::vector<StatePtr> pending;
  std::vector<const StateData *> parents;
  std::unordered_set<std::uint64_t> visited;
  for (const auto &state : states) {
    if (impl->states.count(state->handle_id) ||
        !visited.insert(state->handle_id).second)
      continue;
    if (!state->parent || !impl->states.count(state->parent->handle_id)) {
      serial();
      return;
    }
    pending.push_back(state);
    parents.push_back(&impl->states.at(state->parent->handle_id));
  }
  workers = std::min({workers, pending.size(), std::size_t(32)});
  if (workers < 2 || pending.size() < workers * 16) {
    serial();
    return;
  }
  while (impl->observation_workers.size() < workers)
    impl->observation_workers.push_back(std::make_unique<Impl>(*impl));
  std::vector<std::optional<StateData>> results(pending.size());
  std::vector<std::exception_ptr> errors(pending.size());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  try {
    for (std::size_t worker = 0; worker < workers; ++worker)
      threads.emplace_back([&, worker] {
        auto &environment = *impl->observation_workers[worker];
        for (;;) {
          // Nearby candidates usually share a parent. Keep a small contiguous
          // range on one worker to reuse its input caches and avoid bouncing
          // shared evidence owners between cores for every sibling.
          const auto begin = next.fetch_add(64, std::memory_order_relaxed);
          if (begin >= pending.size())
            break;
          const auto end = std::min(begin + 64, pending.size());
          for (auto index = begin; index < end; ++index)
            try {
              results[index] =
                  environment.build_state(pending[index], parents[index]);
            } catch (...) {
              errors[index] = std::current_exception();
            }
        }
      });
  } catch (...) {
    for (auto &thread : threads)
      thread.join();
    throw;
  }
  for (auto &thread : threads)
    thread.join();
  for (std::size_t index = 0; index < pending.size(); ++index) {
    if (errors[index])
      std::rethrow_exception(errors[index]);
    impl->states.emplace(pending[index]->handle_id, std::move(*results[index]));
  }
  auto &metric = impl->metrics["profile_observe"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  metric.second += states.size();
}
void ProfileEnvironment::refresh(const std::vector<StatePtr> &states) {
  auto started = std::chrono::steady_clock::now();
  for (const auto &state : states) {
    impl->observe(state);
    // Prepared operator facts and ancestors are immutable during search; only runtime feedback refreshes them.
    if (!impl->runtime_observed)
      continue;
    auto &data = impl->states.at(state->handle_id);
    for (const auto &choice : data.actions) {
      const auto &chosen = *choice;
      if (!semantic(impl->ops.at(chosen.operator_id)))
        continue;
      data.node_records[impl->node_index->slots.at(chosen.node_id)] =
          std::make_shared<const Record>(impl->estimate(
              data, data, impl->indexed_node(data, chosen.node_id), chosen));
    }
  }
  auto &metric = impl->metrics["profile_refresh"];
  metric.first +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  ++metric.second;
}
void ProfileEnvironment::accuracy(const StatePtr &) {}
void ProfileEnvironment::retain(const std::vector<StatePtr> &live) {
  std::unordered_set<std::uint64_t> keep;
  std::unordered_set<std::uint64_t> keep_nodes;
  for (const auto &state : live)
    for (auto current = state;
         current && keep.insert(current->handle_id).second;
         current = current->parent) {
      for (const auto &node : current->nodes())
        keep_nodes.insert(node->handle_id);
    }
  // Release independent discarded state records on the same bounded planner
  // workers. Extracting nodes first leaves the live bank stable and lets every
  // shared evidence owner outlive all references still used by another state.
  using Discarded = decltype(impl->states)::node_type;
  std::vector<Discarded> discarded;
  discarded.reserve(impl->states.size());
  for (auto state = impl->states.begin(); state != impl->states.end();)
    if (!keep.count(state->first)) {
      auto removed = state++;
      discarded.push_back(impl->states.extract(removed));
    } else
      ++state;
  auto prune_memo = [&](Impl &environment) {
    for (auto memo = environment.survival_memo.begin();
         memo != environment.survival_memo.end();)
      if (std::any_of(memo->second.inputs.begin(), memo->second.inputs.end(),
                      [&](auto id) { return !keep_nodes.count(id); }))
        memo = environment.survival_memo.erase(memo);
      else
        ++memo;
  };
  prune_memo(*impl);
  for (auto &worker : impl->observation_workers)
    prune_memo(*worker);
  const auto workers =
      std::min(impl->observation_workers.size(), discarded.size() / 256);
  if (workers < 2)
    return;
  std::atomic<std::size_t> next{0};
  auto dispose = [&] {
    for (;;) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= discarded.size())
        return;
      discarded[index] = Discarded{};
    }
  };
  std::vector<std::thread> threads;
  try {
    for (std::size_t i = 1; i < workers; ++i)
      threads.emplace_back(dispose);
  } catch (...) {
  }
  dispose();
  for (auto &thread : threads)
    thread.join();
}
Json ProfileEnvironment::statistics(const StatePtr &state) const {
  return impl->statistics(state, true);
}
Json ProfileEnvironment::full_statistics(const StatePtr &state) const {
  return impl->statistics(state, false);
}
std::map<Id, std::shared_ptr<const Json>>
ProfileEnvironment::statistic_blocks(const StatePtr &state) const {
  std::map<Id, Value> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->scalar();
  return result;
}
std::map<Id, std::shared_ptr<const Json>>
ProfileEnvironment::feature_blocks(const StatePtr &state) const {
  std::map<Id, Value> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->features();
  return result;
}
const Json *StatisticView::find(const std::string &key) const {
  if (!fields)
    return nullptr;
  const auto &values = *static_cast<const FieldMap *>(fields);
  const auto found = values.find(key);
  return found == values.end() ? nullptr : found->second.get();
}
const Json &StatisticView::get(const std::string &key) const {
  static const Json missing = nullptr;
  const auto value = find(key);
  return value ? *value : missing;
}
std::map<Id, StatisticView>
ProfileEnvironment::feature_views(const StatePtr &state) const {
  std::map<Id, StatisticView> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result.emplace(impl->node_index->ids[slot],
                     StatisticView{record.get(), &record->fields});
  return result;
}
StatisticView ProfileEnvironment::feature_view(const StatePtr &state,
                                               Id node_id) const {
  const auto &record =
      impl->node_record(impl->states.at(state->handle_id), node_id);
  return record ? StatisticView{record.get(), &record->fields}
                : StatisticView{};
}
std::map<Id, std::map<std::string, std::shared_ptr<const Json>>>
ProfileEnvironment::statistic_fields(const StatePtr &state) const {
  std::map<Id, std::map<std::string, Value>> result;
  const auto &records = impl->states.at(state->handle_id).node_records;
  for (std::size_t slot = 0; slot < records.size(); ++slot)
    if (const auto &record = records[slot])
      result[impl->node_index->ids[slot]] = record->fields;
  return result;
}
Json ProfileEnvironment::cache(const StatePtr &) const {
  Json profile = Json::object(), history = Json::array();
  profile["remaining_selectivity"] = impl->remaining_selectivity_;
  for (const auto *key :
       {"models", "accuracy", "task_accuracy", "request_profiles",
        "task_request_profiles", "relational_profiles", "constraints",
        "system_prefix", "operator_specifications", "cost_seconds_per_unit",
        "operator_time_calibration"})
    if (impl->profile_.contains(key))
      profile[key] = impl->profile_.at(key);
  for (const auto &source : impl->history_) {
    Json row = Json::object();
    for (auto field = source.begin(); field != source.end(); ++field)
      if (field.key() != "summary")
        row[field.key()] = field.value();
    history.push_back(std::move(row));
  }
  return {{"profile", std::move(profile)},
          {"history", std::move(history)},
          {"version", 1}};
}
Json ProfileEnvironment::time_calibration() const {
  return impl->profile_.value("operator_time_calibration", Json::object());
}
Json ProfileEnvironment::profile() const {
  return impl->imported->value("prepared_statistics", false) ? impl->profile_
                                                             : Json::object();
}
Json ProfileEnvironment::history() const { return impl->history_; }
Json ProfileEnvironment::calls() const {
  return impl->imported->value("sampling_calls", Json::object());
}
Json ProfileEnvironment::timings() const {
  Json result = impl->imported->value("sampling_timings", Json::object());
  for (const auto &metric : impl->metrics)
    result[metric.first] = {{"seconds", metric.second.first},
                            {"calls", metric.second.second}};
  return result;
}
Json ProfileEnvironment::config() const { return *impl->imported; }
void ProfileEnvironment::apply_runtime(const StatePtr &state, Id operator_id,
                                       const Json &entry,
                                       const Json &observation) {
  impl->runtime_observed = true;
  impl->observation_workers.clear();
  impl->survival_memo.clear();
  impl->observe(state);
  auto &data = impl->states.at(state->handle_id);
  auto action = std::find_if(
      data.actions.begin(), data.actions.end(),
      [&](const auto &value) { return value->operator_id == operator_id; });
  if (action == data.actions.end())
    throw std::invalid_argument("Runtime observation has no selected operator");
  Record info = *impl->node_record(data, (*action)->node_id);
  auto completed = entry.at("completed").get<double>();
  if (entry.at("status") == "DONE") {
    info.set("input_rows", entry.at("completed"));
    info.set("output_rows", entry.at("emitted"));
    if (completed)
      info.set("selectivity", entry.at("emitted").get<double>() / completed);
  } else {
    if (!observation.is_null() && !observation.empty()) {
      info.set("selectivity", observation.at("selectivity"));
      info.set(
          "effective_samples",
          observation.value("effective_samples", observation.at("completed")));
      auto reliability = info.get("reliability").is_object()
                             ? info.get("reliability")
                             : Json::object();
      auto count =
          observation.value("effective_samples", observation.at("completed"))
              .get<double>();
      reliability["selectivity"] = count / (count + 1);
      info.set("reliability", std::move(reliability));
    }
    if (completed && !info.get("input_rows").is_null()) {
      auto incoming = std::max(completed, info.get("input_rows").get<double>());
      info.set("input_rows", incoming);
      info.set("output_rows",
               info.get("selectivity").is_null()
                   ? Json(nullptr)
                   : Json(entry.at("emitted").get<double>() +
                          (incoming - completed) *
                              info.get("selectivity").get<double>()));
    }
  }
  // Semantic selectivity is shared across implementations. Runtime evidence
  // replaces that estimate too; leaving it unchanged would make the value
  // encoder ignore the newly observed selectivity.
  if (!info.get("selectivity").is_null() &&
      (entry.at("status") == "DONE" || !observation.is_null())) {
    info.set("estimated_selectivity", info.get("selectivity"));
    auto valid =
        info.get("valid").is_object() ? info.get("valid") : Json::object();
    valid["selectivity"] = true;
    info.set("valid", std::move(valid));
  }
  if (observation.is_object() && observation.contains("input_statistics")) {
    const auto &actual = observation.at("input_statistics");
    const auto &summaries = actual.at("input_field_summaries");
    const auto &chosen = **action;
    const auto &op = impl->ops.at(operator_id);
    const auto &layout = chosen.action.at("layout");
    auto attach = [&](const char *order, const char *slot) {
      const auto names = layout.value(order, Json::array());
      if (names.empty()) return;
      auto key = names[0].get<std::string>();
      for (const auto &field : input_schema(op))
        if (field.at("name") == names[0]) key = field.value("value_key", key);
      if (summaries.contains(key)) info.set(slot, summaries.at(key).at("summary"));
    };
    attach(layout.contains("field_order") ? "field_order" : "left_field_order", "summary");
    attach("right_field_order", "right_summary");
    const bool pairs = kind(op).find("JOIN") != std::string::npos;
    info.set("input_field_statistics", actual.at("input_field_statistics"));
    info.set("input_field_summaries", summaries);
    info.set("pair_input", pairs);
    info.set("pair_request_summary", pairs);
    info.set("alpha", 1.0);
    info.set("right_alpha", 1.0);
    for (const auto *key : {"cache_field_origin", "pre", "l_existing_features"}) info.erase(key);
    info.set("runtime_statistics_basis", "actual consumed execution HTTP inputs");
    impl->cache_edges(info, data, data.nodes, impl->indexed_node(data, chosen.node_id), chosen);
    auto actual_valid = info.get("valid").is_object() ? info.get("valid") : Json::object();
    actual_valid["l_self"] = !info.get("l_self").is_null();
    info.set("valid", std::move(actual_valid));
    auto actual_reliability = info.get("reliability").is_object() ? info.get("reliability") : Json::object();
    const auto requests = actual.at("observed_requests").get<double>();
    actual_reliability["l_self"] = requests / (requests + 1.);
    info.set("reliability", std::move(actual_reliability));
  }
  data.node_records[impl->node_index->slots.at((*action)->node_id)] =
      std::make_shared<const Record>(std::move(info));
}
}
