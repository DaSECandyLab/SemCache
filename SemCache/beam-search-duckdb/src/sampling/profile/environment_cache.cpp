#include "environment_internal.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace semcache::sampling {
namespace profile_detail {

const Json &Record::get(const std::string &key) const {
  static const Json missing = nullptr;
  auto found = fields.find(key);
  return found == fields.end() ? missing : *found->second;
}

Value Record::reference(const std::string &key) const {
  auto found = fields.find(key);
  return found == fields.end() ? Value{} : found->second;
}

void Record::set(const std::string &key, Json value) {
  scalar_statistics.reset();
  feature_statistics.reset();
  fields[key] = std::make_shared<const Json>(std::move(value));
}

void Record::share(const std::string &key, Value value) {
  if (value) {
    scalar_statistics.reset();
    feature_statistics.reset();
    fields[key] = std::move(value);
  }
}

void Record::merge(const Record &other) {
  scalar_statistics.reset();
  feature_statistics.reset();
  for (const auto &field : other.fields)
    fields[field.first] = field.second;
}

void Record::erase(const std::string &key) {
  scalar_statistics.reset();
  feature_statistics.reset();
  fields.erase(key);
}

Json Record::json(bool raw) const {
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

Value Record::scalar() const {
  if (!scalar_statistics)
    scalar_statistics = std::make_shared<const Json>(json(true));
  return scalar_statistics;
}

Json Record::feature_json() const {
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

Value Record::features() const {
  if (!feature_statistics)
    feature_statistics = std::make_shared<const Json>(feature_json());
  return feature_statistics;
}

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
bool number(const Json &value, bool nonnegative) {
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

} // namespace profile_detail

using namespace profile_detail;

ProfileEnvironment::Impl::Impl(const Json &config)
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
  history_owner = std::make_shared<const Json>(history_);
  for (const auto &source : *history_owner)
    sources.push_back(import_record(history_owner, source));
  index_sources();
}

ProfileEnvironment::Impl::Impl(const Impl &source)
    : imported(source.imported), profile_(source.profile_),
      query(source.query), fields(source.fields),
      history_(source.history_), history_owner(source.history_owner),
      remaining_selectivity_(source.remaining_selectivity_), ops(source.ops),
      logical(source.logical), node_index(source.node_index),
      initial(source.initial), node_priors(source.node_priors),
      operator_priors(source.operator_priors),
      prefix_cooccurrences(source.prefix_cooccurrences),
      sample_depth_selectivity(source.sample_depth_selectivity),
      sources(source.sources) {
  index_sources();
}

void ProfileEnvironment::Impl::index_sources() {
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

std::set<Id> ProfileEnvironment::Impl::descendants(const std::vector<NodePtr> &nodes, Id target) const {
  std::unordered_map<Id, const PlanNode *> by_id;
  by_id.reserve(nodes.size() * 2 + 1);
  const PlanNode *root = nullptr;
  for (const auto &node : nodes) {
    by_id.emplace(node->node_id, node.get());
    if (node->node_id == target)
      root = node.get();
  }
  if (!root)
    throw std::out_of_range("Missing target node in descendants");
  std::set<Id> result;
  Ids pending = root->children;
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!result.insert(id).second)
      continue;
    const auto found = by_id.find(id);
    if (found != by_id.end())
      pending.insert(pending.end(), found->second->children.begin(),
                     found->second->children.end());
  }
  return result;
}

const NodePtr &ProfileEnvironment::Impl::indexed_node(const StateData &data, Id id) const {
  const auto &node = data.node_by_slot.at(node_index->slots.at(id));
  if (!node)
    throw std::out_of_range("Missing state node");
  return node;
}

const Evidence &ProfileEnvironment::Impl::node_record(const StateData &data, Id id) const {
  static const Evidence missing;
  const auto slot = node_index->slots.find(id);
  return slot == node_index->slots.end() ? missing
                                         : data.node_records.at(slot->second);
}

const Record &ProfileEnvironment::Impl::child(const StateData &data, const NodePtr &node) const {
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

CrossDependence ProfileEnvironment::Impl::cache_dependence(const Record &source, const Record &target) const {
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

void ProfileEnvironment::Impl::cache_edges(Record &result, const StateData &context,
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
    const auto key = std::make_pair(chosen.operator_id, chosen.layout_key);
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
  const Json cross_requests =
      (right && sides.is_array() && sides.size() == 2 && number(sides[0]) &&
       number(sides[1]))
          ? ((sides[0].get<double>() <= 0. || sides[1].get<double>() <= 0.)
                 ? Json(0.)
                 : Json(sides[0].get<double>() *
                        std::max(1., sides[1].get<double>())))
          : requests;
  auto edge = [&](const Record &source, const Json &source_alpha,
                  const CrossDependence &dependence = CrossDependence{}) {
    auto bank = source.reference("summary");
    if (!bank || bank->is_null() || !compatible(*bank, *summary))
      return Json(nullptr);
    auto value = reuse.cross(bank, summary, source_alpha, alpha, dependence,
                             cross_requests);
    if (!value.is_null() && right && requests.is_null())
      value = right_count > 0
                  ? Json(value.get<double>() / std::max(1., right_count))
                  : Json(0.);
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
  const auto &target_model = chosen.action.at("model_id");
  bool has_same_model_prior = false;
  for (const auto &action_ptr : context.actions) {
    if (semantic(ops.at(action_ptr->operator_id)) &&
        action_ptr->action.at("model_id") == target_model) {
      has_same_model_prior = true;
      break;
    }
  }
  if (has_same_model_prior) {
    auto ancestors = descendants(nodes, node->node_id);
    for (std::size_t index = 0; index < context.actions.size(); ++index) {
      const auto &old = *context.actions[index];
      if (!semantic(ops.at(old.operator_id)) ||
          old.action.at("model_id") != target_model ||
          !ancestors.count(old.node_id))
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
  }
  result.set("l_existing", std::move(existing));
  result.set("l_previous", std::move(previous));
}

bool ProfileEnvironment::Impl::has_field(const NodePtr &node, const std::string &name) const {
  for (const auto id : node->fields) {
    const auto &field = fields.at(id);
    if ((field.is_string() && field.get<std::string>() == name) ||
        (field.is_object() && field.value("name", std::string()) == name))
      return true;
  }
  return false;
}

} // namespace semcache::sampling
