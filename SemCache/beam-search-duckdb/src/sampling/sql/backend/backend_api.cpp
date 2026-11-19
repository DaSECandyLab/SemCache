#include "backend_internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace semcache::sampling {

using namespace sql_detail;

NativeSQLBackend::NativeSQLBackend(const Json &config)
    : impl(std::make_unique<Impl>(config)) {}
NativeSQLBackend::~NativeSQLBackend() = default;
Json NativeSQLBackend::load(const std::string &relation,
                            const std::string &sql) {
  impl->relational_estimates.clear();
  impl->estimate_order.clear();
  Json output = Json::array();
  for (const auto &record :
       impl->query("SELECT semantic_sample_json(t) FROM (" + sql + ") t")) {
    auto values = Json::parse(record[0].get<std::string>());
    auto id = values.value("__sample_id", Json());
    if (id.is_null())
      throw std::invalid_argument(
          "Base SQL must expose a stable non-NULL __sample_id");
    auto identity = id.is_string() ? id.get<std::string>() : id.dump();
    values.erase("__sample_id");
    for (auto it = values.begin(); it != values.end(); ++it)
      it.value() = impl->value(it.key(), std::move(it.value()));
    output.push_back(
        Json{{"id", identity},
             {"values", std::move(values)},
             {"lineage", Json::array({Json::array({relation, id})})}});
  }
  return output;
}
std::shared_ptr<NativeSQLBackend> NativeSQLBackend::fork_semantic() const {
  Json config = impl->config;
  // Stage SQL can retain macros, types or subqueries from the source catalog
  // and session setup. Preserve that environment; only transport telemetry
  // and connection-local parallelism differ. Workers are reused across layers.
  config.erase("telemetry");
  config["parallelism"] = 1;
  auto result = std::make_shared<NativeSQLBackend>(config);
  return result;
}
std::shared_ptr<NativeSQLBackend>
NativeSQLBackend::fork_calibration(const Json &settings) const {
  Json config = impl->config;
  config.erase("telemetry");
  config["parallelism"] = settings.value("parallelism", std::size_t(1));
  config["cpu_workers"] = settings.value("parallelism", std::size_t(1));
  config["semantic_unrestricted"] = false;
  config["setup"] =
      config.value("setup", std::string()) +
      "; SET semantic_response_cache='{}'; SET semantic_request_reuse=false; "
      "SET semantic_max_concurrent_requests_per_model=" +
      std::to_string(
          settings.value("max_concurrent_requests_per_model", std::size_t(1))) +
      ";";
  return std::make_shared<NativeSQLBackend>(config);
}
std::string
NativeSQLBackend::scheduling_key(const std::string &model) const {
  const auto &entry = impl->config.at("models").at(model);
  // The semantic extension validated this registry when the backend opened.
  // Match ModelRegistry's normalization, including explicit default HTTPS
  // ports and a trailing slash on the optional path prefix.
  static const std::regex pattern(
      "^(http|https)://([A-Za-z0-9][A-Za-z0-9.-]*)(:([0-9]{1,5}))?(/[A-Za-z0-9/_-]*)?$");
  std::smatch match;
  const auto url = entry.at("base_url").get<std::string>();
  if (!std::regex_match(url, match, pattern))
    throw std::invalid_argument("Invalid calibration model endpoint");
  auto prefix = match[5].str();
  if (!prefix.empty() && prefix.back() == '/')
    prefix.pop_back();
  return Json::array({match[1].str() == "https", match[2].str(),
                      match[4].matched ? std::stoi(match[4].str()) : 443,
                      prefix, entry.at("model")})
      .dump();
}
std::size_t NativeSQLBackend::sample_workers_peak() const {
  return impl->stage_peak;
}
Json NativeSQLBackend::execute(const Json &op, const SampleInputs &inputs,
                               const Json &action) {
  const auto started = std::chrono::steady_clock::now();
  const auto first = impl->events.size();
  impl->stage_seconds = 0.;
  impl->timed_input_units.reset();
  impl->selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  auto observe = [&](bool failed) {
    impl->collect_events();
    impl->selection_progress = {
        {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
    Json sizes = Json::array(), events = Json::array();
    for (const auto &rows : inputs)
      sizes.push_back(row_count(rows));
    for (std::size_t i = first; i < impl->events.size(); ++i) {
      const auto &event = impl->events[i];
      events.push_back(event);
      // A retransmission repeats a logical request but is a billed HTTP call,
      // so it counts against the request budget like a first attempt.
      const auto name = event.value("event", std::string());
      if (name == "sampling_llm_submit" || name == "sampling_llm_retransmit")
        impl->selection_progress["requests"] =
            impl->selection_progress["requests"].get<std::size_t>() + 1;
      if (event.value("event", std::string()) == "sampling_stage_progress") {
        impl->selection_progress["completed_inputs"] =
            event.at("completed_inputs");
        impl->selection_progress["budget_exhausted"] =
            event.at("budget_exhausted");
      }
    }
    Json schema = Json::array();
    for (const auto &field : op.value("input_field_schema", Json::array()))
      schema.push_back(
          Json::array({field.at("name"), field.value("type", Json())}));
    Json logical_task = op.contains("task")
                            ? Json(digest(Json::array(
                                  {op.at("kind"), op.at("task"), schema,
                                   op.value("output_schema", Json::array())})))
                            : Json();
    impl->observations.push_back(
        Json{{"operator_id", op.value("operator_id", Json())},
             {"kind", op.value("kind", op.value("type", std::string()))},
             {"task_key", logical_task},
             {"model_id", action.is_null() ? Json() : action.at("model_id")},
             {"inputs", sizes},
             {"seconds", std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count()},
             {"execution_seconds", impl->stage_seconds},
             {"purpose", op.value("calibration_purpose", std::string())},
             {"output_discarded", op.value("calibration_discard_output", false)},
             {"events", events},
             {"failed", failed}});
    if (impl->timed_input_units)
      impl->observations.back()["timed_input_units"] = *impl->timed_input_units;
  };
  try {
    std::map<std::string, std::string> ordinal_ids, original_ids;
    struct IdentityGuard {
      Impl &backend;
      ~IdentityGuard() {
        backend.sampling_ids = nullptr;
        backend.sampling_ids_key.clear();
      }
    } identities{*impl};
    // Accuracy evaluates one ordinary Join pair through two singleton sides.
    // Its only pair already has a known order, so retain that original path
    // and its row-level prompt instead of applying selected-pair ordinals.
    const auto kind = upper(op.value("kind", op.value("type", std::string())));
    const bool singleton_join =
        (kind == "SEMANTIC_JOIN" || kind == "SEMANTIC_NESTED_LOOP_JOIN") &&
        inputs.size() == 2 && inputs[0].is_array() && inputs[1].is_array() &&
        inputs[0].size() == 1 && inputs[1].size() == 1;
    if (!action.is_null() && action.contains("llm_request_limit") &&
        !singleton_join) {
      if (inputs.size() != 1 || !inputs[0].is_array())
        throw std::invalid_argument(
            "Bounded semantic stages require flat ordered inputs");
      std::size_t ordinal = 0;
      Json ordered_ids = Json::array();
      for (const auto &row : inputs[0]) {
        if (!row.at("id").is_string())
          throw std::invalid_argument(
              "Sample identities must be unique non-NULL strings");
        const auto &original = row.at("id").get_ref<const std::string &>();
        auto id = std::to_string(ordinal++);
        id.insert(0, 20 - id.size(), '0');
        if (!ordinal_ids.emplace(original, id).second)
          throw std::invalid_argument(
              "Sample identities must be unique non-NULL strings");
        original_ids.emplace(id, original);
        ordered_ids.push_back(original);
      }
      impl->sampling_ids = &ordinal_ids;
      impl->sampling_ids_key = digest(ordered_ids);
    }
    auto output = impl->run(op, inputs, action,
                            op.value("_preserve_sample_inputs", false));
    if (!original_ids.empty())
      for (auto &row : output)
        row["id"] = original_ids.at(row.at("id").get<std::string>());
    observe(false);
    return output;
  } catch (...) {
    try {
      observe(true);
    } catch (...) {
    }
    throw;
  }
}
std::vector<Json> NativeSQLBackend::prepare_inputs(const Json &op,
                                                   const SampleInputs &inputs) {
  std::vector<Json> output;
  for (std::size_t side = 0; side < inputs.size(); ++side) {
    std::string expressions;
    for (const auto &field : op.value("input_field_schema", Json::array())) {
      auto key = field.value("value_key", std::string());
      if (key.rfind("__semantic_input:", 0) != 0)
        continue;
      bool right = false;
      for (const auto &name : op.value("right_input_fields", Json::array()))
        right |= name == field.at("name");
      if (inputs.size() == 2 && right != (side == 1))
        continue;
      if (!expressions.empty())
        expressions += ',';
      expressions += field.at("sample_expression").get<std::string>() + " AS " +
                     identifier(key);
      impl->field_type(key, field.value("type", std::string("VARCHAR")));
    }
    if (expressions.empty() || !row_count(inputs[side])) {
      output.push_back(inputs[side]);
      continue;
    }
    Json stage{{"operator_id", op.value("operator_id", Json(0))},
               {"kind", "PROJECTION"},
               {"sql", "SELECT *," + expressions + " FROM $INPUT"},
               {"_preserve_sample_inputs", true}};
    output.push_back(impl->run(stage, {&inputs[side]}, nullptr, true));
  }
  return output;
}
Json NativeSQLBackend::join_pairs(const Json &op, const Json &pairs,
                                  const Json &action) {
  impl->selection_progress = {
      {"completed_inputs", 0}, {"requests", 0}, {"budget_exhausted", false}};
  auto kind = upper(op.value("kind", std::string()));
  if (action.is_null()) {
    // Evaluate only the already drawn relational pairs. Equal private IDs
    // constrain the trusted binary SQL without forming a Cartesian cohort.
    if (!pairs.is_array())
      throw std::invalid_argument("Selected relational pairs must be an array");
    std::vector<Json> inputs{Json::array(), Json::array()};
    for (const auto &pair : pairs) {
      if (!pair.is_array() || pair.size() != 2)
        throw std::invalid_argument("Candidate pair must contain two rows");
      const auto id = digest(Json::array({pair[0].at("id"), pair[1].at("id")}));
      for (std::size_t side = 0; side < 2; ++side) {
        auto row = pair[side];
        row["id"] = id;
        inputs[side].push_back(std::move(row));
      }
    }
    if (pairs.empty())
      return Json::array();
    auto stage = op;
    auto sql = kind == "FILTER" ? op.value("candidate_sql", std::string())
                                : op.value("sql", op.value("sampling_sql", std::string()));
    const auto left = sql.find(" FROM $LEFT l ");
    const auto right = left == std::string::npos ? std::string::npos
                                                : sql.find(" $RIGHT r ON ", left);
    if (right == std::string::npos)
      throw std::invalid_argument("Direct relational pairs require trusted binary SQL");
    const auto predicate = right + std::string(" $RIGHT r ON ").size();
    sql = sql.substr(0, predicate) + "l.__sample_id=r.__sample_id AND (" +
          sql.substr(predicate) + ")";
    stage[kind == "FILTER" ? "candidate_sql" : "sql"] = std::move(sql);
    return execute(stage, SampleInputs(inputs), nullptr);
  }
  if ((kind != "SEMANTIC_JOIN" && kind != "SEMANTIC_NESTED_LOOP_JOIN") ||
      action.is_null())
    throw std::invalid_argument(
        "Pair batching requires a semantic Join and model");
  if (!pairs.is_array())
    throw std::invalid_argument("Selected pairs must be an array");
  if (pairs.empty())
    return Json::array();
  std::array<Json, 2> sides{{Json::array(), Json::array()}};
  for (const auto &pair : pairs) {
    if (!pair.is_array() || pair.size() != 2)
      throw std::invalid_argument("Candidate pair must contain two rows");
    for (std::size_t side = 0; side < 2; ++side)
      sides[side].push_back(pair[side]);
  }
  auto left = columns(sides[0]), right = columns(sides[1]);
  if (left.empty() || right.empty())
    throw std::invalid_argument("Candidate pairs need both field partitions");
  for (const auto &key : right)
    if (std::find(left.begin(), left.end(), key) != left.end())
      throw std::invalid_argument(
          "Candidate-pair inputs require disjoint field partitions");
  Json rows = Json::array();
  std::set<std::string> identities;
  for (const auto &pair : pairs) {
    auto identity = digest(Json::array({pair[0].at("id"), pair[1].at("id")}));
    if (!identities.insert(identity).second)
      throw std::invalid_argument("Candidate-pair identities must be unique");
    Json values = Json::object();
    for (const auto &key : left)
      values[key] = pair[0].at("values").value(key, Json());
    for (const auto &key : right)
      values[key] = pair[1].at("values").value(key, Json());
    Json row{
        {"id", identity},
        {"values", std::move(values)},
        {"lineage", merged_lineage(pair[0].at("lineage"), pair[1].at("lineage"),
                                   impl->sql_lineage_canonicalizations)}};
    if (pair[0].contains("_sample_weight_factor") ||
        pair[1].contains("_sample_weight_factor"))
      sample_factor(row, sample_factor(pair[0]) * sample_factor(pair[1]));
    merge_sample_origins(row, pair[0]);
    merge_sample_origins(row, pair[1]);
    rows.push_back(std::move(row));
  }
  auto stage = op;
  stage["pair_columns"] = Json::array({left, right});
  auto result = execute(stage, {&rows}, action);
  if (!impl->observations.empty()) {
    auto &observation = impl->observations.back();
    std::array<std::set<std::string>, 2> unique;
    for (const auto &pair : pairs)
      for (std::size_t side = 0; side < 2; ++side)
        unique[side].insert(pair[side].at("id").get<std::string>());
    observation["candidate_pair_count"] = pairs.size();
    observation["source_input_rows"] =
        Json::array({unique[0].size(), unique[1].size()});
  }
  return result;
}
Json NativeSQLBackend::sample_population(const Json &op) {
  const auto &spec = op.at("population_sample");
  const auto started = std::chrono::steady_clock::now();
  const auto records = impl->query(spec.at("probe_sql").get<std::string>());
  if (records.size() != 1 || records.at(0).size() != 4)
    throw std::runtime_error("Population sample requires four aggregate counts");
  std::array<double, 4> counts;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const auto &value = records.at(0).at(i);
    counts[i] = value.is_string() ? std::stod(value.get<std::string>())
                                : value.get<double>();
    if (!std::isfinite(counts[i]) || counts[i] < 0)
      throw std::runtime_error("Invalid relational sample count");
  }
  const auto left = counts[0], right = counts[1], n = counts[2], passed = counts[3];
  if (n > left || (left > 0 && n == 0))
    throw std::runtime_error("Relational population sample is incomplete");
  const bool join = spec.at("kind") == "join", census = n == left;
  const double denominator = n * (join ? right : 1.);
  const bool floored = denominator > 0 && passed == 0 && !census;
  const double point = denominator ? passed / denominator : 0.;
  const double rate = floored ? 1. / denominator : point;
  Json sides = Json::array({left});
  if (join)
    sides.push_back(right);
  Json sampled = {{"input_rows", left * (join ? right : 1.)},
          {"output_rows", left * (join ? right : 1.) * rate},
          {"selectivity", rate},
          {"sampled_selectivity", point},
          {"sampled_selectivity_complete", true},
          {"population_input_rows", sides},
          {"population_output_rows", left * (join ? right : 1.) * rate},
          {"population_sample_rows", n},
          {"population_sample_output_rows", passed},
          {"population_sample_census", census},
          {"population_sample_zero_floor", floored},
          {"population_cardinality_sampling", true},
          {"conditioned_relational_filters", spec.at("conditioned_filters")},
          {"effective_samples", n},
          {"uncertain", !census},
          {"reliability", {{"selectivity", census ? 1. : n / (n + 1.)}}},
          {"population_sampling_seconds", std::chrono::duration<double>(
               std::chrono::steady_clock::now() - started).count()},
          {"cardinality_basis", join
              ? "one-sided reservoir join sample against full peer with ready relational filters"
              : "reservoir sample of base-field relational predicate"}};
  Json estimated = Json::object();
  try {
    if (op.contains("population_estimate_sql") && op.contains("population_input_sql"))
      estimated = estimate_relation(op, SampleInputs{});
  } catch (const std::exception &error) {
    sampled["duckdb_estimate_error"] = error.what();
  }
  return mean_relational_estimate(sampled, estimated);
}
Json NativeSQLBackend::selection_progress() const {
  return impl->selection_progress;
}
const Json &NativeSQLBackend::events() const { return impl->events; }
Json NativeSQLBackend::drain_events() {
  auto result = std::move(impl->events);
  impl->events = Json::array();
  return result;
}
const Json &NativeSQLBackend::observations() const {
  return impl->observations;
}
Json NativeSQLBackend::drain_observations() {
  auto result = std::move(impl->observations);
  impl->observations = Json::array();
  return result;
}
Json NativeSQLBackend::diagnostics() const {
  return {
      {"relational_explains", impl->relational_explains},
      {"relational_estimate_hits", impl->relational_estimate_hits},
      {"normalized_leaf_reads", impl->normalized_leaf_reads},
      {"normalized_row_copies", impl->normalized_row_copies},
      {"sql_value_normalizations", impl->sql_value_normalizations},
      {"sql_lineage_canonicalizations", impl->sql_lineage_canonicalizations},
      {"sql_table_hits", impl->sql_table_hits},
      {"sql_table_misses", impl->sql_table_misses},
      {"sql_payload_measurements", impl->payloads.measurements()},
      {"sql_payload_hits", impl->payloads.hits()},
      {"id_only_stages", impl->id_only_stages},
      {"predicate_evaluated_rows", impl->predicate_evaluated_rows},
      {"predicate_cache_hits", impl->predicate_cache_hits}};
}
Json NativeSQLBackend::timings() const {
  Json result = Json::object();
  for (const auto &entry : impl->phase_metrics)
    result[entry.first] = {{"seconds", entry.second.seconds},
                           {"calls", entry.second.calls},
                           {"spans_ns", entry.second.spans}};
  return result;
}
void NativeSQLBackend::close() {
  try {
    impl->collect_events();
  } catch (...) {
  }
  impl->close(true);
}
}
