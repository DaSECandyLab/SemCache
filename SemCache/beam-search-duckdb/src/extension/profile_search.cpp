#include "profile_search.hpp"
#include "sampling/common/exact_integer.hpp"
#include "sampling/common/physical_work.hpp"
#include "sampling/profile/environment.hpp"
#include "sampling/statistics/statistics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <cctype>

namespace semcache {
namespace {
using sampling::Json;
std::optional<double> number(const Json &value) {
  if (!value.is_number())
    return {};
  const auto result = value.get<double>();
  return std::isfinite(result) && result >= 0 ? std::optional<double>(result)
                                              : std::nullopt;
}
ActionPriorities request_action_priorities(const Json &operators) {
  ActionPriorities priorities;
  std::map<Id, double> base_groups;
  std::map<Id, Ids> dependencies;
  std::set<Id> predicates;
  for (const auto &op : operators) {
    const auto oid = op.at("operator_id").get<Id>();
    auto kind = op.at("kind").get<std::string>();
    std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    const bool semantic = op.at("semantic").get<bool>();
    const double category = !semantic && kind == "filter" ? 0.
                            : !semantic && kind.find("projection") != kind.npos
                                ? 4.
                            : !semantic                            ? 1.
                            : kind.find("filter") != kind.npos     ? 2.
                            : kind.find("projection") != kind.npos ? 5.
                                                                   : 3.;
    priorities[oid] = {category, category, 0.};
    base_groups[oid] = category;
    dependencies[oid] = op.at("priority_dependencies").get<Ids>();
    if (kind.find("filter") != kind.npos || kind.find("join") != kind.npos ||
        kind == "cross_product")
      predicates.insert(oid);
  }
  for (auto oid : predicates) {
    Ids pending = dependencies.at(oid);
    std::set<Id> visited{oid};
    while (!pending.empty()) {
      const auto dependency = pending.back();
      pending.pop_back();
      if (!visited.insert(dependency).second)
        continue;
      priorities.at(dependency)[0] =
          std::min(priorities.at(dependency)[0], base_groups.at(oid));
      const auto &upstream = dependencies.at(dependency);
      pending.insert(pending.end(), upstream.begin(), upstream.end());
    }
  }
  return priorities;
}
std::size_t positive_size(const Json &value) {
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<std::int64_t>() <= 0))
    throw std::invalid_argument("Search budgets must be positive integers");
  const auto result = value.get<std::uint64_t>();
  if (!result || result > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Search budget exceeds native size range");
  return static_cast<std::size_t>(result);
}
void import_resource(Json &value) {
  if (value.is_object() && value.size() == 1 && value.contains("$integer") &&
      value.at("$integer").is_string()) {
    const auto text = value.at("$integer").get<std::string>();
    if (text.empty() || (text.size() > 1 && text.front() == '0') ||
        !std::all_of(text.begin(), text.end(),
                     [](char c) { return c >= '0' && c <= '9'; }))
      throw std::invalid_argument(
          "$integer requires a nonnegative decimal integer");
    value = sampling::make_integer(text);
  } else if (!value.is_null() &&
             (!number(value) || (value.is_number_float() &&
                                 value.get<double>() > 9007199254740991.)))
    throw std::invalid_argument(
        "Invalid or lossy resource number; use {$integer: decimal string} for "
        "large integers");
}
Json import_problem(const Json &config) {
  Json problem = config;
  for (auto &model : problem.at("models"))
    for (const auto *key : {"gpu_weight_bytes", "gpu_kv_reserve_bytes",
                            "cpu_weight_bytes", "cpu_kv_reserve_bytes"})
      if (model.contains(key))
        import_resource(model[key]);
  for (const auto *key : {"gpu_memory_bytes", "cpu_memory_bytes"})
    if (problem.at("constraints").contains(key))
      import_resource(problem["constraints"][key]);
  return problem;
}
ResidentPrediction request_prediction(sampling::ProfileEnvironment &environment,
                                      const std::vector<StatePtr> &states,
                                      Json *audits = nullptr) {
  ResidentPrediction out;
  const auto config = environment.config();
  const auto calibration = environment.time_calibration();
  std::map<Id, Json> operators, logical;
  for (const auto &op : config.at("operators"))
    operators.emplace(op.at("operator_id").get<Id>(), op);
  for (const auto &node : config.at("query").at("operators"))
    logical.emplace(node.at("node_id").get<Id>(), node);
  std::vector<std::optional<double>> requests;
  for (const auto &state : states) {
    const auto nodes = state->nodes();
    const auto work =
        sampling::physical_work(nodes, operators, logical, config.at("fields"));
    Json audit = {{"estimator", "calibrated_input_rows_v1"},
                  {"operators", Json::object()}};
    const auto stats = environment.statistics(state);
    double seconds = 0.;
    double semantic_requests = 0.;
    bool requests_known = true;
    bool known = true;
    std::map<Id, std::optional<double>> sizes;
    std::map<Id, Json> actions;
    for (const auto &raw : state->history()) {
      const auto action = Json::parse(raw);
      actions.emplace(action.at("operator_id").get<Id>(), action);
    }
    for (const auto &node : nodes) {
      const auto info =
          stats.value(std::to_string(node->node_id), Json::object());
      std::string kind = "SCAN", model;
      std::optional<double> units;
      if (!node->operator_id) {
        units = number(info.value(
            "output_rows",
            logical.at(node->node_id).value("estimated_rows", Json())));
      } else {
        const auto &op = operators.at(*node->operator_id);
        kind = op.value("kind", op.value("type", std::string()));
        std::transform(kind.begin(), kind.end(), kind.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        if (kind.rfind("SEMANTIC", 0) == 0)
          model =
              actions.at(*node->operator_id).at("model_id").get<std::string>();
        if (op.contains("sampling_passthrough_input"))
          units = sizes.at(node->children.at(
              op.at("sampling_passthrough_input").get<std::size_t>()));
        else if (node->children.size() > 1 && model.empty() &&
                 kind != "CROSS_PRODUCT") {
          units = 0.;
          for (const auto id : node->children)
            if (units && sizes.at(id))
              *units += *sizes.at(id);
            else
              units.reset();
        } else {
          units = 1.;
          for (const auto id : node->children)
            if (units && sizes.at(id))
              *units *= *sizes.at(id);
            else
              units.reset();
          if (!units)
            units = number(info.value("input_rows", Json()));
        }
      }
      sizes[node->node_id] = number(info.value(
          "output_rows", !node->operator_id && units ? Json(*units) : Json()));
      if (!model.empty()) {
        if (units && std::isfinite(semantic_requests + *units))
          semantic_requests += *units;
        else
          requests_known = false;
      }
      auto cost_kind = kind;
      const auto overlay = work.find(node->node_id);
      if (overlay != work.end()) {
        cost_kind = overlay->second.cost_kind;
        units = 0.;
        for (const auto child : overlay->second.input_nodes)
          if (units && sizes.at(child))
            *units += *sizes.at(child);
          else
            units.reset();
      }
      const auto bank = model.empty()
                            ? calibration.value("relational", Json::object())
                            : calibration.value("semantic", Json::object())
                                  .value(model, Json::object());
      auto rate = number(bank.value(cost_kind, Json::object())
                             .value("seconds_per_input", Json()));
      const bool cardinality_only =
          node->operator_id
              ? operators.at(*node->operator_id)
                    .value("sampling_cardinality_only", false)
              : config.at("profile")
                    .value("cardinality_only_sources", Json::object())
                    .contains(std::to_string(node->node_id));
      if (cardinality_only) {
        rate.reset();
        known = false;
      }
      bool join_rate_fallback = false;
      if (!cardinality_only && overlay != work.end() &&
          cost_kind == "COMPARISON_JOIN" && !rate) {
        rate = number(bank.value("FILTER", Json::object())
                          .value("seconds_per_input", Json()));
        join_rate_fallback = bool(rate);
      }
      if (audits)
        audit["operators"][std::to_string(node->node_id)] = {
            {"kind", kind},
            {"logical_kind", kind},
            {"cost_kind", cost_kind},
            {"model_id", model.empty() ? Json() : Json(model)},
            {"fusion_owner", overlay != work.end()
                                 ? Json(overlay->second.fusion_owner)
                                 : Json()},
            {"input_units_basis", overlay != work.end()
                                      ? overlay->second.input_units_basis
                                      : "logical operator inputs"},
            {"join_rate_fallback", join_rate_fallback},
            {"sampling_cardinality_only", cardinality_only},
            {"input_units", units ? Json(*units) : Json()},
            {"seconds_per_input", rate ? Json(*rate) : Json()},
            {"seconds", cardinality_only        ? Json()
                        : units && *units == 0. ? Json(0.)
                        : units && rate         ? Json(*units * *rate)
                                                : Json()}};
      if (units && *units == 0.)
        continue;
      if (!units || !rate || !std::isfinite(seconds + *units * *rate))
        known = false;
      else
        seconds += *units * *rate;
    }
    out.times.push_back(known ? std::optional<double>(seconds) : std::nullopt);
    requests.push_back(requests_known ? std::optional<double>(semantic_requests)
                                      : std::nullopt);
    audit["estimated_requests"] =
        requests_known ? Json(semantic_requests) : Json();
    audit["ranking_basis"] = "calibrated_input_time";
    if (audits)
      audits->push_back(std::move(audit));
    out.accuracies.push_back(std::nullopt);
    out.time_reasons.push_back(
        known ? std::nullopt
              : std::optional<std::string>(
                    "Missing per-input time calibration or input cardinality"));
  }
  const bool request_fallback =
      !config.at("profile")
           .value("cardinality_only_sources", Json::object())
           .empty() &&
      std::any_of(out.times.begin(), out.times.end(),
                  [](const auto &value) { return !value; }) &&
      std::all_of(requests.begin(), requests.end(),
                  [](const auto &value) { return bool(value); });
  if (request_fallback) {
    for (std::size_t i = 0; i < requests.size(); ++i) {
      out.scores.push_back({-*requests[i], 0.});
      if (out.time_reasons[i])
        *out.time_reasons[i] += "; ranking by estimated semantic request count";
      if (audits)
        audits->at(i)["ranking_basis"] = "semantic_request_count_fallback";
    }
    return out;
  }
  double scale = 1.;
  const bool mixed = std::any_of(out.times.begin(), out.times.end(),
                                 [](const auto &v) { return bool(v); }) &&
                     std::any_of(out.times.begin(), out.times.end(),
                                 [](const auto &v) { return !v; });
  if (mixed)
    for (const auto &value : out.times)
      scale = std::max(scale, value.value_or(0.));
  for (const auto &value : out.times)
    out.scores.push_back({value ? -*value / scale : mixed ? -2. : 0., 0.});
  return out;
}
}

sampling::Json search_profile_configuration(const sampling::Json &config) {
  const StateProblem problem(
      nlohmann::json(import_problem(config.at("problem"))));
  const auto &imported = config.at("environment");
  auto environment = std::make_shared<sampling::ProfileEnvironment>(imported);
  const auto &settings = config.at("search");
  const auto mode =
      settings.value("plan_generation", std::string("heuristic_requests"));
  if (mode != "heuristic_requests")
    throw std::invalid_argument(
        "Profile-only SQL search supports heuristic_requests; heuristic_value "
        "requires a value-model callback");
  const auto constraints =
      imported.at("profile").value("constraints", Json::object());
  if (!settings.value("a_min", Json()).is_null() ||
      !constraints.value("a_min", Json()).is_null())
    throw std::invalid_argument(
        "An accuracy constraint requires heuristic_value; profile-only "
        "heuristic_requests does not predict accuracy");
  const auto model =
      settings.value("fixed_model_id", std::string("gemma4-e2b"));
  if (model.empty())
    throw std::invalid_argument("fixed_model_id must be nonempty");
  for (const auto &op : config.at("problem").at("operators")) {
    if (!op.at("semantic").get<bool>())
      continue;
    bool found = false;
    for (const auto &text : op.at("domain")) {
      const auto action = Json::parse(text.get<std::string>());
      found |= action.value("model_id", std::string()) == model;
    }
    if (!found)
      throw std::invalid_argument("The fixed request-search model must "
                                  "implement every semantic operator");
  }
  ResidentOptions options{positive_size(settings.at("beam_width")),
                          positive_size(settings.at("max_expansions")),
                          settings.at("max_seconds").get<double>(),
                          settings.value("training", false),
                          0.,
                          std::nullopt,
                          settings.value("trace_timings", false)};
  options.mode = mode;
  options.fixed_model = model;
  options.planner_workers = positive_size(settings.value("workers", Json(1)));
  options.action_priorities =
      request_action_priorities(config.at("problem").at("operators"));
  ResidentRandom random(
      config.at("random_state").get<std::vector<std::uint64_t>>());
  NativeSearchEngine engine(problem, environment);
  const auto result =
      engine.search(options, random, {}, [&](const auto &states) {
        return request_prediction(*environment, states);
      });
  sampling::Json complete = sampling::Json::array(),
                 trace = sampling::Json::array();
  for (const auto &state : result.completed) {
    sampling::Json actions = sampling::Json::array();
    for (const auto &action : state->history())
      actions.push_back(sampling::Json::parse(action));
    complete.push_back({{"actions", actions},
                        {"roots", state->root_ids()},
                        {"remaining", state->remaining()}});
  }
  for (const auto &row : result.trace)
    trace.push_back({{"depth", row.depth},
                     {"enumerated", row.enumerated},
                     {"retained", row.retained},
                     {"complete", row.complete}});
  auto values = [](const auto &rows) {
    sampling::Json out = sampling::Json::array();
    for (const auto &value : rows)
      out.push_back(value ? sampling::Json(*value) : sampling::Json());
    return out;
  };
  Json time_audits = Json::array();
  request_prediction(*environment, result.completed, &time_audits);
  return {{"completed", complete},
          {"time_audits", time_audits},
          {"expansions", result.expansions},
          {"budget_exhausted", result.exhausted},
          {"action_budget_exhausted", result.action_budget_exhausted},
          {"trace", trace},
          {"prediction_source", "calibrated_input_rows"},
          {"plan_generation", "heuristic_requests"},
          {"objective_units", "seconds"},
          {"times", values(result.prediction.times)},
          {"accuracies", values(result.prediction.accuracies)},
          {"time_reasons", values(result.prediction.time_reasons)},
          {"selected", result.selection.selected},
          {"statuses", result.selection.statuses},
          {"constraint_satisfied",
           result.selection.constraint_satisfied
               ? sampling::Json(*result.selection.constraint_satisfied)
               : sampling::Json()}};
}
}
