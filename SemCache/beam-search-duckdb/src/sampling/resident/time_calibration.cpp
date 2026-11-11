#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace semcache::sampling {
void NativeSampler::Impl::record_selection_time(
    const Json &op, const SampleInputs &inputs, const Json &action,
    const Json *pairs, const Json &observation, const Json &progress) {
  auto calibration = profile_.find("operator_time_calibration");
  if (calibration == profile_.end() || !calibration->is_object() ||
      calibration->value("scope", std::string()) != "workload" ||
      !calibration->value("frozen", false) ||
      observation.value("failed", false))
    return;
  const auto seconds = observation.value("execution_seconds", Json());
  // A predicate-ledger hit may enter the backend without executing SQL.
  if (!finite_number(seconds) || seconds.get<double>() <= 0. || inputs.empty())
    return;
  const bool semantic = !action.is_null();
  if (semantic)
    return; // Semantic costs come only from the global HTTP response mean.
  const auto kind = op.value("kind", op.value("type", std::string()));
  double units = 0.;
  if (semantic) {
    double expected = pairs ? double(pairs->size()) : 1.;
    if (!pairs)
      for (const auto &input : inputs)
        expected *= row_count(input);
    const auto completed = progress.at("completed_inputs").get<std::size_t>();
    if (progress.value("budget_exhausted", false) ||
        double(completed) != expected)
      return;
    units = double(completed);
  } else if (op.contains("sampling_passthrough_input")) {
    const auto side =
        inputs.size() == 1
            ? std::size_t(0)
            : op.at("sampling_passthrough_input").get<std::size_t>();
    units = row_count(inputs[side]);
  } else if (inputs.size() > 1 && kind != "CROSS_PRODUCT") {
    for (const auto &input : inputs)
      units += row_count(input);
  } else {
    units = 1.;
    for (const auto &input : inputs)
      units *= row_count(input);
  }
  // Row-local Filter execution excludes predicate-cache hits from both the
  // timed SQL and its input count.
  if (const auto timed = observation.find("timed_input_units");
      timed != observation.end()) {
    if (!finite_number(*timed) || timed->get<double>() < 0.)
      throw std::invalid_argument("Invalid timed selection input units");
    units = timed->get<double>();
  }
  if (!(units > 0.) || !std::isfinite(units))
    return;
  const auto model =
      semantic ? action.at("model_id").get<std::string>() : std::string();
  auto category = calibration->find(semantic ? "semantic" : "relational");
  Json *bank = category != calibration->end() && category->is_object()
                   ? &*category
                   : nullptr;
  if (semantic && bank) {
    auto implementation = bank->find(model);
    bank = implementation != bank->end() && implementation->is_object()
               ? &*implementation
               : nullptr;
  }
  Json *record = nullptr;
  if (bank) {
    auto found = bank->find(kind);
    if (found != bank->end() && found->is_object())
      record = &*found;
  }
  const auto label = kind + (model.empty() ? std::string() : " model " + model);
  if (!record)
    throw std::invalid_argument(
        "Missing frozen workload time calibration for selection update: " +
        label);
  auto nonnegative = [&](const char *field) {
    const auto value = record->value(field, Json());
    if (!finite_number(value) || value.get<double>() < 0.)
      throw std::invalid_argument("Invalid frozen workload time calibration " +
                                  label + " field " + field);
    return value.get<double>();
  };
  double total_seconds, total_units;
  const auto initial_rate = nonnegative("seconds_per_input");
  const bool initialized = record->contains("timed_seconds_total") ||
                           record->contains("timed_input_units_total");
  if (initialized) {
    total_seconds = nonnegative("timed_seconds_total");
    total_units = nonnegative("timed_input_units_total");
  } else {
    total_seconds = nonnegative("seconds");
    total_units = nonnegative("input_units") * nonnegative("repeats");
  }
  const auto selection_seconds = record->contains("selection_seconds")
                                     ? nonnegative("selection_seconds")
                                     : 0.;
  const auto selection_units = record->contains("selection_input_units")
                                   ? nonnegative("selection_input_units")
                                   : 0.;
  const auto measurement_count =
      initialized ? record->value("selection_measurements", Json(0)) : Json(0);
  if (!measurement_count.is_number_integer() || measurement_count < 0 ||
      (measurement_count.is_number_unsigned() &&
       measurement_count.get<std::uint64_t>() >
           std::numeric_limits<std::size_t>::max()))
    throw std::invalid_argument(
        "Invalid frozen workload selection measurement count for " + label);
  const auto measurements = measurement_count.get<std::size_t>();
  const auto next_seconds = total_seconds + seconds.get<double>();
  const auto next_units = total_units + units;
  const auto next_selection_seconds = selection_seconds + seconds.get<double>();
  const auto next_selection_units = selection_units + units;
  if (!(total_units > 0.) || !std::isfinite(total_units) ||
      !std::isfinite(next_seconds) || !std::isfinite(next_units) ||
      !std::isfinite(next_selection_seconds) ||
      !std::isfinite(next_selection_units) ||
      measurements == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "Invalid frozen workload time calibration totals for " + label);
  // Preserve the initial measurement and snapshot fields. Only the cumulative
  // coefficients follow real, complete selection stages between queries.
  if (!initialized)
    (*record)["initial_seconds_per_input"] = initial_rate;
  (*record)["timed_seconds_total"] = next_seconds;
  (*record)["timed_input_units_total"] = next_units;
  (*record)["selection_seconds"] = next_selection_seconds;
  (*record)["selection_input_units"] = next_selection_units;
  (*record)["selection_measurements"] = measurements + 1;
  (*record)["seconds_per_input"] = next_seconds / next_units;
}

void NativeSampler::Impl::calibrate_time(const StatePtr &state) {
  if (!state)
    throw std::invalid_argument("Operator time calibration requires a state");
  const auto config =
      imported.value("operator_time_calibration", Json::object());
  const auto limit = config.value("sample_rows", std::size_t(10));
  const auto repeats = config.value("relational_repeats", std::size_t(3));
  if (!limit || !repeats)
    throw std::invalid_argument(
        "Relational time calibration budgets must be positive");
  const auto previous =
      profile_.value("operator_time_calibration", Json::object());
  Json evidence = {
      {"protocol", 2},
      {"basis", "global mean semantic response time; relational stage time"},
      {"semantic_http_requests", 0},
      {"sample_rows", limit},
      {"semantic", Json::object()},
      {"relational", Json::object()},
      {"unmaterialized_nodes", Json::array()}};
  const bool frozen = previous.value("scope", std::string()) == "workload" &&
                      previous.value("frozen", false);
  if (frozen)
    evidence["relational"] = previous.value("relational", Json::object());
  const auto virtual_sources =
      profile_.value("cardinality_only_sources", Json::object());
  std::map<std::string, std::vector<NodePtr>> relational;
  for (const auto &node : state->nodes()) {
    const auto op = node->operator_id ? execution_op(state, node)
                                      : Json{{"operator_id", node->node_id},
                                             {"kind", "SCAN"},
                                             {"sql", "SELECT * FROM $INPUT"}};
    const auto kind = op.value("kind", op.value("type", std::string()));
    if (op.value("sampling_cardinality_only", false) ||
        (!node->operator_id &&
         virtual_sources.contains(std::to_string(node->node_id)))) {
      evidence["unmaterialized_nodes"].push_back(
          Json{{"node_id", node->node_id},
               {"seconds_per_input", nullptr},
               {"basis", "unmaterialized correlated terminal stage"}});
      continue;
    }
    if (semantic_operator(op)) {
      for (const auto &model : models(op)) {
        if (profile_.contains("fixed_model_id") &&
            profile_.at("fixed_model_id") != model)
          continue;
        const auto &global = profile_statistic("request_profiles", model, kind);
        evidence["semantic"][model][kind] = {
            {"seconds_per_input",
             global.is_object() ? global.value("seconds_per_request", Json())
                                : Json()},
            {"samples", global.is_object()
                            ? global.value("samples", std::size_t(0))
                            : std::size_t(0)},
            {"http_requests", 0},
            {"basis",
             "reused global model/kind mean from observed HTTP responses"}};
      }
    } else if (!frozen) {
      relational[kind].push_back(node);
    }
  }
  std::shared_ptr<NativeSQLBackend> source;
  std::mt19937_64 random(config.value("seed", std::uint64_t(42)));
  const auto &data = states.at(state->handle_id);
  for (const auto &[kind, choices] : relational) {
    const auto node = choices[std::uniform_int_distribution<std::size_t>(
        0, choices.size() - 1)(random)];
    Json op = node->operator_id ? execution_op(state, node)
                                : Json{{"operator_id", node->node_id},
                                       {"kind", "SCAN"},
                                       {"sql", "SELECT * FROM $INPUT"}};
    const auto &observation = data.observations.at(node->node_id);
    // Reuse prepared independent inputs. Reconstructing them solely for timing
    // could otherwise invoke an additional semantic projection producer.
    OwnedInputs inputs = node->operator_id ? observation->core->inputs
                                           : OwnedInputs{observation->rows};
    if (op.contains("sampling_passthrough_input")) {
      auto sql = op.value("sql", op.value("sampling_sql", std::string()));
      const auto side = op.at("sampling_passthrough_input").get<std::size_t>();
      // Observations retain both child slots for lineage. This SQL consumes
      // only the selected CTE branch, so collapse its inputs along with SQL.
      if (inputs.size() > 1)
        inputs = OwnedInputs{inputs.at(side)};
      const std::string token = side == 0 ? "$LEFT" : "$RIGHT";
      for (auto pos = sql.find(token); pos != std::string::npos;
           pos = sql.find(token, pos + 6))
        sql.replace(pos, token.size(), "$INPUT");
      op["sql"] = std::move(sql);
      if (op.contains("input_columns"))
        op["input_columns"] = Json::array({op.at("input_columns").at(side)});
    }
    const auto left_limit =
        inputs.size() == 2
            ? std::max(std::size_t(1), std::size_t(std::sqrt(double(limit))))
            : limit;
    for (std::size_t side = 0; side < inputs.size(); ++side) {
      const auto count = side
                             ? std::max(std::size_t(1),
                                        limit / std::max(std::size_t(1),
                                                         row_count(*inputs[0])))
                             : left_limit;
      resident_detail::JoinOutputStream stream;
      inputs[side] = std::make_shared<const Json>(
          stream.take(*inputs[side], count, random()));
    }
    double units = inputs.size() > 1 && kind != "CROSS_PRODUCT" ? 0. : 1.;
    for (const auto &input : inputs)
      if (inputs.size() > 1 && kind != "CROSS_PRODUCT")
        units += row_count(*input);
      else
        units *= row_count(*input);
    if (inputs.empty())
      units = 0.;
    Json record = {{"operator_id", op.at("operator_id")},
                   {"node_id", node->node_id},
                   {"input_source", "reused_independent_base"},
                   {"input_units", units},
                   {"seconds_per_input", nullptr},
                   {"repeats", 0},
                   {"http_requests", 0}};
    if (units > 0. && backend) {
      if (!source)
        source = backend->fork_calibration(config);
      op["sampling_row_local"] = false;
      op["calibration_purpose"] = "operator_time_calibration";
      if (op.value("join_type", std::string()) == "LEFT")
        op["calibration_discard_output"] = true;
      double seconds = 0.;
      for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        try {
          source->execute(op, inputs, nullptr);
          seconds += source->observations()
                         .back()
                         .at("execution_seconds")
                         .get<double>();
        } catch (const std::exception &error) {
          harvest(source.get());
          throw std::runtime_error("Relational time calibration " + kind +
                                   "#" + std::to_string(node->node_id) +
                                   ": " + error.what());
        }
        harvest(source.get());
      }
      record["seconds_per_input"] = seconds / (units * repeats);
      record["seconds"] = seconds;
      record["repeats"] = repeats;
    }
    evidence["relational"][kind] = std::move(record);
  }
  counters["calibration_model_requests"] = 0;
  profile_["operator_time_calibration"] = std::move(evidence);
}
}
