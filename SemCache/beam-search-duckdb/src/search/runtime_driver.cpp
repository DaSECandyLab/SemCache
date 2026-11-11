#include "runtime_driver.hpp"
#include "grouped_pruning.hpp"
#include "../sampling/statistics/statistics.hpp"
#include <chrono>
#include <iterator>
#include <random>
#include <algorithm>

namespace semcache {
namespace {
constexpr double default_runtime_timeout_ms = 1000.;
constexpr double milliseconds_per_second = 1000.;
// Leave half of the RPC deadline for prediction, transport and result export.
constexpr double runtime_search_fraction = 0.5;

double runtime_timeout(const sampling::Json &request) {
  const auto value = request.find("timeout_ms");
  if (value == request.end() || value->is_null())
    return default_runtime_timeout_ms;
  if (!value->is_number())
    throw std::invalid_argument("timeout_ms must be a positive finite number");
  const auto timeout = value->get<double>();
  if (!std::isfinite(timeout) || timeout <= 0)
    throw std::invalid_argument("timeout_ms must be a positive finite number");
  return timeout;
}
}
sampling::Json search_runtime(const StateProblem &problem,
                              sampling::ProfileEnvironment &environment,
                              const sampling::Json &config,
                              const ResidentPredictor &predict) {
  using sampling::Json;
  const auto &request = config.at("request");
  const auto maximum = config.at("max_expansions").get<std::size_t>();
  const auto width = config.at("beam_width").get<std::size_t>();
  const double seconds =
      std::min(config.at("max_seconds").get<double>(),
               runtime_timeout(request) / milliseconds_per_second *
                   runtime_search_fraction);
  if (!width || !maximum || !std::isfinite(seconds) || seconds <= 0)
    throw std::invalid_argument("Invalid runtime search budget");
  const auto start = std::chrono::steady_clock::now();
  const auto minimum =
      config.contains("a_min") && !config.at("a_min").is_null()
          ? std::optional<double>(config.at("a_min").get<double>())
          : std::nullopt;
  const auto preference = config.value("preference", std::string("fastest"));
  const auto layer_pruning =
      config.value("layer_pruning", std::string("pareto"));
  auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
               .count() >= seconds;
  };
  std::map<Id, Json> entries;
  for (const auto &entry : request.at("state"))
    entries[entry.at("operator_id").get<Id>()] = entry;
  std::map<std::pair<Id, std::string>, Json> observations;
  std::map<Id, Json> current_selectivities;
  auto implementation = [](const Json &action) {
    return sampling::canonical(
        Json::array({action.at("model_id"), action.at("layout")}));
  };
  for (const auto &observation : request.at("observations")) {
    const auto oid = observation.at("operator_id").get<Id>();
    observations[{oid, implementation(observation.at("action"))}] = observation;
    const auto entry = entries.find(oid);
    if (entry != entries.end() &&
        observation.at("config_generation") ==
            entry->second.at("current").at("config_generation"))
      current_selectivities[oid] = observation;
  }
  std::vector<StatePtr> beam{problem.initial()}, complete;
  ResidentPrediction completion_prediction;
  environment.observe(beam.front());
  std::mt19937_64 random(config.value("seed", std::uint64_t{0}));
  const IndexSampler sample = [&](const auto &population, auto count) {
    std::vector<std::size_t> selected;
    selected.reserve(count);
    std::sample(population.begin(), population.end(),
                std::back_inserter(selected), count, random);
    return selected;
  };
  std::size_t expansions = 0;
  for (const auto &original : config.at("baseline_actions")) {
    const auto oid = original.at("operator_id").get<Id>();
    Json choices = Json::array({original});
    const auto entry = entries.find(oid);
    if (entry != entries.end()) {
      choices.clear();
      if (entry->second.at("status") == "DONE") {
        auto action = original;
        for (auto it = entry->second.at("current").begin();
             it != entry->second.at("current").end(); ++it)
          if (it.key() != "config_generation")
            action[it.key()] = it.value();
        choices.push_back(action);
      } else
        for (const auto &variant :
             config.at("domain").at(std::to_string(oid))) {
          auto action = original;
          action.update(variant);
          choices.push_back(action);
        }
    }
    std::vector<StatePtr> candidates;
    for (const auto &parent : beam) {
      for (const auto &choice : choices) {
        if (expired())
          return Json::array();
        if (expansions >= maximum)
          break;
        const auto action =
            problem.parse_native_action(parent, sampling::canonical(choice));
        if (!problem.feasible(parent, action))
          continue;
        auto child = problem.transition(parent, action);
        ++expansions;
        environment.observe(child);
        if (entry != entries.end()) {
          auto observed = observations.find({oid, implementation(choice)});
          Json observation =
              observed == observations.end() ? Json() : observed->second;
          if (observation.is_null() && request.value("chunk_policy", false)) {
            const auto current = current_selectivities.find(oid);
            if (current != current_selectivities.end())
              observation = current->second;
          }
          environment.apply_runtime(child, oid, entry->second, observation);
        }
        (problem.complete(child) ? complete : candidates).push_back(child);
      }
      if (expansions >= maximum)
        break;
    }
    if (!complete.empty()) {
      completion_prediction = predict(complete);
      break;
    }
    if (candidates.empty())
      break;
    const auto prediction = predict(candidates);
    if (layer_pruning == "grouped") {
      const auto groups = prune_structure_groups(
          problem, candidates, prediction,
          config.value("max_structure_groups", std::size_t(15)),
          config.value("max_implementations_per_group", std::size_t(6)));
      beam.clear();
      for (auto index : groups.order) {
        if (beam.size() == width)
          break;
        beam.push_back(candidates[index]);
      }
      environment.retain(beam);
      continue;
    }
    const auto ranking =
        rank_layer(prediction, preference, minimum, layer_pruning);
    const auto &scores = ranking.scores;
    if (scores.size() != candidates.size())
      throw std::invalid_argument("Runtime prediction count mismatch");
    beam.clear();
    std::vector<std::size_t> eligible;
    Scores eligible_scores;
    const std::unordered_set<std::size_t> excluded(ranking.excluded.begin(),
                                                   ranking.excluded.end());
    for (std::size_t i = 0; i < candidates.size(); ++i)
      if (!excluded.count(i)) {
        eligible.push_back(i);
        eligible_scores.push_back(scores[i]);
      }
    if (ranking.order.empty()) {
      for (auto index : trim(eligible_scores, width, sample))
        beam.push_back(candidates[eligible[index]]);
    } else {
      std::unordered_set<std::size_t> eligible_set(eligible.begin(),
                                                   eligible.end());
      for (auto index : ranking.order) {
        if (beam.size() == width)
          break;
        if (eligible_set.count(index))
          beam.push_back(candidates[index]);
      }
    }
    environment.retain(beam);
  }
  if (complete.empty())
    return Json::array();
  const auto &prediction = completion_prediction;
  const auto decision = select_completions(prediction.scores, prediction.times,
                                           prediction.accuracies, minimum);
  if (decision.selected.empty())
    return Json::array();
  auto best = complete.at(preference == "accurate" && !minimum
                              ? *decision.accurate_index
                              : decision.selected.front());
  Json updates = Json::array();
  for (const auto &text : best->history()) {
    auto action = Json::parse(text);
    const auto entry = entries.find(action.at("operator_id").get<Id>());
    if (entry == entries.end() || entry->second.at("status") == "DONE")
      continue;
    const auto &current = entry->second.at("current");
    if (action.at("model_id") != current.at("model_id") ||
        action.at("layout") != current.at("layout"))
      updates.push_back(
          {{"operator_id", action.at("operator_id")},
           {"expected_generation", current.at("config_generation")},
           {"model_id", action.at("model_id")},
           {"layout", action.at("layout")}});
  }
  return updates;
}
}
