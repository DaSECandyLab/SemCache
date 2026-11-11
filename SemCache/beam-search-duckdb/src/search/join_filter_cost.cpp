#include "join_filter_cost.hpp"
#include "../sampling/profile/environment.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace semcache {
namespace {
using sampling::Json;
std::optional<double> number(const Json &object, const char *key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number())
    return {};
  const auto value = it->get<double>();
  return std::isfinite(value) && value >= 0 ? std::optional<double>(value)
                                            : std::nullopt;
}
std::optional<double> cost(const Json &bank, const std::string &kind) {
  const auto it = bank.find(kind);
  return it == bank.end() ? std::nullopt : number(*it, "seconds_per_input");
}
}

JoinFilterPruning prune_join_filter_cost(const StateProblem &problem,
                                         const std::vector<StatePtr> &states,
                                         const std::vector<Json> &statistics,
                                         const Json &calibration) {
  if (states.size() != statistics.size())
    throw std::invalid_argument("Join/Filter statistics count mismatch");
  JoinFilterPruning result;
  if (!calibration.contains("relational") || !calibration.contains("semantic"))
    return result;
  using ParentRoot = std::pair<std::uint64_t, Id>;
  std::map<ParentRoot, std::vector<std::size_t>> joins;
  for (std::size_t i = 0; i < states.size(); ++i) {
    const auto &s = states[i];
    if (!s->parent || s->action.inputs.size() != 2 ||
        !problem.relational_inner_join(s->action.operator_id))
      continue;
    for (auto input : s->action.inputs)
      joins[{s->parent->handle_id, input}].push_back(i);
  }
  std::map<std::pair<std::uint64_t, Id>, bool> movable;
  for (std::size_t i = 0; i < states.size(); ++i) {
    const auto &f = states[i];
    if (!f->parent || f->action.inputs.size() != 1 ||
        !problem.semantic_filter(f->action.operator_id))
      continue;
    const auto source = f->action.inputs.front();
    const auto peers = joins.find({f->parent->handle_id, source});
    if (peers == joins.end())
      continue;
    const auto input = number(statistics[i], "input_rows");
    const auto output = number(statistics[i], "output_rows");
    if (!input || !output || *output > *input || *input == 0)
      continue;
    const auto action = nlohmann::json::parse(f->action_json);
    if (!action.contains("model_id"))
      continue;
    const auto &semantic = calibration.at("semantic");
    const auto model = semantic.find(action.at("model_id").get<std::string>());
    if (model == semantic.end())
      continue;
    const auto filter_cost =
        cost(*model, problem.operator_kind(f->action.operator_id));
    if (!filter_cost)
      continue;
    std::optional<JoinFilterDecision> best;
    double saving = 0.;
    for (auto index : peers->second) {
      const auto &j = states[index];
      const auto joined = number(statistics[index], "output_rows");
      const auto join_cost = cost(calibration.at("relational"),
                                  problem.operator_kind(j->action.operator_id));
      const auto sides = statistics[index].find("input_side_rows");
      if (!joined || !join_cost || sides == statistics[index].end() ||
          !sides->is_array() || sides->size() != 2)
        continue;
      const auto side = j->action.inputs[0] == source ? 0 : 1;
      const auto shared = number(Json{{"value", (*sides)[side]}}, "value");
      const auto peer = number(Json{{"value", (*sides)[1 - side]}}, "value");
      if (!shared || !peer ||
          std::abs(*shared - *input) > 1e-9 * std::max(1., *input))
        continue;
      const auto join_first =
          (*input + *peer) * *join_cost + *joined * *filter_cost;
      const auto filter_first =
          *input * *filter_cost + (*output + *peer) * *join_cost;
      const auto gain = filter_first - join_first;
      if (!std::isfinite(join_first) || !std::isfinite(filter_first) ||
          gain <= std::max(1e-9, std::abs(filter_first) * 1e-9) ||
          gain <= saving)
        continue;
      const auto key = std::make_pair(j->handle_id, f->action.operator_id);
      auto legal = movable.find(key);
      if (legal == movable.end()) {
        const auto choices = problem.input_choices(j, f->action.operator_id);
        legal = movable
                    .emplace(key,
                             std::find(choices.begin(), choices.end(),
                                       Ids{j->added->node_id}) != choices.end())
                    .first;
      }
      if (!legal->second)
        continue;
      auto moved = f->action;
      moved.inputs = {j->added->node_id};
      moved.pair_for = problem.pair_target(j, moved.operator_id, moved.inputs);
      if (!problem.feasible(j, moved))
        continue;
      saving = gain;
      best = JoinFilterDecision{i,
                                index,
                                f->action.operator_id,
                                j->added->node_id,
                                *input,
                                *peer,
                                *joined,
                                *output,
                                *join_cost,
                                *filter_cost,
                                join_first,
                                filter_first};
    }
    if (best) {
      result.excluded.push_back(i);
      result.decisions.push_back(*best);
    }
  }
  return result;
}
sampling::Json join_filter_decisions(const JoinFilterPruning &result) {
  auto rows = Json::array();
  for (const auto &d : result.decisions)
    rows.push_back({{"filter_index", d.filter_index},
                    {"join_index", d.join_index},
                    {"filter_operator", d.filter_operator},
                    {"join_node", d.join_node},
                    {"input_rows", d.input_rows},
                    {"peer_rows", d.peer_rows},
                    {"join_output", d.join_output},
                    {"filter_output", d.filter_output},
                    {"join_seconds_per_row", d.join_seconds_per_row},
                    {"filter_seconds_per_row", d.filter_seconds_per_row},
                    {"join_first_seconds", d.join_first_seconds},
                    {"filter_first_seconds", d.filter_first_seconds}});
  return rows;
}
std::vector<sampling::Json>
join_filter_statistics(const sampling::SearchEnvironment &environment,
                       const std::vector<StatePtr> &states) {
  const auto *profile =
      dynamic_cast<const sampling::ProfileEnvironment *>(&environment);
  std::vector<Json> result;
  result.reserve(states.size());
  for (const auto &state : states) {
    Json row = Json::object();
    if (state->added) {
      if (profile) {
        const auto view = profile->feature_view(state, state->added->node_id);
        for (const auto *key : {"input_rows", "output_rows", "input_side_rows"})
          if (const auto value = view.find(key))
            row[key] = *value;
      } else {
        const auto blocks = environment.statistic_blocks(state);
        const auto block = blocks.find(state->added->node_id);
        if (block != blocks.end())
          for (const auto *key :
               {"input_rows", "output_rows", "input_side_rows"})
            if (const auto value = block->second->find(key);
                value != block->second->end())
              row[key] = *value;
      }
    }
    result.push_back(std::move(row));
  }
  return result;
}
}
