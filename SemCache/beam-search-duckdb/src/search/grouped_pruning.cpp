#include "grouped_pruning.hpp"
#include "resident_driver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace semcache {
namespace {
struct Group {
  LogicalStructure structure;
  std::vector<std::size_t> members, retained;
};

double time_at(const ResidentPrediction &p, std::size_t i) {
  return p.times[i].value_or(std::numeric_limits<double>::infinity());
}
double accuracy_at(const ResidentPrediction &p, std::size_t i) {
  return p.accuracies[i].value_or(-1.);
}
bool faster(const ResidentPrediction &p, std::size_t a, std::size_t b) {
  if (time_at(p, a) != time_at(p, b))
    return time_at(p, a) < time_at(p, b);
  if (accuracy_at(p, a) != accuracy_at(p, b))
    return accuracy_at(p, a) > accuracy_at(p, b);
  return a < b;
}

using Implementation = std::map<Id, std::size_t>;
Implementation implementation(const StatePtr &state) {
  Implementation result;
  for (auto s = state; s->parent; s = s->parent)
    result.emplace(s->action.operator_id, s->action.domain_index);
  return result;
}
double difference(const Implementation &a, const Implementation &b) {
  std::size_t different = 0;
  for (const auto &[id, variant] : a) {
    const auto it = b.find(id);
    different += it == b.end() || it->second != variant;
  }
  for (const auto &[id, _] : b)
    different += !a.count(id);
  return static_cast<double>(different);
}
double distance(const LogicalStructure &a, const LogicalStructure &b) {
  std::size_t intersection = 0;
  for (const auto &value : a.features)
    intersection += b.features.count(value);
  const auto total = a.features.size() + b.features.size() - intersection;
  return total ? 1. - static_cast<double>(intersection) / total : 0.;
}

std::vector<std::size_t> inside_group(const Group &group,
                                      const std::vector<StatePtr> &states,
                                      const ResidentPrediction &prediction,
                                      std::size_t capacity) {
  Scores scores;
  for (auto index : group.members)
    scores.push_back(prediction.scores[index]);
  std::vector<std::size_t> retained;
  for (const auto &local_front : pareto_fronts(scores)) {
    std::vector<std::size_t> front;
    for (auto i : local_front)
      front.push_back(group.members[i]);
    if (retained.size() + front.size() <= capacity) {
      std::sort(front.begin(), front.end(),
                [&](auto a, auto b) { return faster(prediction, a, b); });
      retained.insert(retained.end(), front.begin(), front.end());
    } else {
      std::map<std::size_t, double> crowding;
      for (auto i : front)
        crowding[i] = 0.;
      for (std::size_t axis = 0; axis < 2; ++axis) {
        std::sort(front.begin(), front.end(), [&](auto a, auto b) {
          return prediction.scores[a][axis] != prediction.scores[b][axis]
                     ? prediction.scores[a][axis] < prediction.scores[b][axis]
                     : a < b;
        });
        const auto low = prediction.scores[front.front()][axis];
        const auto high = prediction.scores[front.back()][axis];
        if (high == low)
          continue;
        crowding[front.front()] = crowding[front.back()] =
            std::numeric_limits<double>::infinity();
        for (std::size_t j = 1; j + 1 < front.size(); ++j)
          crowding[front[j]] += (prediction.scores[front[j + 1]][axis] -
                                 prediction.scores[front[j - 1]][axis]) /
                                (high - low);
      }
      std::map<std::size_t, Implementation> variants;
      for (auto i : front)
        variants.emplace(i, implementation(states[i]));
      for (auto i : retained)
        variants.emplace(i, implementation(states[i]));
      while (!front.empty() && retained.size() < capacity) {
        const auto diversity = [&](auto i) {
          double d = std::numeric_limits<double>::infinity();
          for (auto j : retained)
            d = std::min(d, difference(variants.at(i), variants.at(j)));
          return d;
        };
        const auto best =
            std::min_element(front.begin(), front.end(), [&](auto a, auto b) {
              if (crowding[a] != crowding[b])
                return crowding[a] > crowding[b];
              const auto da = diversity(a), db = diversity(b);
              return da != db ? da > db : faster(prediction, a, b);
            });
        retained.push_back(*best);
        front.erase(best);
      }
    }
    if (retained.size() == capacity)
      break;
  }
  return retained;
}
}

GroupedPruning prune_structure_groups(const StateProblem &problem,
                                      const std::vector<StatePtr> &states,
                                      const ResidentPrediction &prediction,
                                      std::size_t maximum_groups,
                                      std::size_t per_group) {
  if (!maximum_groups || !per_group)
    throw std::invalid_argument("Grouped pruning quotas must be positive");
  if (prediction.scores.size() != states.size() ||
      prediction.times.size() != states.size() ||
      prediction.accuracies.size() != states.size())
    throw std::invalid_argument("Grouped prediction count mismatch");
  validate_scores(prediction.scores);
  for (std::size_t i = 0; i < states.size(); ++i) {
    if (!prediction.times[i] || !prediction.accuracies[i])
      throw std::invalid_argument(
          "Grouped means require paired time and accuracy predictions");
    if (prediction.times[i] &&
        (!std::isfinite(*prediction.times[i]) || *prediction.times[i] < 0))
      throw std::invalid_argument("Invalid grouped time prediction");
    if (prediction.accuracies[i] &&
        (!std::isfinite(*prediction.accuracies[i]) ||
         *prediction.accuracies[i] < 0 || *prediction.accuracies[i] > 1))
      throw std::invalid_argument("Invalid grouped accuracy prediction");
  }
  GroupedPruning result;
  std::vector<Group> groups;
  std::map<std::string, std::size_t> identities;
  // Native admission already removes equivalent states. Do not treat a coarse
  // structure group as semantic equivalence or discard different dependencies.
  for (std::size_t i = 0; i < states.size(); ++i) {
    auto shape = problem.logical_structure(states[i]);
    const auto inserted = identities.emplace(shape.key, groups.size());
    if (inserted.second)
      groups.push_back({std::move(shape), {}, {}});
    const auto group = inserted.first->second;
    groups[group].members.push_back(i);
    result.membership.push_back(group);
  }
  result.groups = groups.size();
  Scores averages;
  for (auto &group : groups) {
    group.retained = inside_group(group, states, prediction, per_group);
    std::array<double, 2> mean{};
    for (auto i : group.retained) {
      mean[0] -= *prediction.times[i];
      mean[1] += *prediction.accuracies[i];
    }
    for (auto &coordinate : mean)
      coordinate /= group.retained.size();
    // Group scores summarize the same retained subset; final execution still
    // chooses a real complete candidate, never this average point.
    averages.push_back(mean);
    result.mean_predictions.push_back({-mean[0], mean[1]});
  }
  std::vector<std::size_t> selected;
  for (const auto &front : pareto_fronts(averages)) {
    auto remaining = front;
    while (!remaining.empty() && selected.size() < maximum_groups) {
      const auto diversity = [&](auto i) {
        double d = 1.;
        for (auto j : selected)
          d = std::min(d, distance(groups[i].structure, groups[j].structure));
        return d;
      };
      const auto best = std::min_element(
          remaining.begin(), remaining.end(), [&](auto a, auto b) {
            const auto da = diversity(a), db = diversity(b);
            if (selected.size() + remaining.size() > maximum_groups && da != db)
              return da > db;
            if (averages[a] != averages[b])
              return averages[a] > averages[b];
            return a < b;
          });
      selected.push_back(*best);
      remaining.erase(best);
    }
    if (selected.size() == maximum_groups)
      break;
  }
  for (std::size_t offset = 0; offset < per_group; ++offset)
    for (auto group : selected)
      if (offset < groups[group].retained.size())
        result.order.push_back(groups[group].retained[offset]);
  return result;
}
}
