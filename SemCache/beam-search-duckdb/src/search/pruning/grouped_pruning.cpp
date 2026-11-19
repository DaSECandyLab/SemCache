#include "grouped_pruning.hpp"
#include "../drivers/resident_driver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace semcache {
namespace {
struct Group {
  std::vector<std::uint32_t> features;
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

using Implementation = std::vector<std::pair<Id, std::size_t>>;
Implementation implementation(const StatePtr &state) {
  Implementation result;
  result.reserve(state->depth);
  for (auto s = state.get(); s->parent; s = s->parent.get())
    result.emplace_back(s->action.operator_id, s->action.domain_index);
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  return result;
}
double difference(const Implementation &a, const Implementation &b) {
  std::size_t common = 0, same_variant = 0;
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first < b[j].first)
      ++i;
    else if (b[j].first < a[i].first)
      ++j;
    else {
      ++common;
      same_variant += (a[i].second == b[j].second);
      ++i;
      ++j;
    }
  }
  const auto different = (a.size() + b.size() - 2 * common) + (common - same_variant);
  return static_cast<double>(different);
}
double distance(const std::vector<std::uint32_t> &a,
                const std::vector<std::uint32_t> &b) {
  std::size_t intersection = 0, i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j])
      ++i;
    else if (b[j] < a[i])
      ++j;
    else {
      ++intersection;
      ++i;
      ++j;
    }
  }
  const auto total = a.size() + b.size() - intersection;
  return total ? 1. - static_cast<double>(intersection) / total : 0.;
}

std::vector<std::size_t> inside_group(const Group &group,
                                      const std::vector<StatePtr> &states,
                                      const ResidentPrediction &prediction,
                                      std::size_t capacity) {
  Scores scores;
  scores.reserve(group.members.size());
  for (auto index : group.members)
    scores.push_back(prediction.scores[index]);
  std::vector<std::size_t> retained;
  for (const auto &local_front : pareto_fronts(scores)) {
    std::vector<std::size_t> front;
    front.reserve(local_front.size());
    for (auto i : local_front)
      front.push_back(group.members[i]);
    if (retained.size() + front.size() <= capacity) {
      std::sort(front.begin(), front.end(),
                [&](auto a, auto b) { return faster(prediction, a, b); });
      retained.insert(retained.end(), front.begin(), front.end());
    } else {
      std::unordered_map<std::size_t, double> crowding;
      crowding.reserve(front.size() * 2 + 1);
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
      struct Candidate {
        std::size_t id;
        double crowd;
        Implementation impl;
        double min_dist;
      };
      std::vector<Implementation> retained_impls;
      retained_impls.reserve(capacity);
      for (auto i : retained)
        retained_impls.push_back(implementation(states[i]));
      std::vector<Candidate> candidates;
      candidates.reserve(front.size());
      for (auto i : front) {
        auto impl = implementation(states[i]);
        double d = std::numeric_limits<double>::infinity();
        for (const auto &ret : retained_impls)
          d = std::min(d, difference(impl, ret));
        candidates.push_back({i, crowding[i], std::move(impl), d});
      }
      while (!candidates.empty() && retained.size() < capacity) {
        const auto best = std::min_element(
            candidates.begin(), candidates.end(),
            [&](const auto &a, const auto &b) {
              if (a.crowd != b.crowd)
                return a.crowd > b.crowd;
              return a.min_dist != b.min_dist
                         ? a.min_dist > b.min_dist
                         : faster(prediction, a.id, b.id);
            });
        auto chosen_impl = std::move(best->impl);
        retained.push_back(best->id);
        candidates.erase(best);
        for (auto &cand : candidates)
          cand.min_dist =
              std::min(cand.min_dist, difference(cand.impl, chosen_impl));
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
  result.membership.reserve(states.size());
  std::vector<Group> groups;
  std::unordered_map<std::string, std::size_t> identities;
  std::unordered_map<std::string_view, std::uint32_t> feature_ids;
  std::unordered_map<std::uint64_t, std::string> shape_memo;
  shape_memo.reserve(states.size() * 2 + 16);
  // Native admission already removes equivalent states. Do not treat a coarse
  // structure group as semantic equivalence or discard different dependencies.
  for (std::size_t i = 0; i < states.size(); ++i) {
    auto key = problem.logical_structure_key(states[i], shape_memo);
    const auto inserted = identities.emplace(std::move(key), groups.size());
    if (inserted.second) {
      Group group;
      for (auto s = states[i].get(); s->parent; s = s->parent.get()) {
        if (!s->added->children.empty()) {
          const std::string_view sv(shape_memo.at(s->added->handle_id));
          const auto id_it = feature_ids.emplace(sv, feature_ids.size());
          group.features.push_back(id_it.first->second);
        }
      }
      std::sort(group.features.begin(), group.features.end());
      group.features.erase(
          std::unique(group.features.begin(), group.features.end()),
          group.features.end());
      groups.push_back(std::move(group));
    }
    const auto group_idx = inserted.first->second;
    groups[group_idx].members.push_back(i);
    result.membership.push_back(group_idx);
  }
  result.groups = groups.size();
  Scores averages;
  averages.reserve(groups.size());
  result.mean_predictions.reserve(groups.size());
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
  selected.reserve(std::min(maximum_groups, groups.size()));
  std::vector<double> min_dist(groups.size(), 1.);
  for (const auto &front : pareto_fronts(averages)) {
    auto remaining = front;
    while (!remaining.empty() && selected.size() < maximum_groups) {
      const bool use_diversity =
          selected.size() + remaining.size() > maximum_groups;
      const auto best = std::min_element(
          remaining.begin(), remaining.end(), [&](auto a, auto b) {
            if (use_diversity && min_dist[a] != min_dist[b])
              return min_dist[a] > min_dist[b];
            if (averages[a] != averages[b])
              return averages[a] > averages[b];
            return a < b;
          });
      const auto chosen = *best;
      selected.push_back(chosen);
      remaining.erase(best);
      if (selected.size() < maximum_groups && !remaining.empty())
        for (auto g : remaining)
          min_dist[g] = std::min(
              min_dist[g], distance(groups[g].features, groups[chosen].features));
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
