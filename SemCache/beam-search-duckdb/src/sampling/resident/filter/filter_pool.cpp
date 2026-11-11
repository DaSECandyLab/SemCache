#include "filter_pool.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace semcache::sampling::resident_detail {
FilterPool::FilterPool(std::vector<const Json *> rows,
                       std::vector<double> weights,
                       std::vector<std::size_t> strata)
    : rows_(std::move(rows)), weights_(std::move(weights)),
      strata_(std::move(strata)) {
  if (rows_.size() != weights_.size() || rows_.size() != strata_.size())
    throw std::invalid_argument("Filter pool coordinate sizes differ");
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (!std::isfinite(weights_[i]) || weights_[i] <= 0 ||
        !coordinates_.emplace(rows_[i]->at("id").get<std::string>(), i).second)
      throw std::invalid_argument("Invalid filter pool coordinate/weight");
    // The fixed population and its reduction order never change; only the
    // common-known quotas below depend on published predicate answers.
    stratum_totals_[strata_[i]] += weights_[i];
    total_weight_ += weights_[i];
  }
}

std::optional<std::size_t> FilterPool::coordinate(const std::string &id) const {
  const auto it = coordinates_.find(id);
  return it == coordinates_.end() ? std::nullopt
                                  : std::optional<std::size_t>(it->second);
}

bool FilterPool::compatible(std::size_t i, const Json &row,
                            const std::set<std::string> &fields) const {
  if (i >= rows_.size())
    return false;
  const auto &base = rows_[i]->at("values"), &actual = row.at("values");
  for (const auto &field : fields) {
    const auto a = base.find(field), b = actual.find(field);
    if (a == base.end() || b == actual.end() || a->type() != b->type() ||
        *a != *b)
      return false;
  }
  return true;
}

std::optional<bool> FilterPool::decision(const std::string &task,
                                         std::size_t i) const {
  const auto it = tasks_.find(task);
  if (it == tasks_.end() || !it->second.known(i))
    return std::nullopt;
  return it->second.accepted(i);
}

void FilterPool::publish(const std::string &task, std::size_t i, bool passed) {
  auto &mask = tasks_.try_emplace(task, rows_.size()).first->second;
  if (mask.known(i)) {
    if (mask.accepted(i) != passed)
      throw std::logic_error("Conflicting fixed-pool predicate evidence");
    return;
  }
  mask.publish(i, passed);
  ++version_;
}

std::optional<ConditionEstimate>
FilterPool::estimate(const std::string &task,
                     const std::vector<std::string> &conditions, double minimum,
                     bool relax) const {
  const auto target = tasks_.find(task);
  if (target == tasks_.end())
    return std::nullopt;
  std::vector<const DecisionMask *> ancestors;
  auto common = target->second.known_words();
  for (const auto &condition : conditions) {
    const auto it = tasks_.find(condition);
    if (it == tasks_.end())
      return std::nullopt;
    ancestors.push_back(&it->second);
    for (std::size_t w = 0; w < common.size(); ++w)
      common[w] &= it->second.known_words()[w];
  }
  auto each = [&](const auto &bits, const auto &visit) {
    for (std::size_t w = 0; w < bits.size(); ++w) {
      auto word = bits[w];
      while (word) {
        visit(w * 64 + static_cast<std::size_t>(__builtin_ctzll(word)));
        word &= word - 1;
      }
    }
  };
  std::map<std::size_t, std::pair<double, double>> quotas;
  for (const auto &entry : stratum_totals_)
    quotas.emplace(entry.first, std::make_pair(entry.second, 0.));
  double known = 0;
  each(common, [&](std::size_t i) {
    quotas[strata_[i]].second += weights_[i];
    known += weights_[i];
  });
  if (!known)
    return std::nullopt;
  for (const auto &entry : quotas)
    if (!entry.second.second)
      return std::nullopt;
  // Correct imbalance in evaluated strata using this common-known snapshot,
  // never mutable global admission counts. Within-stratum path selection can
  // still be biased: known coverage is published, not advertised as unbiased.
  std::vector<double> weights(rows_.size(), 0.);
  each(common, [&](std::size_t i) {
    const auto &q = quotas.at(strata_[i]);
    weights[i] = weights_[i] * q.first / q.second;
  });
  struct Moments {
    double mass = 0, squares = 0, passed = 0;
    std::size_t count = 0;
    double effective() const { return squares ? mass * mass / squares : 0.; }
  };
  auto moments = [&](const auto &bits) {
    Moments out;
    each(bits, [&](std::size_t i) {
      out.mass += weights[i];
      out.squares += weights[i] * weights[i];
      out.passed += weights[i] * target->second.accepted(i);
      ++out.count;
    });
    return out;
  };
  auto full = common;
  for (const auto *ancestor : ancestors)
    for (std::size_t w = 0; w < full.size(); ++w)
      full[w] &= ancestor->accepted_words()[w];
  ConditionEstimate out;
  out.conditions = conditions.size();
  out.known_coverage = total_weight_ ? known / total_weight_ : 0.;
  auto result = moments(full);
  if (result.effective() >= minimum) {
    out.retained = conditions;
  } else {
    if (!relax)
      return std::nullopt;
    const auto overall = full == common ? result : moments(common);
    if (overall.effective() < minimum)
      return std::nullopt;
    std::vector<std::pair<double, std::size_t>> order;
    const auto p = overall.passed / overall.mass;
    for (std::size_t j = 0; j < ancestors.size(); ++j) {
      double a = 0, both = 0;
      each(common, [&](std::size_t i) {
        if (ancestors[j]->accepted(i)) {
          a += weights[i];
          both += weights[i] * target->second.accepted(i);
        }
      });
      a /= overall.mass;
      both /= overall.mass;
      const auto denominator = std::sqrt(p * (1 - p) * a * (1 - a));
      order.emplace_back(
          denominator > 0 ? std::abs((both - p * a) / denominator) : 0., j);
    }
    std::stable_sort(
        order.begin(), order.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });
    auto selected = common;
    for (const auto &entry : order) {
      auto trial = selected;
      const auto j = entry.second;
      for (std::size_t w = 0; w < trial.size(); ++w)
        trial[w] &= ancestors[j]->accepted_words()[w];
      if (moments(trial).effective() >= minimum) {
        selected = std::move(trial);
        out.retained.push_back(conditions[j]);
      }
    }
    result = moments(selected);
    out.relaxed = true;
  }
  if (!result.mass)
    return std::nullopt;
  out.rate = result.passed / result.mass;
  out.effective = result.effective();
  out.samples = result.count;
  return out;
}
}
