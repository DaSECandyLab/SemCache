#include "join_candidates.hpp"
#include "../populations.hpp"
#include "../random.hpp"
#include <unordered_map>

namespace semcache::sampling::resident_detail {
JoinCandidates direct_join_candidates(const SampleInputs &inputs,
                                      std::size_t limit) {
  if (inputs.empty() || inputs.size() > 2)
    throw std::invalid_argument("Direct sample requires one or two sides");
  const auto n = row_count(inputs[0]);
  const auto m = inputs.size() == 2 ? row_count(inputs[1]) : 1;
  if (m && n > std::numeric_limits<std::size_t>::max() / m)
    throw std::overflow_error("Direct pair population overflow");
  JoinCandidates result;
  result.population = result.group_population = n * m;
  const auto count = std::min(limit, result.population);
  result.probes.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto left = n >= m ? i % n : (i / m + i % m) % n;
    const auto right = n >= m ? (i / n + i % n) % m : i % m;
    Json unit = Json::array({row_at(inputs[0], left)});
    if (inputs.size() == 2)
      unit.push_back(row_at(inputs[1], right));
    result.probes.push_back(
        {std::move(unit),
         double(result.population) / count,
         {double(std::min(count, n)) / n, double(std::min(count, m)) / m},
         {left, right}});
  }
  return result;
}
SharedSamples direct_product_samples(const std::vector<const Json *> &factors,
                                     std::size_t limit) {
  if (factors.empty())
    throw std::invalid_argument(
        "Direct tuple sampling requires source factors");
  auto rows = std::make_shared<const Json>(*factors.front());
  for (std::size_t i = 1; i < factors.size(); ++i) {
    auto pairs = direct_join_candidates({rows.get(), factors[i]}, limit);
    Json tuples = Json::array();
    for (auto &pair : pairs.probes) {
      auto row = combine(pair.unit[0], pair.unit[1]);
      sample_factor(row, sample_factor(row) * pair.expansion);
      tuples.push_back(std::move(row));
    }
    rows = std::make_shared<const Json>(std::move(tuples));
  }
  return rows;
}
namespace {
// Prefix of Fisher-Yates on a virtual index array: O(k) storage, even if the
// underlying side contains billions of factorized rows.
class Permutation {
  std::size_t size_;
  Random random_;
  std::vector<std::size_t> prefix_;
  std::unordered_map<std::size_t, std::size_t> swaps_;

public:
  Permutation(std::size_t size, std::uint64_t seed)
      : size_(size), random_(seed) {}
  std::size_t at(std::size_t index) {
    while (prefix_.size() <= index) {
      const auto begin = prefix_.size();
      const auto selected = begin + random_.below(size_ - begin);
      const auto read = [&](std::size_t i) {
        auto it = swaps_.find(i);
        return it == swaps_.end() ? i : it->second;
      };
      const auto value = read(selected), replacement = read(begin);
      swaps_.erase(begin);
      if (selected != begin)
        swaps_[selected] = replacement;
      prefix_.push_back(value);
    }
    return prefix_.at(index);
  }
};

class Domain {
  const Json &rows_;
  std::vector<std::vector<std::size_t>> groups_;

public:
  Domain(const Json &rows, const std::set<std::string> &fields,
         std::size_t index_limit, const std::set<std::string> &dependencies)
      : rows_(rows) {
    if (product(rows) || row_count(rows) > index_limit)
      return;
    const auto available = columns(rows);
    std::vector<std::string> keys;
    std::set_intersection(dependencies.begin(), dependencies.end(),
                          available.begin(), available.end(),
                          std::back_inserter(keys));
    if (keys.empty())
      std::set_intersection(fields.begin(), fields.end(), available.begin(),
                            available.end(), std::back_inserter(keys));
    std::map<std::string, std::size_t> index;
    if (keys.empty())
      return;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      Json values = Json::array();
      for (const auto &key : keys)
        values.push_back(rows[i].at("values").at(key));
      auto signature = canonical(values);
      auto [it, inserted] = index.emplace(signature, groups_.size());
      if (inserted)
        groups_.emplace_back();
      groups_[it->second].push_back(i);
    }
  }
  std::size_t size() const {
    return groups_.empty() ? row_count(rows_) : groups_.size();
  }
  std::pair<std::size_t, std::size_t> representative(std::size_t group,
                                                     Random &random) const {
    if (groups_.empty())
      return {group, 1};
    const auto &members = groups_.at(group);
    return {members[random.below(members.size())], members.size()};
  }
};
}

struct JoinCoordinateCursor::Impl {
  Domain left;
  std::optional<Domain> right;
  std::size_t population, n, m;
  Permutation left_order, right_order;
  Random representatives;
  std::vector<JoinCoordinate> coordinates;
  std::array<std::map<std::size_t, std::size_t>, 2> visits, histograms;
  struct PrefixHistogram {
    std::size_t prefix = 0;
    std::map<std::size_t, std::size_t> histogram;
  };
  // Restoring a frozen prefix asks for this histogram once per output row.
  // Keep only the latest historical prefix per side; extension does not
  // change the deterministic positions of coordinates already visited.
  mutable std::array<PrefixHistogram, 2> prefix_histograms;

  Impl(const SampleInputs &inputs, const std::set<std::string> &fields,
       std::size_t index_limit, std::uint64_t seed,
       const std::set<std::string> &dependencies)
      : left(inputs[0], fields, index_limit, dependencies),
        right(inputs.size() == 2
                  ? std::optional<Domain>(std::in_place, inputs[1], fields,
                                          index_limit, dependencies)
                  : std::nullopt),
        population(row_count(inputs[0])), n(left.size()),
        m(right ? right->size() : 1), left_order(n, seed),
        right_order(m, seed ^ 0x9e3779b97f4a7c15ULL),
        representatives(seed ^ 0xd1b54a32d192ed03ULL) {
    if (right) {
      const auto count = row_count(inputs[1]);
      if (count && population > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("Join sample population overflow");
      population *= count;
    }
  }
  std::array<std::size_t, 2> position(std::size_t t) const {
    return {n >= m ? t % n : (t / m + t % m) % n,
            n >= m ? (t / n + t % n) % m : t % m};
  }
  void visit(std::size_t side, std::size_t group) {
    auto &count = visits[side][group];
    if (count && !--histograms[side][count])
      histograms[side].erase(count);
    ++histograms[side][++count];
  }
};

JoinCoordinateCursor::JoinCoordinateCursor(
    const SampleInputs &inputs, const std::set<std::string> &fields,
    std::size_t index_limit, std::uint64_t seed,
    const std::set<std::string> &dependencies) {
  if (inputs.empty() || inputs.size() > 2)
    throw std::invalid_argument("Join sample requires one or two sides");
  impl_ =
      std::make_unique<Impl>(inputs, fields, index_limit, seed, dependencies);
}
JoinCoordinateCursor::~JoinCoordinateCursor() = default;
void JoinCoordinateCursor::extend(std::size_t target) {
  auto &s = *impl_;
  const auto count = std::min(target, s.n * s.m);
  while (s.coordinates.size() < count) {
    const auto pos = s.position(s.coordinates.size());
    auto a = s.left.representative(s.left_order.at(pos[0]), s.representatives);
    auto b = s.right ? s.right->representative(s.right_order.at(pos[1]),
                                               s.representatives)
                     : std::pair<std::size_t, std::size_t>{0, 1};
    s.coordinates.push_back({{a.first, b.first}, {a.second, b.second}});
    s.visit(0, pos[0]);
    s.visit(1, pos[1]);
  }
}
std::size_t JoinCoordinateCursor::size() const {
  return impl_->coordinates.size();
}
std::size_t JoinCoordinateCursor::population() const {
  return impl_->population;
}
std::size_t JoinCoordinateCursor::group_population() const {
  return impl_->n * impl_->m;
}
const JoinCoordinate &JoinCoordinateCursor::at(std::size_t index) const {
  return impl_->coordinates.at(index);
}
double JoinCoordinateCursor::inclusion(std::size_t side, std::size_t members,
                                       std::size_t prefix) const {
  const auto &s = *impl_;
  if (!prefix)
    prefix = size();
  const auto *hist = &s.histograms.at(side);
  if (prefix != size()) {
    auto &cached = s.prefix_histograms.at(side);
    if (cached.prefix != prefix) {
      std::map<std::size_t, std::size_t> visits, histogram;
      for (std::size_t t = 0; t < prefix; ++t)
        ++visits[s.position(t).at(side)];
      for (const auto &entry : visits)
        ++histogram[entry.second];
      cached.histogram = std::move(histogram);
      cached.prefix = prefix;
    }
    hist = &cached.histogram;
  }
  double sum = 0;
  for (const auto &entry : *hist)
    sum +=
        entry.second *
        (members == 1 ? 1.
                      : -std::expm1(entry.first * std::log1p(-1. / members)));
  const auto groups = side ? s.m : s.n;
  return groups ? sum / groups : 0.;
}

JoinCandidates join_candidates(const SampleInputs &inputs,
                               const std::set<std::string> &fields,
                               std::size_t limit, std::size_t index_limit,
                               std::uint64_t seed,
                               const std::set<std::string> &dependencies) {
  if (!limit)
    throw std::invalid_argument("Join sample requires positive budget");
  JoinCoordinateCursor cursor(inputs, fields, index_limit, seed, dependencies);
  cursor.extend(limit);
  JoinCandidates result;
  result.population = cursor.population();
  result.group_population = cursor.group_population();
  result.compact_values =
      inputs.size() == 2 && !fields.empty() &&
      std::none_of(fields.begin(), fields.end(), [](const auto &name) {
        return name.rfind("__semantic_input:", 0) == 0;
      });
  auto payload_fields = fields;
  payload_fields.insert(dependencies.begin(), dependencies.end());
  result.probes.reserve(cursor.size());
  for (std::size_t t = 0; t < cursor.size(); ++t) {
    const auto &coordinate = cursor.at(t);
    Json unit = Json::array();
    double expansion = double(result.group_population) / cursor.size();
    std::vector<double> probabilities;
    for (std::size_t side = 0; side < inputs.size(); ++side) {
      unit.push_back(result.compact_values
                         ? projected_row_at(inputs[side],
                                            coordinate.indices[side],
                                            payload_fields)
                         : row_at(inputs[side], coordinate.indices[side]));
      expansion *= coordinate.members[side];
      probabilities.push_back(cursor.inclusion(side, coordinate.members[side]));
    }
    result.probes.push_back({std::move(unit), expansion,
                             std::move(probabilities), coordinate.indices});
  }
  return result;
}
}
