#include "probe_internal.hpp"

namespace semcache::sampling::resident_detail {
const std::string &JoinProbeEvidence::Impl::signature(const Json &value) {
  auto found = signatures.find(&value);
  if (found == signatures.end())
    found = signatures.emplace(&value, canonical(value)).first;
  return found->second;
}
double JoinProbeEvidence::Impl::quota_weight(const std::vector<QuotaKey> &recipe,
                    const Counts *counts) const {
  double value = 1;
  for (const auto &key : recipe) {
    const auto &q = quotas.at(key);
    const auto count = counts ? counts->at(key) : q.second;
    if (!count)
      throw std::invalid_argument("Missing admitted stratum quota");
    value *= q.first / count;
  }
  return value;
}
const JoinProbeEvidence::Impl::Row &JoinProbeEvidence::Impl::row(std::size_t side, std::size_t index) {
  auto found = rows[side].find(index);
  if (found != rows[side].end())
    return found->second;
  Row value;
  if (product(*inputs[side]))
    value.owned = projected_row_at(*inputs[side], index, {});
  else
    value.borrowed = &inputs[side]->at(index);
  const auto &metadata = value.metadata();
  const auto prior = metadata.find("_sample_origin_probabilities");
  std::set<std::string> seen;
  for (const auto &origin : metadata.at("lineage")) {
    auto key = canonical(origin);
    if (!seen.insert(key).second)
      continue;
    std::optional<double> probability;
    if (prior != metadata.end() && prior->contains(key)) {
      if (!prior->at(key).is_null())
        probability = prior->at(key).get<double>();
    } else if (!metadata.contains("_sample_weight_factor"))
      probability = 1.;
    value.origins.push_back({std::move(key), relkey(origin[0]),
                             origin[1].get<std::string>(), probability});
    // Different origins may share a stratum. Each keeps its own factor,
    // and multiplication follows the first encounter order in lineage.
    value.recipe.push_back(stratum(origin));
  }
  value.weight =
      sample_factor(metadata) * quota_weight(value.recipe, nullptr);
  return rows[side].emplace(index, std::move(value)).first->second;
}
void JoinProbeEvidence::Impl::add_sources(std::map<std::string, Source> &bank, const Row &row,
                 std::size_t side, std::size_t members,
                 std::size_t position) {
  const auto &metadata = row.metadata();
  for (const auto &origin : row.origins) {
    auto [entry, inserted] =
        bank.try_emplace(origin.key, Source{origin.relation, origin.id, {}});
    auto &classes = entry->second.classes;
    InclusionClass c{side, members, origin.prior,
                     metadata.value("_sample_origin_approximate", false),
                     metadata.value("_sample_match_incomplete", false)};
    classes.emplace(std::move(c), position);
  }
}
JoinProbeEvidence::JoinProbeEvidence(OwnedInputs inputs,
                                     const std::set<std::string> &fields,
                                     const std::set<std::string> &dependencies,
                                     std::size_t index_limit,
                                     std::size_t origin_limit,
                                     std::uint64_t seed, QuotaBank quotas,
                                     Stratum stratum, std::size_t origin_scale)
    : impl_(std::make_unique<Impl>(
          std::move(inputs), fields, dependencies, index_limit, origin_limit,
          seed, std::move(quotas), std::move(stratum), origin_scale)) {}
JoinProbeEvidence::~JoinProbeEvidence() = default;
void JoinProbeEvidence::configure_request(RequestKeyPlan plan) {
  auto &s = *impl_;
  if (s.request_memo || !s.records.empty())
    throw std::logic_error(
        "Join request context already configured or prepared");
  s.request_memo = std::make_unique<RequestMemo>(std::move(plan));
}
std::vector<std::size_t> JoinProbeEvidence::prepare(std::size_t limit) {
  auto &s = *impl_;
  if (!s.request_memo)
    throw std::logic_error("Missing Join request context");
  return prepare(limit, [&](std::size_t left, std::size_t right) {
    auto key = s.request_memo->key(
        [&](const std::string &field, bool is_right) -> const Json & {
          const auto *cell = find_value(*s.inputs.at(is_right ? 1 : 0),
                                        is_right ? right : left, field);
          if (!cell)
            throw std::out_of_range("Missing semantic input field");
          return *cell;
        });
    return key;
  });
}
std::vector<std::size_t> JoinProbeEvidence::prepare(std::size_t limit,
                                                    const Key &key) {
  auto &s = *impl_;
  const auto previous = s.records.size();
  s.requested = std::min(limit, s.cursor.group_population());
  s.index_sources(s.requested);
  s.cursor.extend(s.requested);
  std::vector<std::size_t> novel, deferred;
  for (std::size_t i = s.records.size(); i < s.cursor.size(); ++i) {
    const auto &c = s.cursor.at(i);
    double weight = 1.;
    for (std::size_t side = 0; side < s.inputs.size(); ++side)
      weight *= s.row(side, c.indices[side]).weight * c.members[side];
    bool is_novel = s.seen_values.empty();
    std::vector<std::pair<std::string, std::string>> values;
    for (const auto &field : s.dependencies)
      for (std::size_t side = 0; side < s.inputs.size(); ++side)
        if (auto value = find_value(*s.inputs[side], c.indices[side], field)) {
          auto signature = s.signature(*value);
          is_novel |= !s.seen_values[field].count(signature);
          values.emplace_back(field, std::move(signature));
          break;
        }
    if (is_novel)
      for (auto &value : values)
        s.seen_values[value.first].insert(std::move(value.second));
    auto [parameter, fresh] = s.parameter_index.emplace(
        key(c.indices[0], c.indices[1]), s.parameters.size());
    if (fresh)
      s.parameters.push_back(parameter->first);
    auto totals = s.records.empty() ? Impl::Totals{} : s.records.back().totals;
    totals.mass.add(weight);
    totals.squares.add(weight * weight);
    s.records.push_back({parameter->second, is_novel, false, weight, totals});
    (is_novel ? s.novel : s.deferred).push_back(i);
    (is_novel ? novel : deferred).push_back(i);
    for (std::size_t side = 0; side < s.inputs.size(); ++side) {
      const auto &r = s.row(side, c.indices[side]);
      const auto w =
          r.weight * (side ? 1. : double(c.members[0]) * c.members[1]);
      s.representatives[side].try_emplace(
          r.metadata().at("id").get<std::string>(),
          Impl::Representative{i, w, r.weight});
      s.add_sources(s.arrived, r, side, c.members[side], i);
    }
  }
  // Retries can encounter prepared but unresolved coordinates.
  for (std::size_t i = s.completed; i < std::min(previous, s.requested); ++i)
    (s.records[i].novel ? novel : deferred).push_back(i);
  std::sort(novel.begin(), novel.end());
  std::sort(deferred.begin(), deferred.end());
  novel.insert(novel.end(), deferred.begin(), deferred.end());
  return novel;
}
const std::string &JoinProbeEvidence::key(std::size_t index) const {
  return impl_->parameters.at(impl_->records.at(index).parameter);
}
void JoinProbeEvidence::complete(const Label &label) {
  auto &s = *impl_;
  auto positive =
      s.completed ? s.records[s.completed - 1].totals.positive : Sum{};
  for (; s.completed < s.requested; ++s.completed) {
    auto &record = s.records[s.completed];
    record.success = label(s.parameters.at(record.parameter));
    if (record.success) {
      positive.add(record.weight);
      const auto &c = s.cursor.at(s.completed);
      for (const auto &field : s.dependencies) {
        const auto *value = s.inputs.size() == 2
                                ? find_value(*s.inputs[1], c.indices[1], field)
                                : nullptr;
        if (!value)
          value = find_value(*s.inputs[0], c.indices[0], field);
        if (value)
          s.passed_values[field].emplace(s.signature(*value), s.completed);
      }
      for (std::size_t side = 0; side < s.inputs.size(); ++side)
        s.add_sources(s.passed, s.rows[side].at(c.indices[side]), side,
                      c.members[side], s.completed);
    }
    record.totals.positive = positive;
    if (record.success)
      (record.novel ? s.positive_novel : s.positive_deferred)
          .push_back(s.completed);
  }
}
}
