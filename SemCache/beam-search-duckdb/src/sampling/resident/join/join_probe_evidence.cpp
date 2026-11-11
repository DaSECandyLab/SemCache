#include "join_probe_evidence.hpp"
#include "origin_multiplicities.hpp"
#include "../requests/request_memo.hpp"
#include <cstring>
#include <deque>
#include <unordered_map>

namespace semcache::sampling::resident_detail {
struct JoinProbeEvidence::Impl {
  struct Origin {
    std::string key, relation, id;
    std::optional<double> prior;
  };
  struct Row {
    const Json *borrowed = nullptr;
    std::optional<Json> owned;
    std::vector<Origin> origins;
    std::vector<QuotaKey> recipe;
    double weight = 0;
    const Json &metadata() const { return borrowed ? *borrowed : *owned; }
  };
  struct Totals {
    Sum mass, squares, positive;
  };
  struct Record {
    std::size_t parameter;
    bool novel = false, success = false;
    double weight = 0;
    Totals totals;
  };
  struct InclusionClass {
    std::size_t side, members;
    std::optional<double> prior;
    bool approximate, incomplete;
    bool operator<(const InclusionClass &b) const {
      return std::tie(side, members, prior, approximate, incomplete) <
             std::tie(b.side, b.members, b.prior, b.approximate, b.incomplete);
    }
  };
  struct Source {
    std::string relation, id;
    std::map<InclusionClass, std::size_t> classes;
  };
  OwnedInputs inputs;
  std::set<std::string> fields, dependencies, payload;
  QuotaBank quotas;
  Stratum stratum;
  JoinCoordinateCursor cursor;
  std::array<std::map<std::size_t, Row>, 2> rows;
  std::array<std::map<std::string, double>, 2> copies;
  std::array<bool, 2> indexed{};
  std::array<std::size_t, 2> source_sizes{};
  std::size_t origin_min, origin_scale;
  static std::size_t largest_leaf(const Json &rows) {
    if (!product(rows))
      return rows.size();
    return std::max(largest_leaf(rows.at("product")[0]),
                    largest_leaf(rows.at("product")[1]));
  }
  std::size_t source_limit(std::size_t prefix) const {
    const auto scaled =
        prefix > std::numeric_limits<std::size_t>::max() / origin_scale
            ? std::numeric_limits<std::size_t>::max()
            : prefix * origin_scale;
    return std::max(origin_min, scaled);
  }
  bool sources_known(std::size_t side, std::size_t prefix) const {
    return source_sizes[side] <= source_limit(prefix);
  }
  void index_sources(std::size_t prefix) {
    for (std::size_t side = 0; side < inputs.size(); ++side)
      if (!indexed[side] && sources_known(side, prefix)) {
        indexed[side] = origin_multiplicities(
            *inputs[side], 1., source_limit(prefix), copies[side]);
        if (!indexed[side])
          throw std::logic_error("Invalid source index size");
      }
  }
  std::map<std::string, Source> arrived, passed;
  struct Representative {
    std::size_t first;
    double weight, base_weight;
  };
  std::array<std::map<std::string, Representative>, 2> representatives;
  std::vector<Record> records;
  std::vector<std::string> parameters;
  std::map<std::string, std::size_t> parameter_index;
  std::unordered_map<const Json *, std::string> signatures;
  std::map<std::string, std::map<std::string, std::size_t>> passed_values;
  std::vector<std::size_t> novel, deferred;
  std::vector<std::size_t> positive_novel, positive_deferred;
  mutable std::unordered_map<std::size_t, std::string> identities;
  mutable std::unordered_map<std::string, std::size_t> identity_coordinates;
  mutable std::size_t identity_indexed_prefix = 0;
  mutable bool unique_identities = true;
  mutable std::size_t identity_computations = 0;
  std::map<std::string, std::set<std::string>> seen_values;
  std::size_t completed = 0, requested = 0;
  std::unique_ptr<RequestMemo> request_memo;
  struct Publication {
    std::size_t prefix;
    std::uint64_t mass;
    std::weak_ptr<const JoinProbeSnapshot> snapshot;
  };
  // Snapshots own the evidence. Weak entries avoid an ownership cycle, and a
  // small FIFO bounds stale keys without retaining old publications.
  mutable std::deque<Publication> publications;
  // This unpublished preparation has no owner pointer (and thus no ownership
  // cycle). Labels and positive source evidence are filled only at snapshot().
  mutable std::shared_ptr<JoinProbeSnapshot> prepared;
  static constexpr std::size_t publication_capacity = 4;

  Impl(OwnedInputs in, const std::set<std::string> &f,
       const std::set<std::string> &d, std::size_t index_limit,
       std::size_t origin_limit, std::uint64_t seed, QuotaBank q, Stratum s,
       std::size_t scale)
      : inputs(std::move(in)), fields(f), dependencies(d), payload(f),
        quotas(std::move(q)), stratum(std::move(s)),
        cursor(SampleInputs(inputs), fields, index_limit, seed, dependencies),
        origin_min(origin_limit), origin_scale(scale) {
    payload.insert(dependencies.begin(), dependencies.end());
    for (std::size_t side = 0; side < inputs.size(); ++side)
      source_sizes[side] = largest_leaf(*inputs[side]);
  }
  const std::string &signature(const Json &value) {
    auto found = signatures.find(&value);
    if (found == signatures.end())
      found = signatures.emplace(&value, canonical(value)).first;
    return found->second;
  }
  double quota_weight(const std::vector<QuotaKey> &recipe,
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
  const Row &row(std::size_t side, std::size_t index) {
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
  void add_sources(std::map<std::string, Source> &bank, const Row &row,
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
  SourceEvidence sources(const std::map<std::string, Source> &bank,
                         std::size_t prefix, bool census, bool matches,
                         bool *incomplete = nullptr) const {
    SourceEvidence out;
    bool incomplete_matches = false;
    out.approximate = matches && !census;
    std::map<std::pair<std::size_t, std::size_t>, double> probabilities;
    for (const auto &entry : bank) {
      const auto &source = entry.second;
      bool present = false, unknown = false;
      double probability = 0;
      for (const auto &member : source.classes) {
        if (member.second >= prefix)
          continue;
        present = true;
        const auto &c = member.first;
        out.approximate |= c.approximate;
        incomplete_matches |= c.incomplete;
        if (!c.prior || (!census && !sources_known(c.side, prefix))) {
          unknown = true;
          continue;
        }
        const auto key = std::make_pair(c.side, c.members);
        auto [p, fresh] = probabilities.emplace(key, 0.);
        if (fresh)
          p->second = census ? 1. : cursor.inclusion(c.side, c.members, prefix);
        auto included = p->second;
        const auto count = census ? 1. : copies[c.side].at(entry.first);
        if (!census && count > 1) {
          included =
              included == 1. ? 1. : -std::expm1(count * std::log1p(-included));
          out.approximate = true;
        }
        probability = std::max(probability, *c.prior * included);
      }
      if (present)
        out.sources[source.relation][source.id] =
            unknown ? Json(nullptr) : Json(probability);
    }
    if (incomplete)
      *incomplete = incomplete_matches;
    return out;
  }
  Json sampled_row(std::size_t side, std::size_t index, std::size_t members,
                   const JoinProbeSnapshot &snapshot, bool full) const {
    auto value = full ? row_at(*inputs[side], index)
                      : projected_row_at(*inputs[side], index, payload);
    if (snapshot.census)
      return value;
    const auto &row = rows[side].at(index);
    Json probabilities = Json::object();
    bool approximate =
        row.metadata().value("_sample_origin_approximate", false);
    const bool known = sources_known(side, snapshot.prefix);
    const double inclusion =
        known ? cursor.inclusion(side, members, snapshot.prefix) : 0.;
    for (const auto &origin : row.origins) {
      if (!origin.prior || !known) {
        probabilities[origin.key] = nullptr;
      } else {
        auto probability = inclusion;
        const auto count = copies[side].at(origin.key);
        if (count > 1) {
          probability = probability == 1.
                            ? 1.
                            : -std::expm1(count * std::log1p(-probability));
          approximate = true;
        }
        probabilities[origin.key] = *origin.prior * probability;
      }
    }
    value["_sample_origin_probabilities"] = std::move(probabilities);
    if (approximate)
      value["_sample_origin_approximate"] = true;
    return value;
  }
};

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
void JoinProbeEvidence::prepare_snapshot(std::size_t prefix,
                                         double input_mass) const {
  const auto &s = *impl_;
  prefix = std::min(prefix, s.records.size());
  if (!prefix)
    throw std::logic_error("Empty Join prefix");
  if (s.prepared && s.prepared->prefix == prefix &&
      s.prepared->input_mass == input_mass)
    return;
  auto out = std::make_shared<JoinProbeSnapshot>();
  out->prefix = prefix;
  out->input_mass = input_mass;
  out->census = prefix == population();
  for (const auto &input : s.inputs)
    out->census &= !sampled_population(*input);
  const auto &totals = s.records[prefix - 1].totals;
  out->normalization = out->census ? 1. : input_mass / totals.mass.get();
  out->effective = totals.mass.get() * totals.mass.get() / totals.squares.get();
  std::array<Sum, 2> mass, squares;
  for (std::size_t side = 0; side < s.inputs.size(); ++side)
    for (const auto &entry : s.representatives[side]) {
      const auto &r = entry.second;
      if (r.first >= prefix)
        continue;
      const auto w = out->census ? r.base_weight : r.weight;
      mass[side].add(w);
      squares[side].add(w * w);
    }
  for (std::size_t side = 0; side < s.inputs.size(); ++side)
    out->effective =
        std::min(out->effective,
                 mass[side].get() * mass[side].get() / squares[side].get());
  bool incomplete = false;
  out->probed = s.sources(s.arrived, prefix, out->census, false, &incomplete);
  out->arrived = out->probed;
  out->arrived.approximate |= incomplete;
  s.prepared = std::move(out);
}

std::shared_ptr<const JoinProbeSnapshot>
JoinProbeEvidence::snapshot(std::size_t prefix, double input_mass) const {
  const auto &s = *impl_;
  prefix = std::min(prefix, s.records.size());
  if (!prefix || prefix > s.completed)
    throw std::logic_error("Unresolved Join prefix");
  std::uint64_t mass_key;
  static_assert(sizeof(mass_key) == sizeof(input_mass));
  std::memcpy(&mass_key, &input_mass, sizeof(mass_key));
  for (auto it = s.publications.begin(); it != s.publications.end();) {
    auto retained = it->snapshot.lock();
    if (!retained) {
      it = s.publications.erase(it);
      continue;
    }
    if (it->prefix == prefix && it->mass == mass_key)
      return retained;
    ++it;
  }
  prepare_snapshot(prefix, input_mass);
  auto out = std::move(s.prepared);
  out->owner = shared_from_this();
  out->output_mass =
      s.records[prefix - 1].totals.positive.get() * out->normalization;
  out->passed = s.sources(s.passed, prefix, out->census, true);
  for (const auto *order : {&s.positive_novel, &s.positive_deferred}) {
    const auto end = std::lower_bound(order->begin(), order->end(), prefix);
    out->matches.insert(out->matches.end(), order->begin(), end);
    if (order == &s.positive_novel)
      out->novel_matches = out->matches.size();
  }
  s.publications.push_back({prefix, mass_key, out});
  while (s.publications.size() > Impl::publication_capacity)
    s.publications.pop_front();
  return out;
}
std::vector<Json> JoinProbeEvidence::unit(std::size_t index,
                                          const JoinProbeSnapshot *snapshot,
                                          bool full_values) const {
  const auto &s = *impl_;
  const auto &c = s.cursor.at(index);
  std::vector<Json> unit;
  for (std::size_t side = 0; side < s.inputs.size(); ++side)
    unit.push_back(
        snapshot ? s.sampled_row(side, c.indices[side], c.members[side],
                                 *snapshot, full_values)
        : s.inputs.size() == 1
            ? row_at(*s.inputs[side], c.indices[side])
            : projected_row_at(*s.inputs[side], c.indices[side], s.payload));
  if (snapshot && !snapshot->census)
    sample_factor(unit[0], sample_factor(unit[0]) * c.members[0] *
                               c.members[1] * snapshot->normalization);
  return unit;
}
Json JoinProbeEvidence::output(std::size_t index,
                               const JoinProbeSnapshot &snapshot) const {
  auto rows = unit(index, &snapshot, true);
  auto value =
      rows.size() == 1 ? std::move(rows[0]) : combine(rows[0], rows[1]);
  for (auto it = value["values"].begin(); it != value["values"].end();)
    if (it.key().rfind("__semantic_input:", 0) == 0)
      it = value["values"].erase(it);
    else
      ++it;
  if (!snapshot.census)
    value["_sample_match_incomplete"] = true;
  return value;
}
const Json &JoinProbeEvidence::value(std::size_t index,
                                     const std::string &field,
                                     bool right) const {
  const auto &s = *impl_;
  const auto &c = s.cursor.at(index);
  const auto *value = find_value(*s.inputs.at(right ? 1 : 0),
                                 c.indices.at(right ? 1 : 0), field);
  if (!value)
    throw std::out_of_range("Missing semantic input field");
  return *value;
}
const std::string &JoinProbeEvidence::identity(std::size_t index) const {
  const auto &s = *impl_;
  if (const auto cached = s.identities.find(index);
      cached != s.identities.end())
    return cached->second;
  const auto &c = s.cursor.at(index);
  auto identity =
      s.inputs.size() == 1
          ? s.rows[0].at(c.indices[0]).metadata().at("id").get<std::string>()
          : digest(
                Json::array({s.rows[0].at(c.indices[0]).metadata().at("id"),
                             s.rows[1].at(c.indices[1]).metadata().at("id")}));
  const auto cached = s.identities.emplace(index, std::move(identity)).first;
  const auto coordinate = s.identity_coordinates.emplace(cached->second, index);
  // A cached coordinate returned above, so an existing identity belongs to
  // another coordinate and must use the duplicate-validation fallback.
  s.unique_identities &= coordinate.second;
  ++s.identity_computations;
  return cached->second;
}
std::size_t JoinProbeEvidence::identity_computations() const {
  return impl_->identity_computations;
}
std::shared_ptr<const JoinOutputStream::GeneratedPopulation>
JoinProbeEvidence::output_population(
    std::shared_ptr<const JoinProbeSnapshot> snapshot) const {
  if (snapshot->owner.get() != this)
    throw std::invalid_argument("Join population belongs to another owner");
  auto &s = *impl_;
  for (; s.identity_indexed_prefix < snapshot->prefix;
       ++s.identity_indexed_prefix)
    if (s.records.at(s.identity_indexed_prefix).success)
      identity(s.identity_indexed_prefix);
  // Custom inputs with duplicate identities use full vector-based validation.
  if (!s.unique_identities)
    return {};
  return std::make_shared<const JoinOutputStream::GeneratedPopulation>(
      JoinOutputStream::GeneratedPopulation{
          snapshot->owner, snapshot->matches.size(),
          [snapshot](std::size_t position) -> const std::string & {
            return snapshot->owner->identity(snapshot->matches.at(position));
          },
          [snapshot](const std::string &id) -> std::optional<std::size_t> {
            const auto &owner = *snapshot->owner->impl_;
            const auto found = owner.identity_coordinates.find(id);
            if (found == owner.identity_coordinates.end())
              return std::nullopt;
            const auto coordinate = found->second;
            const auto begin =
                snapshot->matches.begin() + (owner.records.at(coordinate).novel
                                                 ? 0
                                                 : snapshot->novel_matches);
            const auto end =
                owner.records.at(coordinate).novel
                    ? snapshot->matches.begin() + snapshot->novel_matches
                    : snapshot->matches.end();
            const auto position = std::lower_bound(begin, end, coordinate);
            return position != end && *position == coordinate
                       ? std::optional<std::size_t>(position -
                                                    snapshot->matches.begin())
                       : std::nullopt;
          },
          true});
}
std::vector<std::pair<std::size_t, double>>
JoinProbeEvidence::quality(const JoinProbeSnapshot &snapshot,
                           const Counts &counts) const {
  const auto &s = *impl_;
  std::vector<std::pair<std::size_t, double>> out;
  std::map<std::size_t, std::size_t> groups;
  std::vector<Sum> weights;
  for (const auto *order : {&s.novel, &s.deferred})
    for (const auto i : *order) {
      if (i >= snapshot.prefix)
        continue;
      auto [group, fresh] = groups.emplace(s.records[i].parameter, out.size());
      if (fresh) {
        out.emplace_back(i, 0.);
        weights.emplace_back();
      }
      const auto &c = s.cursor.at(i);
      double value = snapshot.census ? 1.
                                     : double(c.members[0]) * c.members[1] *
                                           snapshot.normalization;
      for (std::size_t side = 0; side < s.inputs.size(); ++side) {
        const auto &row = s.rows[side].at(c.indices[side]);
        value *=
            sample_factor(row.metadata()) * s.quota_weight(row.recipe, &counts);
      }
      weights[group->second].add(value);
    }
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i].second = weights[i].get();
  return out;
}
std::size_t JoinProbeEvidence::distinct(const std::string &field,
                                        std::size_t prefix) const {
  const auto found = impl_->passed_values.find(field);
  if (found == impl_->passed_values.end())
    return 0;
  std::size_t count = 0;
  for (const auto &entry : found->second)
    count += entry.second < prefix;
  return count;
}
std::size_t JoinProbeEvidence::population() const {
  return impl_->cursor.population();
}
std::size_t JoinProbeEvidence::group_population() const {
  return impl_->cursor.group_population();
}
std::size_t JoinProbeEvidence::size() const { return impl_->records.size(); }
const OwnedInputs &JoinProbeEvidence::inputs() const { return impl_->inputs; }
}
