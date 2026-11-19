#include "probe_internal.hpp"

namespace semcache::sampling::resident_detail {
SourceEvidence JoinProbeEvidence::Impl::sources(const std::map<std::string, Source> &bank,
                       std::size_t prefix, bool census, bool matches,
                       bool *incomplete) const {
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
Json JoinProbeEvidence::Impl::sampled_row(std::size_t side, std::size_t index, std::size_t members,
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
