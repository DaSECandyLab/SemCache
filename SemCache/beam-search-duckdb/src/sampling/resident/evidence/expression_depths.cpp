#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::expression_depths(const Json &op,
                                            const CoreObservation &core) const {
  PhaseTimer timer(phase_metrics, "expression_depth_evidence", trace_timings,
                   true);
  std::set<std::string> fields;
  for (const auto &field : input_keys(op)) {
    bool base = false;
    for (const auto &raw : core.node->relations)
      base |= relations.at(node_relkey(raw)).summaries.count(field) != 0;
    if (!base)
      fields.insert(field);
  }
  if (fields.empty() || (!core.decisions && !core.join_evidence))
    return Json::object();
  Counts quotas;
  for (const auto &[relation, selected] : core.cohort->selected)
    for (std::size_t i = 0; i < selected.size(); ++i)
      quotas[{relation, i}] = selected[i];
  struct Row {
    std::size_t length;
    double weight;
    bool passed;
  };
  std::map<std::string, std::map<std::string, Row>> rows;
  auto visit = [&](const Json &row, bool accepted) {
    const auto id = canonical(row.at("lineage"));
    for (const auto &field : fields) {
      const auto value = row.at("values").find(field);
      if (value == row.at("values").end())
        continue;
      const auto length = character_length(serialize_value(*value));
      const double weight = raw_weight(row, &quotas);
      auto [found, inserted] =
          rows[field].try_emplace(id, Row{length, weight, accepted});
      // A source tuple with any observed accepted counterpart survives. The
      // other-side multiplicity is not another observation of this source.
      if (!inserted)
        found->second.passed |= accepted;
    }
  };
  if (core.join_evidence) {
    const auto &snapshot = *core.join_evidence;
    std::set<std::size_t> matched(snapshot.matches.begin(),
                                  snapshot.matches.end());
    for (std::size_t i = 0; i < snapshot.prefix; ++i)
      for (const auto &row : snapshot.owner->unit(i, &snapshot, true))
        visit(row, matched.count(i));
  } else if (core.decisions && core.probe_units) {
    // Bounded expression Joins use prepared probe units; their decision
    // positions are not coordinates in the original Cartesian product.
    core.decisions->each_known([&](std::size_t position) {
      for (const auto &row : core.probe_units->at(position))
        visit(row, core.decisions->accepted(position));
    });
  } else if (core.decisions && !core.inputs.empty()) {
    SampleInputs inputs(core.inputs);
    // Factorized unary requests index a physical leaf, not the whole product.
    auto leaf =
        inputs.size() == 1 ? leaf_for(inputs[0], input_keys(op)) : std::nullopt;
    const Json *first = leaf ? &leaf_at(inputs[0], *leaf) : &inputs[0];
    const auto right = inputs.size() == 2 ? row_count(inputs[1]) : 1;
    if (right)
      core.decisions->each_known([&](std::size_t position) {
        const auto row = row_at(*first, position / right);
        visit(row, core.decisions->accepted(position));
        if (inputs.size() == 2)
          visit(row_at(inputs[1], position % right),
                core.decisions->accepted(position));
      });
  }
  Json result = Json::object();
  for (const auto &[field, values] : rows) {
    // A suffix scan over observed lengths is O(rows + depths), and builds no
    // additional sketch or large row copy. It only consumes paid decisions.
    struct Totals {
      double weight = 0, squares = 0, passed_weight = 0;
      std::size_t observed = 0, passed = 0;
    };
    std::map<std::size_t, Totals, std::greater<std::size_t>> bins;
    for (const auto &[_, row] : values) {
      auto &bin = bins[row.length];
      bin.weight += row.weight;
      bin.squares += row.weight * row.weight;
      bin.passed_weight += row.passed ? row.weight : 0.;
      ++bin.observed;
      bin.passed += row.passed;
    }
    Totals total;
    auto emit = [&] {
      return Json{
          {"rate",
           total.weight > 0
               ? Json(std::clamp(total.passed_weight / total.weight, 0., 1.))
               : Json(nullptr)},
          {"observed", total.observed},
          {"passed", total.passed},
          {"effective_samples",
           total.squares > 0 ? total.weight * total.weight / total.squares
                             : 0.},
          {"basis", "distinct evaluated expression-input tuples; conditional "
                    "input domain"}};
    };
    auto &bank = result["generated:" + field];
    for (const auto &[length, bin] : bins) {
      total.weight += bin.weight;
      total.squares += bin.squares;
      total.passed_weight += bin.passed_weight;
      total.observed += bin.observed;
      total.passed += bin.passed;
      bank[std::to_string(length)] = emit();
    }
    bank["0"] = emit();
    std::set<std::size_t> depths;
    for (std::size_t d = 1; !bins.empty() && d <= bins.begin()->first; d *= 2) {
      depths.insert(d);
      if (d > std::numeric_limits<std::size_t>::max() / 2)
        break;
    }
    for (const auto &raw : core.node->relations)
      for (const auto &[_, summary] : relations.at(node_relkey(raw)).summaries)
        for (auto d = summary->at("sketches").begin();
             d != summary->at("sketches").end(); ++d)
          if (d.key() != "0")
            depths.insert(std::stoull(d.key()));
    std::map<std::size_t, Json> suffix;
    for (auto d = bank.begin(); d != bank.end(); ++d)
      if (d.key() != "0")
        suffix.emplace(std::stoull(d.key()), d.value());
    for (const auto d : depths) {
      const auto found = suffix.lower_bound(d);
      if (found != suffix.end())
        bank[std::to_string(d)] = found->second;
    }
  }
  return result;
}
}
