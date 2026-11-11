#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::relational_fallback_prior(const NodePtr &node,
                                                    const Json &op,
                                                    const StateData &data) {
  // Only a certified deterministic predicate over existing base fields may
  // relax its preceding semantic conditions. Generated fields are barriers.
  if (!settings.value("relational_fallback_sampling", true) || !backend ||
      deadline_reached() || !op.value("sampling_row_local", false) ||
      op.value("kind", std::string()) != "FILTER")
    return Json::object();
  const auto fields = op.value("sampling_required_fields", Json::array());
  const auto sql = op.value("sampling_row_local_sql", std::string());
  if (fields.empty() || sql.empty())
    return Json::object();
  // Do not replace null-extending joins, aggregation or row bounds with a
  // base Cartesian domain. Inner joins lowered to CROSS_PRODUCT + FILTER
  // are supported; other structural operators remain barriers.
  std::vector<Id> pending(node->children.begin(), node->children.end());
  std::set<Id> visited;
  std::size_t relaxed_conditions = 0;
  const std::set<std::string> allowed{
      "FILTER",          "CROSS_PRODUCT",       "PROJECTION",   "ORDER_BY",
      "SEMANTIC_FILTER", "SEMANTIC_PROJECTION", "SEMANTIC_JOIN"};
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!visited.insert(id).second)
      continue;
    const auto &ancestor = data.observations.at(id)->core->node;
    if (ancestor->operator_id) {
      const auto &description = ops.at(*ancestor->operator_id);
      auto kind =
          description.value("kind", description.value("type", std::string()));
      std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
      if (!allowed.count(kind))
        return Json::object();
      relaxed_conditions += kind == "FILTER" || kind == "SEMANTIC_FILTER" ||
                            kind == "SEMANTIC_JOIN";
    }
    pending.insert(pending.end(), ancestor->children.begin(),
                   ancestor->children.end());
  }
  std::map<std::string, std::set<std::string>> required;
  for (const auto &field : fields) {
    const auto name = field.get<std::string>();
    bool found = false;
    for (const auto &raw : node->relations) {
      const auto key = node_relkey(raw);
      const auto &rows = relations.at(key).admitted;
      if (!rows.empty() && rows[0].at("values").contains(name)) {
        required[key].insert(name);
        found = true;
        break;
      }
    }
    if (!found)
      return Json::object();
  }
  Json recipe = Json::array({sql, relational_sample_rows});
  for (const auto &[key, names] : required) {
    Json selected = Json::array();
    for (const auto &stratum : relations.at(key).strata) {
      if (stratum.population > 0 && !stratum.selected)
        return Json::object();
      selected.push_back(stratum.selected);
    }
    recipe.push_back(Json::array(
        {key, relations.at(key).version, std::move(selected), names}));
  }
  const auto key = canonical(recipe);
  const auto oid = *node->operator_id;
  auto cached = relational_fallback_cache.find(oid);
  if (cached != relational_fallback_cache.end() &&
      cached->second.first == key) {
    counters["independent_relational_cache_hits"] =
        counters.value("independent_relational_cache_hits", std::size_t(0)) + 1;
    Json result = cached->second.second;
    const double n = result.at("independent_relational_effective_samples");
    result["independent_relational_relaxed_conditions"] = relaxed_conditions;
    result["reliability"]["selectivity"] =
        n / (n + input_samples) / (1. + relaxed_conditions);
    return result;
  }
  PhaseTimer timer(phase_metrics, "independent_relational_sampling",
                   trace_timings, true);
  Json population;
  Counts quotas;
  for (const auto &[relation, names] : required) {
    const auto &source = relations.at(relation);
    for (std::size_t i = 0; i < source.strata.size(); ++i)
      quotas[{relation, i}] = source.strata[i].selected;
    Json bank = Json::array();
    for (const auto &row : source.admitted) {
      Json values = Json::object();
      for (const auto &name : names)
        values[name] = row.at("values").at(name);
      Json projected = {{"id", row.at("id")},
                        {"values", std::move(values)},
                        {"lineage", row.at("lineage")}};
      copy_sample_metadata(projected, row);
      bank.push_back(std::move(projected));
    }
    if (population.is_null())
      population = std::move(bank);
    else
      population = {
          {"product", Json::array({std::move(population), std::move(bank)})}};
  }
  // Probe at most the relational allowance, without materializing a base
  // Cartesian product or reducing the normal semantic sampling cohort.
  resident_detail::JoinOutputStream stream;
  auto rows = std::make_shared<const Json>(
      stream.take(population, relational_sample_rows,
                  std::stoull(digest_text(key).substr(0, 16), nullptr, 16)));
  if (rows->empty() || deadline_reached())
    return Json::object();
  Json probe = op;
  probe["sql"] = sql;
  probe["input_columns"] = Json::array({columns(*rows)});
  auto output = execute(probe, OwnedInputs{rows}, nullptr, nullptr,
                        "relational_fallback");
  const double incoming = mass(*rows, &quotas),
               outgoing = mass(output, &quotas), n = effective(*rows, &quotas);
  if (!(incoming > 0) || !std::isfinite(incoming) || !std::isfinite(outgoing) ||
      !(n > 0))
    return Json::object();
  const double point = outgoing / incoming;
  bool census = row_count(population) == rows->size();
  for (const auto &[relation, names] : required)
    for (const auto &stratum : relations.at(relation).strata)
      census &= stratum.selected == stratum.population;
  const double rate = point == 0. && !census ? .5 / (n + 1.) : point;
  Json result = {
      {"selectivity", rate},
      {"independent_relational_prior", true},
      {"independent_relational_point_estimate", point},
      {"independent_relational_sample_rows", rows->size()},
      {"independent_relational_effective_samples", n},
      {"independent_relational_census", census},
      {"cardinality_basis",
       "bounded independent base-field predicate sample; "
       "preceding filters relaxed within an inner-relation domain"},
      {"independent_relational_relaxed_conditions", relaxed_conditions},
      {"reliability",
       {{"selectivity", n / (n + input_samples) / (1. + relaxed_conditions)}}}};
  relational_fallback_cache[oid] = {key, result};
  counters["independent_relational_samples"] =
      counters.value("independent_relational_samples", std::size_t(0)) + 1;
  return result;
}
}
