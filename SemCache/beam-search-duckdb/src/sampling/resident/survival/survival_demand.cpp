#include "../detail.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::input_survival(const CoreObservation &core,
                                         const StateData &data,
                                         const std::string &relation,
                                         const std::string &field) const {
  PhaseTimer timer(phase_metrics, "survival_input", trace_timings, true);
  // Incoming rows are sampled independently and do not measure survival.
  // Survival belongs to this candidate's actual predecessor graph instead.
  for (const auto child : core.node->children) {
    const auto &source = *data.observations.at(child)->core;
    for (const auto &raw : source.node->relations)
      if (node_relkey(raw) == relation)
        return output_survival(source, data, relation, field);
  }
  return Json::object();
}

Json NativeSampler::Impl::output_survival(const CoreObservation &core,
                                          const StateData &data,
                                          const std::string &relation,
                                          const std::string &field) const {
  PhaseTimer timer(phase_metrics, "survival_output", trace_timings, true);
  const auto &info = *core.statistics;
  const auto published = info.find("estimated_output_survival");
  const auto key = relation + ":" + field;
  if (published != info.end() && published->contains(key))
    return published->at(key);
  const SurvivalKey cache_key{core.estimate_serial, core.cohort->serial,
                              relation, field};
  if (auto found = survival_fields.find(cache_key);
      found != survival_fields.end())
    return *found->second;
  Json result;
  if (!core.node->operator_id) {
    const auto &source = relations.at(relation);
    result = {{"0", source.population ? 1. : 0.}};
    const auto &summary = *source.summaries.at(field);
    for (auto depth = summary.at("sketches").begin();
         depth != summary.at("sketches").end(); ++depth)
      result[depth.key()] = depth.value().at("entries").empty() ? 0. : 1.;
  } else {
    result = input_survival(core, data, relation, field);
    const auto survival = depth_selectivity(core, info, relation, field);
    for (auto depth = result.begin(); depth != result.end(); ++depth) {
      const auto &factor = survival.at(depth.key());
      depth.value() =
          depth.value() == 0 || factor == 0 ? Json(0.)
          : finite_number(depth.value()) && finite_number(factor)
              ? Json(depth.value().get<double>() * factor.get<double>())
              : Json(nullptr);
    }
  }
  // Do not pin States or mutable CoreObservations, or publish partial entries.
  if (survival_fields.size() >= 512)
    survival_fields.erase(survival_fields.begin());
  survival_fields.emplace(cache_key, std::make_shared<const Json>(result));
  return result;
}
}
