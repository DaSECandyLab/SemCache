#include "../detail.hpp"
#include "../../common/prefix_selectivity.hpp"

namespace semcache::sampling {
Json NativeSampler::Impl::depth_selectivity(const CoreObservation &core,
                                            const Json &info,
                                            const std::string &relation,
                                            const std::string &field) const {
  const SurvivalKey cache_key{core.estimate_serial, core.cohort->serial,
                              relation, field};
  if (auto found = selection_fields.find(cache_key);
      found != selection_fields.end())
    return *found->second;
  const auto &summary = *relations.at(relation).summaries.at(field);
  const auto &op = ops.at(*core.node->operator_id);
  const auto kind = op.value("kind", op.value("type", std::string()));
  const bool preserving = prefix_preserving_operator(kind);
  Json scalar =
      preserving ? Json(1.) : info.value("selectivity", Json(nullptr));
  // Join survival is source-row match probability, not pair acceptance.
  if (join_operator(op, core.node->children.size()) &&
      !info.value("selectivity_floor_applied", false)) {
    for (std::size_t side = 0; side < core.node->children.size(); ++side) {
      if (!core.inputs.empty() && side < core.inputs.size()) {
        const auto present = origins(*core.inputs[side]);
        if (present.count(relation))
          scalar = info.value(side ? "right_selectivity" : "left_selectivity",
                              scalar);
      }
    }
  }
  const auto bank = info.find("depth_selectivity_statistics");
  const auto key = relation + ":" + field;
  const Json *evidence =
      bank != info.end() && bank->contains(key) ? &bank->at(key) : nullptr;
  const auto dense = complete_prefix_selectivity(
      prefix_depth_support(summary), evidence ? *evidence : Json::object(),
      scalar, preserving, relations.at(relation).population == 0,
      info.value("selectivity_floor_applied", false));
  Json result = Json::object(), sources = Json::object();
  for (auto depth = dense.begin(); depth != dense.end(); ++depth) {
    result[depth.key()] = depth.value().at("rate");
    sources[depth.key()] = depth.value().at("source");
  }
  if (selection_source_fields.size() >= 512)
    selection_source_fields.erase(selection_source_fields.begin());
  selection_source_fields[cache_key] =
      std::make_shared<const Json>(std::move(sources));
  if (selection_fields.size() >= 512)
    selection_fields.erase(selection_fields.begin());
  selection_fields[cache_key] = std::make_shared<const Json>(result);
  return result;
}

Json NativeSampler::Impl::blend_depths(
    const CoreObservation &core, const Json &info, const std::string &relation,
    const std::string &field, const std::map<std::string, bool> &depth_support,
    const Json &evidence) const {
  const SurvivalKey cache_key{core.estimate_serial, core.cohort->serial,
                              relation, field};
  if (auto found = selection_fields.find(cache_key);
      found != selection_fields.end())
    return *found->second;
  const auto &op = ops.at(*core.node->operator_id);
  const auto kind = op.value("kind", std::string());
  const bool preserving = prefix_preserving_operator(kind);
  const auto scalar =
      preserving ? Json(1.) : info.value("selectivity", Json(nullptr));
  const auto dense =
      complete_prefix_selectivity(depth_support, evidence, scalar, preserving,
                                  false, info.value("selectivity_floor_applied", false));
  Json result = Json::object(), sources = Json::object();
  for (auto depth = dense.begin(); depth != dense.end(); ++depth) {
    result[depth.key()] = depth.value().at("rate");
    sources[depth.key()] = depth.value().at("source");
  }
  if (selection_fields.size() >= 512)
    selection_fields.erase(selection_fields.begin());
  if (selection_source_fields.size() >= 512)
    selection_source_fields.erase(selection_source_fields.begin());
  selection_fields[cache_key] = std::make_shared<const Json>(result);
  selection_source_fields[cache_key] =
      std::make_shared<const Json>(std::move(sources));
  return result;
}

Json NativeSampler::Impl::depth_selectivity_fusion(
    const CoreObservation &core, const std::string &relation,
    const std::string &field) const {
  const SurvivalKey key{core.estimate_serial, core.cohort->serial, relation,
                        field};
  const auto found = selection_fusion_fields.find(key);
  return found == selection_fusion_fields.end() ? Json::object()
                                                : *found->second;
}

Json NativeSampler::Impl::depth_selectivity_sources(
    const CoreObservation &core, const std::string &relation,
    const std::string &field) const {
  const SurvivalKey cache_key{core.estimate_serial, core.cohort->serial,
                              relation, field};
  const auto found = selection_source_fields.find(cache_key);
  return found == selection_source_fields.end() ? Json::object()
                                                : *found->second;
}
}
