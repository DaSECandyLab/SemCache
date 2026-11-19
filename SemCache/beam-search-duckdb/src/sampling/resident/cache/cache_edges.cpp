#include "../detail.hpp"
#include "../../common/accuracy.hpp"
#include "semcache/history_features.hpp"

namespace semcache::sampling {
void NativeSampler::Impl::cache_edges(StateData &data) {
  PhaseTimer timer(phase_metrics, "cache_edges", trace_timings);
  std::map<Id, std::set<Id>> ancestors;
  auto nodes = data.state->nodes();
  for (auto &node : nodes) {
    for (auto child : node->children) {
      ancestors[node->node_id].insert(child);
      ancestors[node->node_id].insert(ancestors[child].begin(),
                                      ancestors[child].end());
    }
    if (!node->operator_id)
      continue;
    const auto &op = ops.at(*node->operator_id);
    if (!semantic_operator(op))
      continue;
    const auto &action = action_for(data.state, *node->operator_id);
    auto &overlay = [&]() -> Json & {
      PhaseTimer detach(phase_metrics, "cache_edges_detach", trace_timings);
      return data.stats.at(node->node_id).write();
    }();
    std::vector<Json> sizes;
    for (auto child : node->children)
      sizes.push_back(
          data.stats.at(child).read().value("output_rows", Json(nullptr)));
    propagate_inplace(op, overlay, sizes);
    const auto &observation = *data.observations.at(node->node_id);
    overlay.erase("result_cache");
    const auto &info = observation.statistics.read();
    if (info.value("survival_estimated", false)) {
      overlay["output_rows"] = info.value("output_rows", Json(nullptr));
      overlay["selectivity"] = info.value("selectivity", Json(nullptr));
    }
    const auto &quality = accuracy_evidence(
        profile_statistic("accuracy", action.at("model_id"), op.at("kind")),
        action.at("layout"));
    const auto &agreement = quality.is_object() && quality.contains("value")
                                ? quality.at("value")
                                : quality;
    if (finite_number(agreement))
      overlay["accuracy"] = agreement;
    else
      overlay.erase("accuracy");
    if (info.value("pair_input", false) &&
        !info.value("pair_request_summary", false)) {
      overlay["l_self"] = nullptr;
      overlay["l_existing"] = Json::object();
      overlay["l_previous"] = Json::object();
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          Json::array({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
      continue;
    }
    if (!observation.summary) {
      // Precision/budget annotations may survive a replay. Cache facts cannot
      // survive loss of the current input evidence, even if once known.
      for (auto key : {"alpha", "right_alpha", "pre", "pre_length",
                       "left_total_length", "conditional_sketch_plan", "basis"})
        overlay.erase(key);
      overlay["l_self"] = nullptr;
      overlay["l_existing"] = Json::object();
      overlay["l_previous"] = Json::object();
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          Json::array({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
      continue;
    }
    auto alpha = info.value("alpha", Json(1.));
    if (observation.core->input_match_incomplete)
      for (const auto *key : {"l_self", "l_existing", "l_previous"})
        overlay["reliability"][key] = 0.;
    bool right = observation.right_summary && !info.value("pair_input", false);
    const auto requests = overlay.value("input_rows", Json(nullptr));
    const auto sides = overlay.value("input_side_rows", Json::array());
    const Json left_requests =
                   right && sides.size() == 2 ? sides[0] : Json(nullptr),
               right_requests =
                   right && sides.size() == 2 ? sides[1] : Json(nullptr);
    const auto model = action.at("model_id").get<std::string>();
    {
      PhaseTimer self_timer(phase_metrics, "cache_edges_self", trace_timings);
      overlay["l_self"] =
          right
              ? reuse_cache.join(observation.summary, observation.right_summary,
                                 info.value("left_total_length", Json(nullptr)),
                                 alpha, info.value("right_alpha", Json(1.)),
                                 left_requests, right_requests)
              : reuse_cache.self(observation.summary, alpha, requests);
    }
    overlay["l_existing"] = Json::object();
    overlay["l_previous"] = Json::object();
    overlay["cache_sources_complete"] =
        !info.value("pre", Json(nullptr)).is_null();
    overlay["cache_dependence"] = Json::object();
    Json right_rate = right ? survival_rate(info.value("right_alpha", Json(1.)))
                            : Json(nullptr);
    if (right && right_rate.is_null()) {
      overlay["cache_sources_complete"] = false;
      overlay["l_existing_features"] =
          semcache::statistics::existing_history_features(history_, overlay,
                                                          model);
      continue;
    }
    const auto pre = info.value("pre", Json(nullptr));
    semcache::statistics::ExistingHistoryFeatures packed_history(pre, model);
    const auto indexed_history = history_by_model.find(model);
    if (indexed_history != history_by_model.end())
      for (const auto *source : indexed_history->second)
        packed_history.observe(*source);
    const Json cross_requests =
        (right && sides.is_array() && sides.size() == 2 &&
         finite_number(sides[0]) && finite_number(sides[1]))
            ? ((sides[0].get<double>() <= 0. || sides[1].get<double>() <= 0.)
                   ? Json(0.)
                   : Json(sides[0].get<double>() *
                          std::max(1., sides[1].get<double>())))
            : requests;
    auto edge = [&](const std::shared_ptr<const Json> &source,
                    const Json &source_alpha,
                    const CrossDependence &dependence = CrossDependence{}) {
      auto value = reuse_cache.cross(source, observation.summary, source_alpha,
                                     alpha, dependence, cross_requests);
      return value;
    };
    {
      PhaseTimer history_timer(phase_metrics, "cache_edges_existing",
                               trace_timings);
      if (history_incomplete_models.count(model))
        overlay["cache_sources_complete"] = false;
      const auto model_history = history_by_prefix.find(model);
      if (model_history != history_by_prefix.end()) {
        const auto matching = model_history->second.find(pre);
        if (matching != model_history->second.end())
          for (const auto *source : matching->second) {
            const auto id = source->at("id").get<std::string>();
            overlay["l_existing"][id] =
                source->contains("summary") && !source->at("summary").is_null()
                    ? edge(history_summaries.at(id), 1.)
                    : Json(nullptr);
          }
      }
    }
    {
      PhaseTimer previous_timer(phase_metrics, "cache_edges_previous",
                                trace_timings);
      for (auto &old : nodes) {
        if (!old->operator_id || !ancestors[node->node_id].count(old->node_id))
          continue;
        const auto &old_action = action_for(data.state, *old->operator_id);
        if (old_action.value("model_id", Json(nullptr)) != action["model_id"])
          continue;
        const auto &old_observation = *data.observations.at(old->node_id);
        const auto &source = old_observation.statistics.read();
        if (source.value("pre", Json(nullptr)).is_null())
          overlay["cache_sources_complete"] = false;
        else if (source.value("pre", Json(nullptr)) ==
                 info.value("pre", Json(nullptr))) {
          auto value =
              old_observation.summary
                  ? edge(old_observation.summary,
                         source.value("alpha", Json(1.)),
                         cache_dependence(old_observation, observation))
                  : Json(nullptr);
          overlay["cache_dependence"][std::to_string(*old->operator_id)] =
              old_observation.cache_relation.empty() ||
                      observation.cache_relation.empty()
                  ? "unknown-field independent approximation"
              : old_observation.cache_relation != observation.cache_relation
                  ? "cross-relation independent approximation"
              : old_observation.cache_field == observation.cache_field
                  ? "same-field lineage containment"
              : relations.at(observation.cache_relation).offline_cooccurrence &&
                      relations.at(observation.cache_relation)
                          .offline_columns.count(old_observation.cache_field) &&
                      relations.at(observation.cache_relation)
                          .offline_columns.count(observation.cache_field)
                  ? "same-relation offline conditional row samples"
                  : "same-relation offline statistics unavailable; independent "
                    "approximation";
          overlay["l_previous"][std::to_string(*old->operator_id)] = value;
        }
      }
    }
    overlay["l_existing_features"] =
        packed_history.finish(overlay["l_existing"]);
  }
}
}
