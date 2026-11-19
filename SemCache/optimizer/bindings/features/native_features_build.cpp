#include "native_features_internal.hpp"

namespace semcache::python_features_detail {
NativeFeatureContext::NativeFeatureContext(py::dict value)
    : config(from_python(value)), chunk_size(config.at("chunk_size")),
      cache_features_enabled(config.value("use_cache_features", true)),
      initial_history(std::make_shared<const HistoryIndex>(config.at("history"))) {
  if (!chunk_size)
    throw std::invalid_argument(
        "Native inference chunk size must be positive");
  if (config.contains("workers")) {
    const auto &workers = config.at("workers");
    if (!workers.is_number_integer() ||
        (workers.is_number_unsigned() ? workers.get<std::uint64_t>() == 0
                                      : workers.get<std::int64_t>() <= 0))
      throw std::invalid_argument(
          "Native inference workers must be positive integers");
    worker_limit = std::min(std::size_t(32), workers.get<std::size_t>());
  }
  const auto &desc_obj = config.at("descriptors");
  descriptors.reserve(desc_obj.size() * 2 + 1);
  for (auto it = desc_obj.begin(); it != desc_obj.end(); ++it)
    descriptors.emplace(std::stoll(it.key()), &it.value());
  const auto &var_obj = config.at("variants");
  variants.reserve(var_obj.size() * 2 + 1);
  for (auto it = var_obj.begin(); it != var_obj.end(); ++it)
    variants.emplace(std::stoll(it.key()), &it.value());
  const auto &initial_profile = config.at("profile");
  if (initial_profile.contains("node_statistics") &&
      initial_profile.at("node_statistics").is_object()) {
    const auto &ns = initial_profile.at("node_statistics");
    initial_node_statistics.reserve(ns.size() * 2 + 1);
    for (auto it = ns.begin(); it != ns.end(); ++it)
      initial_node_statistics.emplace(std::stoll(it.key()), &it.value());
  }
  const auto &fb = config.at("field_buckets");
  field_buckets.reserve(fb.size());
  for (const auto &v : fb)
    field_buckets.push_back(v.get<int>());
}
Shard NativeFeatureContext::build_range(
    const std::vector<StatePtr> &states,
    const ProfileEnvironment::BatchViews &state_views, const Json &profile,
    const HistoryIndex &history_index, std::size_t begin,
    std::size_t end) const {
  Shard shard;
  auto &rows = shard.rows;
  auto &graphs = shard.graphs;
  auto &work = shard.work;
  auto &descriptions = shard.descriptions;
  auto &offsets = shard.offsets;
  std::unordered_map<std::string, std::size_t> unique_rows, unique_work;
  std::unordered_map<RecordKey, std::size_t, RecordKeyHash> record_rows;
  std::unordered_map<RecordKey, std::size_t, RecordKeyHash> node_rows;
  StaticInputMap static_inputs;
  std::unordered_map<std::uint64_t, const StaticInputMap::value_type *>
      node_static_inputs;
  std::unordered_map<RecordKey, std::vector<CachedLink>, RecordKeyHash>
      node_topology;
  std::unordered_map<std::uint64_t, const Json *> node_variants;
  std::unordered_set<Id> described;
  struct TemplateEntry {
    std::size_t ordinal;
    Json value;
    const std::string *dumped;
  };
  std::unordered_map<std::string, TemplateEntry> static_templates;
  std::unordered_map<std::string, std::size_t> template_ids;
  auto &max_nodes = shard.max_nodes;
  auto &max_degree = shard.max_degree;
  const Json *remaining_sel =
      profile.contains("remaining_selectivity") &&
              profile.at("remaining_selectivity").is_object()
          ? &profile.at("remaining_selectivity")
          : nullptr;
  std::unordered_map<Id, const Json *> remaining_sel_by_id;
  if (remaining_sel) {
    remaining_sel_by_id.reserve(remaining_sel->size() * 2 + 1);
    for (auto it = remaining_sel->begin(); it != remaining_sel->end(); ++it)
      remaining_sel_by_id.emplace(std::stoll(it.key()), &it.value());
  }
  std::unordered_map<Id, std::size_t> node_indices;
  std::vector<std::uint8_t> is_child;
  std::vector<std::pair<Id, std::size_t>> by_operator;
  std::vector<const Json *> choices;
  for (std::size_t gi = begin; gi < end; ++gi) {
    if (gi % chunk_size == 0)
      descriptions.emplace_back();
    auto describe = [&](Id nid) {
      if (described.insert(nid).second)
        descriptions.back().push_back(
            {nid, descriptors.at(nid)->at("description")});
    };
    const auto &state = states[gi];
    for (auto current = state; current && current->parent;
         current = current->parent) {
      if (node_variants.count(current->added->handle_id))
        break;
      const auto *variant =
          &variants.at(current->action.operator_id)->at(current->action.domain_index);
      node_variants.emplace(current->added->handle_id, variant);
    }
    const auto &state_entry = state_views.states[gi];
    const auto &nodes = *state_entry.nodes;
    const auto view_offset = state_entry.offset;
    choices.clear();
    choices.reserve(nodes.size());
    for (const auto &node : nodes) {
      const Json *variant = nullptr;
      if (node->operator_id) {
        const auto found = node_variants.find(node->handle_id);
        if (found != node_variants.end())
          variant = found->second;
      }
      choices.push_back(variant);
    }
    max_nodes = std::max(max_nodes, nodes.size());
    Graph graph;
    graph.rows.reserve(nodes.size());
    graph.roots.resize(nodes.size(), true);
    graph.links.reserve(nodes.size() * 3);
    graph.link_offsets.reserve(nodes.size() + 1);
    node_indices.clear();
    by_operator.clear();
    for (std::size_t ni = 0; ni < nodes.size(); ++ni) {
      const auto &node = nodes[ni];
      const auto nid = node->node_id;
      node_indices[nid] = ni;
      for (auto child : node->children)
        graph.roots[node_indices.at(child)] = false;
      const auto &descriptor = *descriptors.at(nid);
      const Json *variant = choices[ni];
      const StatisticView source_view = state_views.views[view_offset + ni];
      const void *source = source_view.identity;
      const RecordKey node_key{node->handle_id, source};
      const auto previous_node = node_rows.find(node_key);
      if (previous_node != node_rows.end()) {
        graph.rows.push_back(previous_node->second);
        describe(nid);
        if (variant && descriptor.at("semantic").get<bool>() &&
            node->operator_id)
          by_operator.emplace_back(*node->operator_id, ni);
        continue;
      }
      auto static_input = node_static_inputs.find(node->handle_id);
      if (static_input == node_static_inputs.end()) {
        std::vector<int> fields;
        fields.reserve(node->fields.size());
        for (auto field : node->fields)
          fields.push_back(field_buckets.at(field));
        std::sort(fields.begin(), fields.end());
        fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
        std::string static_key;
        static_key.reserve(32 + fields.size() * 6);
        static_key += std::to_string(nid);
        static_key.push_back(':');
        static_key +=
            (variant ? std::to_string(variant->at("index").get<std::size_t>())
                     : "-");
        static_key.append(":[");
        for (std::size_t fi = 0; fi < fields.size(); ++fi) {
          if (fi)
            static_key.push_back(',');
          static_key += std::to_string(fields[fi]);
        }
        static_key.push_back(']');
        const auto inserted =
            static_inputs.try_emplace(std::move(static_key));
        if (inserted.second) {
          inserted.first->second.ordinal = static_inputs.size() - 1;
          inserted.first->second.fields = std::move(fields);
        }
        // unordered_map preserves element addresses when it rehashes.
        static_input =
            node_static_inputs.emplace(node->handle_id, &*inserted.first)
                .first;
      }
      const auto &static_key = static_input->second->first;
      const auto &fields = static_input->second->second.fields;
      const RecordKey record_key{static_input->second->second.ordinal,
                                 source};
      auto found_record = record_rows.find(record_key);
      std::size_t row_id;
      if (found_record != record_rows.end())
        row_id = found_record->second;
      else {
        Row row;
        StatOverlay stat(source_view);
        const bool semantic = descriptor.at("semantic").get<bool>();
        if (!variant) {
          stat.defaults["input_rows"] = &descriptor.at("estimated_rows");
          stat.defaults["output_rows"] = &descriptor.at("estimated_rows");
          const auto priors_it = initial_node_statistics.find(nid);
          if (priors_it != initial_node_statistics.end() &&
              priors_it->second->is_object()) {
            const auto &priors = *priors_it->second;
            for (auto i = priors.begin(); i != priors.end(); ++i)
              stat.defaults[i.key()] = &i.value();
          }
          if (get(stat, "input_width").is_null())
            stat["input_width"] = get(stat, "output_width");
        }
        if (semantic && variant) {
          if (get(stat, "pre").is_null() && !variant->at("pre").is_null()) {
            stat["pre"] = variant->at("pre");
            // The prior may have sampled a different layout before this
            // implementation was bound. Its old length is not transferable.
            stat["pre_length"] = variant->at("pre_length");
          }
          if (get(stat, "pre_length").is_null()) {
            stat["pre_length"] = prefix_length(get(stat, "pre"));
            if (stat["pre_length"].is_null())
              stat["pre_length"] = variant->at("pre_length");
          }
          const auto q = quality(profile, *variant);
          if (q.is_object())
            stat["accuracy"] = get(q, "value");
          else if (known(q))
            stat["accuracy"] = q;
          else
            stat.erase("accuracy");
          if (!get(stat, "reliability").is_object())
            stat["reliability"] = Json::object();
          auto reliability = [&](const char *name, const Json &count,
                                 double uncertainty) {
            if (known(count) && count.get<double>() > 0.)
              stat["reliability"][name] = count.get<double>() /
                                          (count.get<double>() + 1.) /
                                          (1. + uncertainty);
          };
          reliability("accuracy", get(q, "effective_samples"), 0.);
        }
        add_input_time(stat, profile, descriptor, variant);
        const auto &kind_vec = descriptor.at("kind");
        if (kind_vec.size() != 16)
          throw std::logic_error("Unexpected node kind descriptor width");
        std::size_t index = 0;
        auto push = [&](double value) {
          if (index >= numeric_width)
            throw std::logic_error("Native numeric feature overflow");
          row.numeric[index++] = value;
        };
        for (const auto &v : kind_vec)
          push(v.get<double>());
        if (variant) {
          const auto &impl_vec = variant->at("implementation");
          if (impl_vec.size() != 71)
            throw std::logic_error(
                "Unexpected operator implementation descriptor width");
          for (const auto &v : impl_vec)
            push(v.get<double>());
        } else {
          if (index + 71 > numeric_width)
            throw std::logic_error("Native numeric feature overflow");
          index += 71;
        }
        if (variant && variant->at("has_model").get<bool>()) {
          const auto &length = get(stat, "pre_length");
          row.numeric.at(86) = log_value(
              length.is_null() ? prefix_length(get(stat, "pre")) : length);
        }
        const char *names[] = {"input_rows", "input_width", "selectivity",
                               "accuracy", "seconds_per_input"};
        std::array<std::array<double, 3>, 5> triples;
        for (std::size_t j = 0; j < 5; ++j)
          triples[j] = estimated(stat, names[j], j == 2 || j == 3,
                                 semantic || j != 2, semantic || j != 3);
        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
          for (const auto &t : triples)
            push(t[coordinate]);
        for (auto v : estimated(stat, "l_self", false, true, semantic))
          push(cache_features_enabled ? v : 0.);
        if (index != numeric_width)
          throw std::logic_error("Native feature width mismatch");
        row.has_history = cache_features_enabled && semantic && variant;
        if (row.has_history)
          row.history = history_features(history_index, *variant, stat);
        auto input_template = static_templates.find(static_key);
        if (input_template == static_templates.end()) {
          Json value = {
              {"fields",
               Json::array({descriptor.at("fields").at(0), Json(fields)})},
              {"description", descriptor.at("description")},
              {"first_fields",
               variant ? variant->at("first_fields")
                       : Json::array({Json::array(), Json::array()})}};
          auto inserted =
              template_ids.emplace(value.dump(), template_ids.size());
          input_template =
              static_templates
                  .emplace(static_key,
                           TemplateEntry{inserted.first->second,
                                         std::move(value),
                                         &inserted.first->first})
                  .first;
        }

        std::string key = std::to_string(input_template->second.ordinal);
        key.append(reinterpret_cast<const char *>(row.numeric.data()),
                   sizeof(row.numeric));
        key.push_back(row.has_history ? '1' : '0');
        key.append(reinterpret_cast<const char *>(row.history.data()),
                   sizeof(row.history));
        // Prefix/previous-match data affect links, not node projection.
        // Keep estimates separate below if otherwise identical inputs
        // coalesce.
        auto found = unique_rows.find(key);
        if (found == unique_rows.end()) {
          row_id = rows.size();
          unique_rows.emplace(std::move(key), row_id);
          row.value = input_template->second.value;
          row.template_key = *input_template->second.dumped;
          rows.push_back(std::move(row));
        } else
          row_id = found->second;
        record_rows.emplace(record_key, row_id);
      }
      node_rows.emplace(node_key, row_id);
      graph.rows.push_back(row_id);
      describe(nid);
      if (variant && descriptor.at("semantic").get<bool>() &&
          node->operator_id)
        by_operator.emplace_back(*node->operator_id, ni);
    }
    // A retained PlanNode's construction prefix and subtree never change.
    // Its ancestors' model/prefix choices are frozen; runtime feedback
    // only replaces statistics (not the bound prompt). This node's record
    // identity covers changed l_previous strengths/validity. Cache source
    // node IDs, since graph-local positions belong to the current batch.
    for (std::size_t ni = 0; ni < nodes.size(); ++ni) {
      const auto &node = nodes[ni];
      const StatisticView view = state_views.views[view_offset + ni];
      const RecordKey key{node->handle_id, view.identity};
      auto topology = node_topology.find(key);
      if (topology == node_topology.end()) {
        StatOverlay stat(view);
        const Json *variant = choices[ni];
        const auto &raw_prefix = get(stat, "pre");
        const Json &prefix =
            variant && raw_prefix.is_null() ? variant->at("pre") : raw_prefix;
        std::map<Id, std::optional<double>> cache;
        if (cache_features_enabled && variant &&
            descriptors.at(node->node_id)->at("semantic").get<bool>() &&
            !prefix.is_null() && !by_operator.empty()) {
          const auto &target_model = variant->at("model");
          bool has_same_model = false;
          for (const auto &[_, source_index] : by_operator) {
            if (choices[source_index]->at("model") == target_model) {
              has_same_model = true;
              break;
            }
          }
          if (has_same_model) {
            std::unordered_set<Id> descendants;
            std::vector<Id> pending = node->children;
            while (!pending.empty()) {
              const auto id = pending.back();
              pending.pop_back();
              if (!descendants.insert(id).second)
                continue;
              const auto &below = nodes.at(node_indices.at(id));
              pending.insert(pending.end(), below->children.begin(),
                             below->children.end());
            }
            for (const auto &[oid, source_index] : by_operator) {
              const auto source_id = nodes[source_index]->node_id;
              if (!descendants.count(source_id))
                continue;
              const auto *source_variant = choices[source_index];
              if (source_variant->at("model") != target_model)
                continue;
              const StatOverlay source_stat(
                  state_views.views[view_offset + source_index]);
              const auto &raw_source_prefix = get(source_stat, "pre");
              const auto &source_prefix = raw_source_prefix.is_null()
                                              ? source_variant->at("pre")
                                              : raw_source_prefix;
              if (source_prefix != prefix)
                continue;
              const auto &match =
                  get(get(stat, "l_previous"), std::to_string(oid).c_str());
              cache[source_id] = known(match) && match.get<double>() >= 0 &&
                                         valid(stat, "l_previous")
                                     ? std::optional<double>(log_value(match))
                                     : std::nullopt;
            }
          }
        }
        std::set<Id> sources(node->children.begin(), node->children.end());
        for (const auto &cached : cache)
          sources.insert(cached.first);
        sources.insert(node->node_id);
        std::vector<CachedLink> links;
        links.reserve(sources.size());
        for (auto source : sources) {
          const auto child =
              std::find(node->children.begin(), node->children.end(), source);
          const bool plan = child != node->children.end();
          const bool cached = source != node->node_id && cache.count(source);
          const int type = source == node->node_id ? 3
                           : plan && cached        ? 2
                           : plan                  ? 0
                                                   : 1;
          const int position =
              plan ? (node->children.size() == 1
                          ? 0
                          : 1 + std::distance(node->children.begin(), child))
                   : 3;
          CachedLink link;
          link.source = source;
          link.relation[type] = 1.;
          if (position < 4)
            link.relation[4 + position] = 1.;
          if (cached && cache.at(source)) {
            link.reuse = *cache.at(source);
            link.present = true;
          }
          links.push_back(link);
        }
        topology = node_topology.emplace(key, std::move(links)).first;
      }
      const auto &cached = topology->second;
      max_degree = std::max(max_degree, cached.size());

      for (const auto &source : cached) {
        Link link;
        link.source = node_indices.at(source.source);
        link.relation = source.relation;
        link.reuse = source.reuse;
        link.present = source.present;
        graph.links.push_back(link);
      }
      graph.link_offsets.push_back(graph.links.size());
    }
    auto remaining = state->remaining();
    std::string work_key;
    work_key.reserve(remaining.size() * 6 + 2);
    work_key.push_back('[');
    for (std::size_t wi = 0; wi < remaining.size(); ++wi) {
      if (wi)
        work_key.push_back(',');
      work_key += std::to_string(remaining[wi]);
    }
    work_key.push_back(']');
    auto w = unique_work.find(work_key);
    if (w == unique_work.end()) {
      std::vector<Id> group;
      group.reserve(remaining.size());
      for (auto nid : remaining) {
        group.push_back(nid);
        if (shard.work_operators.count(nid))
          continue;
        const auto &d = *descriptors.at(nid);
        Json numeric = d.at("kind");
        static const Json missing_stat = nullptr;
        const auto live_it = remaining_sel_by_id.find(nid);
        const auto &live =
            live_it != remaining_sel_by_id.end() ? *live_it->second : missing_stat;
        const auto &source =
            live.is_object() ? live : d.at("remaining_statistics");
        StatOverlay stat;
        stat.overlay = source;
        for (auto value : estimated(stat, "selectivity", true,
                                    d.at("semantic").get<bool>(), true))
          numeric.push_back(value);
        add_input_time(stat, profile, d, nullptr);
        for (auto value :
             estimated(stat, "seconds_per_input", false, true, true))
          numeric.push_back(value);
        // Unselected semantic implementations keep separate measured means
        // at the same model slots used by selected implementation features.
        std::array<double, 64 * 3> model_times{};
        for (const auto &model : d.at("remaining_models")) {
          const auto slot = model.at("slot").get<std::size_t>();
          if (slot >= 64)
            throw std::invalid_argument("Unfinished model timing slot out of range");
          Json variant = {{"model", model.at("model_id")},
                          {"kind", d.at("timing_kind")}};
          StatOverlay timing;
          add_input_time(timing, profile, d, &variant);
          const auto values = estimated(timing, "seconds_per_input", false, true, true);
          std::copy(values.begin(), values.end(), model_times.begin() + 3 * slot);
        }
        for (auto value : model_times)
          numeric.push_back(value);
        shard.work_operators.emplace(
            nid, Json{{"fields", d.at("fields")},
                      {"description", d.at("description")},
                      {"numeric", std::move(numeric)}});
      }
      graph.work = work.size();
      shard.work_keys.push_back(work_key);
      unique_work.emplace(std::move(work_key), graph.work);
      work.push_back(std::move(group));
    } else
      graph.work = w->second;
    for (auto nid : remaining)
      describe(nid);
    graph.size = std::log1p(static_cast<double>(remaining.size()));
    graphs.push_back(std::move(graph));
    if ((gi + 1) % chunk_size == 0 || gi + 1 == end)
      offsets.push_back(rows.size());
  }
  return shard;
}
} // namespace semcache::python_features_detail
