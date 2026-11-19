#include "native_features_internal.hpp"

namespace semcache::python_features_detail {
std::shared_ptr<const HistoryIndex>
NativeFeatureContext::history_for(const ProfileEnvironment &environment) const {
  const auto token = environment.identity_token();
  {
    std::lock_guard<std::mutex> lock(history_mutex);
    if (cached_history && cached_history_token.lock() == token)
      return cached_history;
  }
  const auto hist = environment.history_shared();
  auto built =
      hist ? std::make_shared<const HistoryIndex>(*hist) : initial_history;
  std::lock_guard<std::mutex> lock(history_mutex);
  cached_history_token = token;
  cached_history = built;
  return built;
}
FeatureBatch NativeFeatureContext::pack(const std::vector<StatePtr> &states,
                  const ProfileEnvironment &environment) const {
  Shard merged;
  std::vector<std::pair<std::size_t, std::size_t>> chunk_shapes;
  std::size_t B = states.size(), N = 1, D = 1, U = 0;
  std::vector<float> numeric, hist, relations, reuse, sizes;
  std::vector<std::int64_t> ids, sources, work_indices;
  std::vector<std::uint8_t> present, mask, node_mask, root_mask;
  {
    py::gil_scoped_release release;
    const auto &initial_profile = config.at("profile");
    // Large immutable scan priors stay borrowed; only calibration banks and
    // remaining-operator evidence need a current per-call overlay.
    Json profile = Json::object();
    for (const auto *key :
         {"accuracy", "request_profiles", "remaining_selectivity", "operator_time_calibration", "relational_profiles"})
      if (initial_profile.contains(key))
        profile[key] = initial_profile.at(key);
    std::shared_ptr<const HistoryIndex> history_index;
    ProfileEnvironment::BatchViews state_views;
    if (!states.empty()) {
      auto overlay = environment.profile_overlay();
      for (auto it = overlay.begin(); it != overlay.end(); ++it)
        profile[it.key()] = std::move(it.value());
      history_index = history_for(environment);
      state_views = environment.batch_state_views(states);
    } else
      history_index = initial_history;
    const auto chunks = states.size() / chunk_size +
                        std::size_t(states.size() % chunk_size != 0);
    const auto workers = std::min(worker_limit, chunks);
    if (workers <= 1) {
      merged = build_range(states, state_views, profile,
                           *history_index, 0, states.size());
    } else {
      std::vector<Shard> shards(workers);
      std::vector<std::exception_ptr> errors(workers);
      const auto run = [&](std::size_t worker) {
        try {
          const auto begin = (chunks * worker / workers) * chunk_size;
          const auto end = std::min(
              states.size(), (chunks * (worker + 1) / workers) * chunk_size);
          shards[worker] =
              build_range(states, state_views, profile,
                          *history_index, begin, end);
        } catch (...) {
          errors[worker] = std::current_exception();
        }
      };
      std::vector<std::thread> threads;
      threads.reserve(workers - 1);
      try {
        for (std::size_t worker = 1; worker < workers; ++worker)
          threads.emplace_back(run, worker);
      } catch (...) {
        for (auto &thread : threads)
          thread.join();
        throw;
      }
      run(0);
      for (auto &thread : threads)
        thread.join();
      for (const auto &error : errors)
        if (error)
          std::rethrow_exception(error);
      merged = merge_shards(shards);
    }
    const auto &graphs = merged.graphs;
    auto &rows = merged.rows;
    N = merged.max_nodes;
    D = merged.max_degree;
    U = rows.size();
    for (std::size_t start = 0; start < graphs.size(); start += chunk_size) {
      std::size_t n = 1, d = 1;
      for (std::size_t i = start;
           i < std::min(graphs.size(), start + chunk_size); ++i) {
        n = std::max(n, graphs[i].rows.size());
        const auto &offsets = graphs[i].link_offsets;
        for (std::size_t j = 0; j + 1 < offsets.size(); ++j)
          d = std::max(d, offsets[j + 1] - offsets[j]);
      }
      chunk_shapes.emplace_back(n, d);
    }
    numeric.resize(U * numeric_width);
    hist.resize(U * history_width);
    relations.resize(B * N * D * 8);
    reuse.resize(B * N * D);
    sizes.resize(B);
    ids.resize(B * N);
    sources.resize(B * N * D);
    work_indices.resize(B);
    present.resize(B * N * D);
    mask.resize(B * N * D);
    node_mask.resize(B * N);
    root_mask.resize(B * N);
    for (std::size_t i = 0; i < U; ++i) {
      auto &r = rows[i];
      std::transform(r.numeric.begin(), r.numeric.end(),
                     numeric.begin() + i * numeric_width,
                     [](double v) { return static_cast<float>(v); });
      std::transform(r.history.begin(), r.history.end(),
                     hist.begin() + i * history_width,
                     [](double v) { return static_cast<float>(v); });
    }
    auto fill_graphs = [&](std::size_t b_begin, std::size_t b_end) {
      for (std::size_t i = b_begin; i < b_end; ++i) {
        const auto &g = graphs[i];
        work_indices[i] = g.work;
        sizes[i] = static_cast<float>(g.size);
        for (std::size_t j = 0; j < N; ++j) {
          const auto base = (i * N + j) * D;
          if (j >= g.rows.size()) {
            std::fill(mask.begin() + base, mask.begin() + base + D, 1);
            continue;
          }
          ids[i * N + j] = g.rows[j];
          node_mask[i * N + j] = 1;
          root_mask[i * N + j] = g.roots[j];
          for (std::size_t k = 0; k < g.link_offsets[j + 1] - g.link_offsets[j];
               ++k) {
            const auto &link = g.links[g.link_offsets[j] + k];
            sources[base + k] = link.source;
            reuse[base + k] = static_cast<float>(link.reuse);
            present[base + k] = link.present;
            mask[base + k] = 1;
            std::transform(link.relation.begin(), link.relation.end(),
                           relations.begin() + (base + k) * 8,
                           [](double v) { return static_cast<float>(v); });
          }
        }
      }
    };
    const auto fill_workers = std::min(worker_limit, B / 256);
    if (fill_workers <= 1) {
      fill_graphs(0, B);
    } else {
      std::vector<std::thread> fill_threads;
      fill_threads.reserve(fill_workers - 1);
      for (std::size_t w = 1; w < fill_workers; ++w) {
        const auto b0 = (B * w) / fill_workers;
        const auto b1 = (B * (w + 1)) / fill_workers;
        fill_threads.emplace_back(fill_graphs, b0, b1);
      }
      fill_graphs(0, B / fill_workers);
      for (auto &t : fill_threads)
        t.join();
    }
  }
  FeatureBatch batch;
  batch.cache_features_enabled = cache_features_enabled;
  batch.chunk_offsets = std::move(merged.offsets);
  batch.chunk_size = chunk_size;
  batch.chunk_shapes = std::move(chunk_shapes);
  // Numeric/history arrays already own these values. Only immutable task
  // and field metadata cross Python here, shared across identical templates.
  std::unordered_map<std::string, py::object> templates;
  for (auto &row : merged.rows) {
    auto found = templates.find(row.template_key);
    if (found == templates.end())
      found = templates.emplace(row.template_key, to_python(row.value)).first;
    batch.templates.append(found->second);
  }
  std::unordered_map<Id, py::object> work_rows;
  for (auto &group : merged.work) {
    batch.has_work = batch.has_work || !group.empty();
    py::list rows;
    for (auto nid : group) {
      auto found = work_rows.find(nid);
      if (found == work_rows.end())
        found = work_rows
                    .emplace(nid, to_python(merged.work_operators.at(nid)))
                    .first;
      rows.append(found->second);
    }
    batch.work.append(rows);
  }
  for (auto &chunk : merged.descriptions) {
    py::list values;
    for (auto &description : chunk)
      values.append(to_python(description.value));
    batch.descriptions_by_chunk.append(values);
  }
  const auto b = static_cast<py::ssize_t>(B), n = static_cast<py::ssize_t>(N),
             d = static_cast<py::ssize_t>(D), u = static_cast<py::ssize_t>(U);
  batch.numeric = array(std::move(numeric), {u, numeric_width});
  batch.history = array(std::move(hist), {u, history_width});
  batch.node_indices = array(std::move(ids), {b, n});
  batch.source_indices = array(std::move(sources), {b, n, d});
  batch.relations = array(std::move(relations), {b, n, d, 8});
  batch.reuse = array(std::move(reuse), {b, n, d});
  batch.reuse_present = array(std::move(present), {b, n, d});
  batch.edge_mask = array(std::move(mask), {b, n, d});
  batch.node_mask = array(std::move(node_mask), {b, n});
  batch.root_mask = array(std::move(root_mask), {b, n});
  batch.work_indices = array(std::move(work_indices), {b});
  batch.sizes = array(std::move(sizes), {b, 1});
  {
    py::gil_scoped_release release;
    merged = Shard{};
  }
  return batch;
}
} // namespace semcache::python_features_detail

using namespace semcache::python_features_detail;
void bind_native_features(py::module_ &module) {
  module.attr("SUPPORTS_CACHE_FEATURE_ABLATION") = true;
  py::class_<FeatureBatch>(module, "NativeFeatureBatch")
      .def_readonly("cache_features_enabled", &FeatureBatch::cache_features_enabled)
      .def_readonly("numeric", &FeatureBatch::numeric)
      .def_readonly("history", &FeatureBatch::history)
      .def_readonly("templates", &FeatureBatch::templates)
      .def_readonly("work", &FeatureBatch::work)
      .def_readonly("node_indices", &FeatureBatch::node_indices)
      .def_readonly("source_indices", &FeatureBatch::source_indices)
      .def_readonly("relations", &FeatureBatch::relations)
      .def_readonly("reuse", &FeatureBatch::reuse)
      .def_readonly("reuse_present", &FeatureBatch::reuse_present)
      .def_readonly("edge_mask", &FeatureBatch::edge_mask)
      .def_readonly("node_mask", &FeatureBatch::node_mask)
      .def_readonly("root_mask", &FeatureBatch::root_mask)
      .def_readonly("work_indices", &FeatureBatch::work_indices)
      .def_readonly("sizes", &FeatureBatch::sizes)
      .def_readonly("chunk_offsets", &FeatureBatch::chunk_offsets)
      .def_readonly("descriptions_by_chunk",
                    &FeatureBatch::descriptions_by_chunk)
      .def_readonly("has_work", &FeatureBatch::has_work)
      .def_readonly("chunk_size", &FeatureBatch::chunk_size)
      .def_readonly("chunk_shapes", &FeatureBatch::chunk_shapes);
  py::class_<NativeFeatureContext>(module, "NativeFeatureContext")
      .def(py::init<py::dict>())
      .def("pack", &NativeFeatureContext::pack);
}
py::object
native_feature_batch(py::handle context, const std::vector<StatePtr> &states,
                     const semcache::sampling::SearchEnvironment &environment) {
  const auto *profile = dynamic_cast<const ProfileEnvironment *>(&environment);
  if (!profile)
    return py::none();
  return py::cast(
      context.cast<const NativeFeatureContext &>().pack(states, *profile));
}
