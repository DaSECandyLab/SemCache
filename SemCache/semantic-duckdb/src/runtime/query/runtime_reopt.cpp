#include "runtime/query_runtime.hpp"
#include "optimizer/optimizer_client.hpp"
#include <algorithm>
#include <chrono>
#include <thread>

namespace duckdb {
namespace semantic_extension {

bool QueryRuntime::HasPendingPlanUpdate(int64_t id) const {
	for (size_t i = 0; i < pending_plan_updates.size(); ++i) {
		const auto &updates = pending_plan_updates[i].updates;
		for (size_t j = 0; j < updates.size(); ++j)
			if (updates[j].at("operator_id").get<int64_t>() == id)
				return true;
	}
	return false;
}

void QueryRuntime::BeginChunk(ClientContext &context, int64_t id, uint64_t first, uint64_t rows) {
	if (!adaptive || !adaptive->ChunkMode())
		return;
	Reoptimize(context);
	if (progress)
		progress->ObserveChunk(id, first, rows);
	adaptive->BeginChunk(id, first, rows);
}

void QueryRuntime::ExtendChunk(ClientContext &, int64_t id, uint64_t rows) {
	if (!adaptive || !adaptive->ChunkMode())
		return;
	if (progress)
		progress->ObserveChunk(id, adaptive->ActiveEnd(id), rows);
	adaptive->ExtendChunk(id, rows);
}

void QueryRuntime::EndChunk(ClientContext &context, int64_t id) {
	if (!adaptive || !adaptive->ChunkMode())
		return;
	adaptive->EndChunk(id);
	Reoptimize(context);
}

Json QueryRuntime::RuntimeStatistics() const {
	Json result = Json::object();
	for (const auto &observation : adaptive->Observations()) {
		const auto id = observation.at("operator_id").get<int64_t>();
		if (observation.at("config_generation") != adaptive->Current(id).at("current").at("config_generation"))
			continue;
		auto found = operator_statistics.find(id);
		Json total = found == operator_statistics.end() ? Json::object() : found->second;
		total["config_generation"] = observation.at("config_generation");
		total["completed"] = observation.at("completed");
		total["emitted"] = observation.at("emitted");
		total["chunks"] = total.value("chunks", uint64_t(0));
		const auto active = chunk_usage.find(id);
		const auto usage = active == chunk_usage.end() ? ChunkUsage() : active->second;
		auto add = [&](const char *key, uint64_t count) {
			total[key] = total.value(key, uint64_t(0)) + count;
		};
		add("responses", usage.responses);
		add("known_responses", usage.known);
		add("cached_tokens", usage.cached);
		add("prompt_cache_hits", usage.prompt_cache_hits);
		add("known_prompt_cache_hits", usage.prompt_cache_known_hits);
		add("prompt_cache_tokens", usage.prompt_cache_tokens);
		const auto count = total.at("responses").get<uint64_t>() + total.at("prompt_cache_hits").get<uint64_t>();
		const bool known = total["responses"] == total["known_responses"] &&
		                   total["prompt_cache_hits"] == total["known_prompt_cache_hits"];
		total["mean_matched_tokens"] = count && known ? Json(double(total.at("cached_tokens").get<uint64_t>() +
		                                                            total.at("prompt_cache_tokens").get<uint64_t>()) /
		                                                     count)
		                                              : Json(nullptr);
		total["statistics_basis"] = "actual completed inputs including the active chunk";
		result[std::to_string(id)] = std::move(total);
	}
	return result;
}

void QueryRuntime::Reoptimize(ClientContext &context) {
	if (!reopt_interval && !progress)
		return;
	if (progress && runtime_switches >= progress->Limit() && pending_plan_updates.empty()) {
		return;
	}
	// Only transport runs in the worker. DuckDB objects, admission snapshots and
	// compare-and-swap publication remain owned by the serial execution thread.
	Json response;
	bool ready = false;
	if (!ready && reoptimization.valid() &&
	    reoptimization.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
		try {
			response = reoptimization.get();
		} catch (const std::exception &error) {
			response = {{"error", error.what()}};
		}
		ready = true;
	}
	if (ready) {
		try {
			auto &result = response;
			if (result.contains("error"))
				throw semantic::ValidationError(result["error"].get<std::string>());
			if (result.value("kind", "") != "RuntimeUpdate" || result.value("version", 0) != 1 ||
			    result.value("query_id", "") != query_ir["query_id"].get<std::string>())
				throw semantic::ValidationError("runtime value response identity mismatch");
			if (!result.at("updates").is_array())
				throw semantic::ValidationError("runtime updates must be an array");
			if (!result.at("updates").empty()) {
				if (adaptive->ChunkMode()) {
					std::set<int64_t> updated_operators;
					for (const auto &update : result["updates"])
						updated_operators.insert(update.at("operator_id").get<int64_t>());
					for (auto &existing : pending_plan_updates) {
						Json kept = Json::array();
						for (auto &item : existing.updates) {
							if (!updated_operators.count(item.at("operator_id").get<int64_t>()))
								kept.push_back(std::move(item));
						}
						existing.updates = std::move(kept);
					}
					pending_plan_updates.erase(
					    std::remove_if(pending_plan_updates.begin(), pending_plan_updates.end(),
					                   [](const PendingPlanUpdate &item) { return item.updates.empty(); }),
					    pending_plan_updates.end());
					PendingPlanUpdate pending;
					pending.updates = result["updates"];
					pending.checkpoint = reoptimization_checkpoint;
					pending_plan_updates.push_back(std::move(pending));
				} else
					adaptive->Update(result["updates"]);
			}
			if (result.contains("diagnostics"))
				Log({{"event", "runtime_adaptation_check"},
				     {"query_id", query_ir["query_id"]},
				     {"after_request_seq", sequence},
				     {"diagnostics", result["diagnostics"]}});
		} catch (const std::exception &error) {
			Log({{"event", "runtime_value_rejected"},
			     {"query_id", query_ir["query_id"]},
			     {"after_request_seq", sequence},
			     {"reason", error.what()}});
		}
	}
	for (auto &pending : pending_plan_updates) {
		if (!pending.counted && runtime_switches >= progress->Limit()) {
			pending.updates = Json::array();
			continue;
		}
		pending_update_checkpoint = pending.checkpoint;
		Json waiting = Json::array();
		for (auto &update : pending.updates) {
			try {
				auto oid = update.at("operator_id").get<int64_t>();
				if (!adaptive->AtBoundary(oid)) {
					waiting.push_back(update);
					continue;
				}
				adaptive->Update(Json::array({update}));
				if (!pending.counted) {
					pending.counted = true;
					++runtime_switches;
					Log({{"event", "runtime_plan_update_applied"},
					     {"query_id", query_ir["query_id"]},
					     {"after_request_seq", sequence},
					     {"progress_checkpoint", pending_update_checkpoint},
					     {"switches", runtime_switches}});
					if (runtime_switches == progress->Limit())
						Log({{"event", "runtime_switch_limit_reached"},
						     {"query_id", query_ir["query_id"]},
						     {"after_request_seq", sequence},
						     {"switches", runtime_switches},
						     {"limit", progress->Limit()}});
				}
				// New implementations start fresh cumulative usage statistics,
				// even before their first input chunk has completed.
				operator_statistics.erase(oid);
			} catch (const std::exception &error) {
				Log({{"event", "runtime_value_rejected"},
				     {"query_id", query_ir["query_id"]},
				     {"after_request_seq", sequence},
				     {"reason", error.what()}});
			}
		}
		pending.updates = std::move(waiting);
	}
	pending_plan_updates.erase(std::remove_if(pending_plan_updates.begin(), pending_plan_updates.end(),
	                                          [](const PendingPlanUpdate &pending) { return pending.updates.empty(); }),
	                           pending_plan_updates.end());
	if (progress && runtime_switches >= progress->Limit())
		return;
	Json checkpoint;
	if (progress) {
		const auto due = progress->TakeDue();
		if (!due)
			return;
		checkpoint = progress->Snapshot(due);
		if (reoptimization.valid() || progress->AllDone()) {
			Log({{"event", "runtime_progress_check_skipped"},
			     {"query_id", query_ir["query_id"]},
			     {"after_request_seq", sequence},
			     {"progress_checkpoint", checkpoint},
			     {"reason", progress->AllDone() ? "query_work_completed" : "previous_opportunity_in_flight"}});
			return;
		}
	} else {
		if (reoptimization.valid())
			return;
		uint64_t actual_submissions;
		{
			std::lock_guard<std::recursive_mutex> guard(submission_mutex);
			actual_submissions = llm_submissions;
		}
		if (actual_submissions - last_reopt_sequence < reopt_interval)
			return;
		last_reopt_sequence = actual_submissions;
	}
	Json request {{"version", 1},
	              {"kind", "RuntimeRequest"},
	              {"query_id", query_ir["query_id"]},
	              {"query_ir", query_ir},
	              {"plan_spec", plan_spec},
	              {"state", adaptive->Report()},
	              {"observations", adaptive->Observations()},
	              {"events", pending_events},
	              {"events_complete", pending_complete},
	              {"timeout_ms", reopt_timeout_ms},
	              {"after_request_seq", sequence}};
	if (adaptive->ChunkMode()) {
		request["chunk_policy"] = true;
		request["completed_chunks"] = completed_chunks;
		request["progress_checkpoint"] = checkpoint;
		request["max_switches"] = progress->Limit();
		request["applied_switches"] = runtime_switches;
		request["operator_statistics"] = RuntimeStatistics();
		request["telemetry_path"] = telemetry_path;
		std::lock_guard<std::mutex> guard(log_mutex);
		file.flush();
		request["telemetry_bytes"] = static_cast<int64_t>(file.tellp());
	}
	pending_events = Json::array();
	pending_complete = true;
	auto path = GetSetting(context, "semantic_optimizer_socket");
	auto timeout = reopt_timeout_ms;
	// Promise-backed futures do not join on destruction. The detached worker
	// owns only immutable values and its deadline-bounded transport, never this
	// query or DuckDB objects; query completion does not wait for the optimizer.
	auto promise = std::make_shared<std::promise<Json>>();
	reoptimization_checkpoint = checkpoint;
	reoptimization = promise->get_future();
	std::thread([request, path, timeout, promise]() {
		try {
			semantic::UnixSocketOptimizerClient client(path, timeout);
			promise->set_value(client.Optimize(request));
		} catch (...) {
			promise->set_exception(std::current_exception());
		}
	}).detach();
	Log({{"event", "runtime_value_requested"},
	     {"query_id", query_ir["query_id"]},
	     {"after_request_seq", sequence},
	     {"progress_checkpoint", checkpoint},
	     {"observations", request["observations"]}});
}

}
}
