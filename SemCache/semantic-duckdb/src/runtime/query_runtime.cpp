#include "runtime/query_runtime.hpp"
#include "optimizer/optimizer_client.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <iterator>
#include <thread>

namespace duckdb {
namespace semantic_extension {

SemanticResultCache::Bucket &QueryRuntime::ResultBucket(semantic::LLMRequest &request, uint64_t generation) {
	// Request preparation owns transparent response-cache lookup. Repeat it for
	// every input, even when this generation already has a result-cache bucket.
	client.ConfigureRequest(request);
	next_request_cached = bool(request.cached_response);
	if (client.ResultCacheVerificationEnabled()) {
		// Provisional transport results must still receive their independent
		// second-use verification before they can be shared as final answers.
		return results.Disabled();
	}
	// A configuration generation fixes model, system/output contract and decode settings.
	auto key = std::make_pair(request.operator_id, generation);
	auto found = result_bindings.find(key);
	if (found == result_bindings.end()) {
		found = result_bindings.emplace(key, &results.Bind(request)).first;
	}
	if (!found->second->Enabled())
		reuse_results = false;
	return *found->second;
}

void QueryRuntime::CheckRequestAllowance() {
	std::lock_guard<std::recursive_mutex> guard(submission_mutex);
	if (llm_submissions >= llm_request_limit) {
		budget_exhausted = true;
		throw SamplingBudgetExhausted();
	}
}

void QueryRuntime::ReserveRequestAllowance() {
	std::lock_guard<std::recursive_mutex> guard(submission_mutex);
	if (!next_request_cached && (llm_submissions >= llm_request_limit ||
	                             llm_reserved - cached_reserved >= llm_request_limit - llm_submissions)) {
		budget_exhausted = true;
		throw SamplingBudgetExhausted();
	}
	++llm_reserved;
	if (next_request_cached)
		++cached_reserved;
}

QueryRuntime::QueryRuntime(const SemanticBindData &binding, const string &path, const string &runtime_options,
                           size_t parallelism, std::shared_ptr<semantic::HttpLLMClient::ConnectionPool> pool,
                           bool unrestricted, size_t max_inflight_per_model, bool enforce_model_operator_order)
    : file(path.empty() ? "" : path, std::ios::app),
      client(
          semantic::ModelRegistry::Parse(binding.models), {},
          [this](const semantic::LLMRequest &request, bool retransmit) {
	          std::lock_guard<std::recursive_mutex> guard(submission_mutex);
	          CheckRequestAllowance();
	          // Retries may spend only unreserved allowance. Requests already
	          // admitted by the CPU thread retain their first-send reservation.
	          if (retransmit && llm_reserved - cached_reserved >= llm_request_limit - llm_submissions) {
		          budget_exhausted = true;
		          throw SamplingBudgetExhausted();
	          }
	          // Retransmissions are billed under the original request_seq; audit records every consumed or discarded call.
	          if (!retransmit) {
		          if (llm_reserved)
			          --llm_reserved;
		          ++request_counts[request.operator_id];
		          if (file.is_open())
			          unconsumed_requests.emplace(request.request_seq,
			                                      UnconsumedRequest {request.operator_id, request.model_id});
	          }
	          if (file.is_open()) {
		          Json event {{"event", retransmit ? "sampling_llm_retransmit" : "sampling_llm_submit"},
		                      {"query_id", request.query_id},
		                      {"operator_id", request.operator_id},
		                      {"request_seq", request.request_seq},
		                      {"model_id", request.model_id}};
		          if (request.cache_correction_retry)
			          event["retry_reason"] = "cache_correction";
		          Log(event);
	          }
	          ++llm_submissions;
          },
          std::move(pool),
          [this](const semantic::LLMRequest &request) {
	          std::lock_guard<std::recursive_mutex> guard(submission_mutex);
	          if (llm_reserved)
		          --llm_reserved;
	          if (request.cached_response && cached_reserved)
		          --cached_reserved;
	          if (file.is_open())
		          cached_request_sequences.insert(request.request_seq);
          }),
      // Normal execution optionally prioritizes earlier operators and permits
      // the configured in-flight requests per physical model. Beam sampling passes
      // unrestricted=true; its request ledger and immutable inputs still bound
      // work, but transport no longer waits for a same-model predecessor.
      scheduler(binding.execution_context.query_id, client, path.empty() ? nullptr : &file, parallelism, &log_mutex,
                unrestricted, max_inflight_per_model, enforce_model_operator_order) {
	sampling = binding.sampling;
	telemetry_path = path;
	llm_request_limit = binding.llm_request_limit;
	this->max_inflight_per_model = max_inflight_per_model;
	if (!path.empty() && !file)
		throw InvalidInputException("Cannot open semantic_telemetry path");
	if (!runtime_options.empty()) {
		if (binding.decision_json.empty())
			throw InvalidInputException("Runtime configuration requires a validated native SQL/PlanSpec");
		const auto id = binding.execution_context.query_id;
		adaptive = make_uniq<semantic::RuntimePlanState>(
		    binding.execution_context.operators, client.KnownModelIds(), runtime_options,
		    [this, id](const Json &event) {
			    auto record = event;
			    record["query_id"] = id;
			    record["after_request_seq"] = sequence;
			    Log(record);
		    },
		    [](int64_t, double latency_ms) { return latency_ms > 0; });
	}
	plan_spec = semantic::llm_detail::ParseJSON(binding.decision_json);
	query_ir = semantic::llm_detail::ParseJSON(binding.query_ir_json);
	std::vector<int64_t> ordered_operators;
	if (!plan_spec.is_null()) {
		for (const auto &action : plan_spec["actions"]) {
			operator_ids.insert(action["operator_id"].get<int64_t>());
			ordered_operators.push_back(action["operator_id"].get<int64_t>());
		}
	} else {
		for (const auto &op : binding.execution_context.operators)
			ordered_operators.push_back(op.semantic.operator_id);
	}
	scheduler.SetOperatorOrder(ordered_operators);
	if (!runtime_options.empty()) {
		auto options = semantic::llm_detail::ParseJSON(runtime_options);
		if (options.contains("value_policy")) {
			if (!file.is_open())
				throw InvalidInputException("Runtime value policy requires semantic_telemetry");
			reopt_timeout_ms = options["value_policy"]["timeout_ms"].get<int>();
			if (adaptive->ChunkMode()) {
				const auto &policy = options["value_policy"];
				progress =
				    make_uniq<semantic::RuntimeProgress>(query_ir, policy.value("estimated_input_rows", Json::object()),
				                                         policy.at("max_switches").get<uint64_t>());
			} else {
				reopt_interval = options["value_policy"]["interval_requests"].get<uint64_t>();
			}
		}
	}
	if (file.is_open())
		Log({{"event", "planning_decision"}, {"query_ir", query_ir}, {"plan_spec", plan_spec}});
}

QueryRuntime::~QueryRuntime() {
	scheduler.Shutdown();
}

void QueryRuntime::RecordOperatorInterval(int64_t operator_id, int64_t lo, int64_t hi) {
	if (!file.is_open() || operator_id < 0 || lo < 0 || hi <= lo)
		return;
	std::lock_guard<std::mutex> guard(timing_mutex);
	auto &bank = operator_intervals[operator_id];
	// Allocate before modifying existing intervals, so allocation failure cannot
	// corrupt an already recorded union. Every subsequent operation is in-place.
	auto inserted = bank.intervals.emplace(lo, hi);
	auto current = inserted.first;
	if (inserted.second) {
		bank.union_ns += uint64_t(hi - lo);
	} else if (hi > current->second) {
		bank.union_ns += uint64_t(hi - current->second);
		current->second = hi;
	}
	if (current != bank.intervals.begin()) {
		auto previous = std::prev(current);
		if (previous->second >= current->first) {
			bank.union_ns -= uint64_t(previous->second - previous->first);
			bank.union_ns -= uint64_t(current->second - current->first);
			previous->second = std::max(previous->second, current->second);
			bank.union_ns += uint64_t(previous->second - previous->first);
			bank.intervals.erase(current);
			current = previous;
		}
	}
	auto next = std::next(current);
	while (next != bank.intervals.end() && next->first <= current->second) {
		bank.union_ns -= uint64_t(current->second - current->first);
		bank.union_ns -= uint64_t(next->second - next->first);
		current->second = std::max(current->second, next->second);
		bank.union_ns += uint64_t(current->second - current->first);
		next = bank.intervals.erase(next);
	}
}

double QueryRuntime::OperatorProcessingSeconds(int64_t operator_id) {
	std::lock_guard<std::mutex> guard(timing_mutex);
	auto found = operator_intervals.find(operator_id);
	return found == operator_intervals.end() ? 0.0 : double(found->second.union_ns) / 1000000000.0;
}

void QueryRuntime::Log(const Json &event) {
	std::lock_guard<std::recursive_mutex> submission_guard(submission_mutex);
	std::lock_guard<std::mutex> guard(log_mutex);
	if (!file.is_open())
		return;
	auto record = event;
	const auto kind = record.value("event", string());
	if (kind == "semantic_row" && record.contains("request_seq") &&
	    cached_request_sequences.count(record["request_seq"].get<uint64_t>())) {
		const auto source_sequence = record["request_seq"].get<uint64_t>();
		const auto operator_id = record["operator_id"].get<int64_t>();
		// A transport hit counts as a reuse only when the operator consumes it.
		// Speculative hits abandoned by LIMIT have no physical request to discard.
		++response_cache_hits[operator_id];
		record["event"] = "semantic_result_cache_hit";
		record["after_request_seq"] = sequence;
		// This identifies an admitted cache answer, never a physical HTTP call.
		// Later original rows can refer to it without inventing request telemetry.
		record["cache_source_seq"] = source_sequence;
		record["prompt_cache_token_scope"] = "stored_request_prompt";
		for (const auto *field : {"request_seq", "submit_ns", "response_ns", "server_start_ns", "latency_ms",
		                          "input_tokens", "output_tokens", "cached_input_tokens", "server_metadata"})
			record.erase(field);
		// Operators retain their existing completion accounting. Remove only
		// the synthetic zero-duration cache sample before it reaches summaries.
		auto &total = latency_totals[operator_id];
		if (total.first)
			--total.first;
		auto &samples = latencies[operator_id];
		auto zero = std::find(samples.rbegin(), samples.rend(), 0.0);
		if (zero != samples.rend())
			samples.erase(std::next(zero).base());
		cached_request_sequences.erase(source_sequence);
	}
	// Inspect the final event kind: transparent cache hits have no physical HTTP
	// interval and must not add synthetic zero-duration requests to this bank.
	if (record["event"] == "semantic_row" && record.contains("submit_ns") && record.contains("response_ns") &&
	    record["submit_ns"].is_number_integer() && record["response_ns"].is_number_integer()) {
		RecordOperatorInterval(record["operator_id"].get<int64_t>(), record["submit_ns"].get<int64_t>(),
		                       record["response_ns"].get<int64_t>());
	}
	if (kind == "semantic_row" && record.contains("request_seq"))
		unconsumed_requests.erase(record["request_seq"].get<uint64_t>());
	if ((record["event"] == "semantic_row" || record["event"] == "semantic_result_cache_hit" ||
	     record["event"] == "semantic_result_reuse") &&
	    record.value("parse_status", "") == "validated") {
		auto &published = published_counts[record.at("operator_id").get<int64_t>()];
		++published.first;
		published.second += record.at("emitted").get<bool>();
		if (progress)
			progress->Observe(record.at("operator_id").get<int64_t>());
	}
	if (progress && record["event"] == "runtime_operator_done")
		progress->Finish(record.at("operator_id").get<int64_t>());
	if (progress && record["event"] == "runtime_update")
		record["progress_checkpoint"] = pending_update_checkpoint;

	record["observed_unix_seconds"] =
	    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
	if (adaptive && adaptive->ChunkMode()) {
		if (record["event"] == "semantic_row") {
			auto &usage = chunk_usage[record.at("operator_id").get<int64_t>()];
			++usage.responses;
			const auto tokens = record.value("input_tokens", Json());
			const auto cached = record.value("cached_input_tokens", Json());
			if (tokens.is_number_integer() && cached.is_number_integer() && tokens.get<int64_t>() >= 0 &&
			    cached.get<int64_t>() >= 0 && cached.get<uint64_t>() <= tokens.get<uint64_t>()) {
				++usage.known;
				usage.cached += cached.get<uint64_t>();
			}
			if (record.contains("input_tokens") && record["input_tokens"].is_number_integer() &&
			    record["input_tokens"].get<int64_t>() >= 0)
				usage.tokens += record["input_tokens"].get<uint64_t>();
		} else if (record["event"] == "semantic_result_cache_hit") {
			auto &usage = chunk_usage[record.at("operator_id").get<int64_t>()];
			++usage.prompt_cache_hits;
			const auto tokens = record.value("prompt_cache_input_tokens", Json());
			if (tokens.is_number_integer() && tokens.get<int64_t>() >= 0) {
				++usage.prompt_cache_known_hits;
				usage.prompt_cache_tokens += tokens.get<uint64_t>();
			}
		} else if (record["event"] == "batch_completed") {
			auto oid = record.at("operator_id").get<int64_t>();
			auto usage = chunk_usage[oid];
			record["event"] = "runtime_chunk_completed";
			record["responses"] = usage.responses;
			record["usage_known_responses"] = usage.known;
			record["prompt_cache_hits"] = usage.prompt_cache_hits;
			record["prompt_cache_known_hits"] = usage.prompt_cache_known_hits;
			record["prompt_cache_matched_tokens"] = usage.prompt_cache_tokens;
			const auto matched_count = usage.responses + usage.prompt_cache_hits;
			const bool matched_known =
			    usage.responses == usage.known && usage.prompt_cache_hits == usage.prompt_cache_known_hits;
			record["matched_requests"] = matched_count;
			record["mean_api_cached_tokens"] = usage.responses && usage.known == usage.responses
			                                       ? Json(double(usage.cached) / usage.responses)
			                                       : Json(nullptr);
			record["mean_prompt_cache_tokens"] =
			    usage.prompt_cache_hits && usage.prompt_cache_known_hits == usage.prompt_cache_hits
			        ? Json(double(usage.prompt_cache_tokens) / usage.prompt_cache_hits)
			        : Json(nullptr);
			record["mean_matched_tokens"] = matched_count && matched_known
			                                    ? Json(double(usage.cached + usage.prompt_cache_tokens) / matched_count)
			                                    : Json(nullptr);
			record["cached_tokens"] =
			    usage.responses && usage.known == usage.responses ? Json(usage.cached) : Json(nullptr);
			record["mean_cached_tokens"] = usage.responses && usage.known == usage.responses
			                                   ? Json(double(usage.cached) / usage.responses)
			                                   : Json(nullptr);
			record["mean_prompt_tokens"] =
			    usage.responses ? Json(double(usage.tokens) / usage.responses) : Json(nullptr);
			record["chunk_index"] = ++completed_chunks;
			auto &total = operator_statistics[oid];
			if (!total.is_object() || total.at("config_generation") != record.at("config_generation"))
				total = {{"config_generation", record.at("config_generation")},
				         {"completed", 0},
				         {"emitted", 0},
				         {"chunks", 0},
				         {"responses", 0},
				         {"known_responses", 0},
				         {"cached_tokens", 0},
				         {"prompt_cache_hits", 0},
				         {"known_prompt_cache_hits", 0},
				         {"prompt_cache_tokens", 0}};
			auto add = [&](const char *key, uint64_t value) {
				total[key] = total.at(key).get<uint64_t>() + value;
			};
			add("completed", record.at("completed").get<uint64_t>());
			add("emitted", record.at("emitted").get<uint64_t>());
			add("chunks", 1);
			add("responses", usage.responses);
			add("known_responses", usage.known);
			add("cached_tokens", usage.cached);
			add("prompt_cache_hits", usage.prompt_cache_hits);
			add("known_prompt_cache_hits", usage.prompt_cache_known_hits);
			add("prompt_cache_tokens", usage.prompt_cache_tokens);
			total["last_chunk_rows"] = record.at("completed");
			total["last_chunk_index"] = completed_chunks;
			const auto count = total.at("responses").get<uint64_t>() + total.at("prompt_cache_hits").get<uint64_t>();
			const bool known = total.at("responses") == total.at("known_responses") &&
			                   total.at("prompt_cache_hits") == total.at("known_prompt_cache_hits");
			total["mean_matched_tokens"] = count && known
			                                   ? Json(double(total.at("cached_tokens").get<uint64_t>() +
			                                                 total.at("prompt_cache_tokens").get<uint64_t>()) /
			                                          count)
			                                   : Json(nullptr);
			chunk_usage.erase(oid);
		}
	}
	if (reopt_interval && (!adaptive || !adaptive->ChunkMode()) && record["event"] == "semantic_row") {
		if (pending_events.size() < 256)
			pending_events.push_back(record);
		else
			pending_complete = false;
	}
	file << record.dump() << '\n';
	file.flush();
	if (!file)
		throw InvalidInputException("Semantic telemetry write failed");
}

void QueryRuntime::BeginChunk(ClientContext &context, int64_t id, uint64_t first, uint64_t rows) {
	if (!adaptive || !adaptive->ChunkMode())
		return;
	Reoptimize(context);
	adaptive->BeginChunk(id, first, rows);
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
		if (reoptimization.valid() || progress->Fraction() >= 1) {
			Log({{"event", "runtime_progress_check_skipped"},
			     {"query_id", query_ir["query_id"]},
			     {"after_request_seq", sequence},
			     {"progress_checkpoint", checkpoint},
			     {"reason", progress->Fraction() >= 1 ? "query_work_completed" : "previous_opportunity_in_flight"}});
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

void QueryState::QueryBegin(ClientContext &) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	runtime.reset();
}

void QueryState::QueryEnd(ClientContext &, optional_ptr<ErrorData> error) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	if (runtime)
		runtime->scheduler.Shutdown();
	if (runtime && runtime->file.is_open()) {
		// Shutdown cancels queued work and joins every admitted transport. The
		// remaining identities were paid, but no operator consumed their result.
		// Audit that work without parsing it or inventing downstream rows.
		for (const auto &entry : runtime->unconsumed_requests) {
			try {
				runtime->Log(
				    {{"event", "semantic_request_discarded"},
				     {"query_id", query_id},
				     {"operator_id", entry.second.operator_id},
				     {"request_seq", entry.first},
				     {"model_id", entry.second.model_id},
				     {"reason", error ? "query_failed_before_consumption" : "query_terminated_before_consumption"}});
			} catch (...) {
				if (!error)
					throw;
			}
			++runtime->discarded_requests[entry.second.operator_id];
		}
		runtime->unconsumed_requests.clear();
		runtime->cached_request_sequences.clear();
		// Include operators whose paid work was entirely abandoned before the
		// first consumed row; their processed cardinalities remain zero.
		for (const auto &entry : runtime->request_counts)
			runtime->counts.emplace(entry.first, std::make_pair(uint64_t(0), uint64_t(0)));
	}
	// LIMIT and optimized-away/empty branches can bypass operator finalization.
	// On successful termination, close their remaining batches without inventing
	// calls. Failed queries retain their partial PENDING/RUNNING observations.
	if (runtime && runtime->adaptive && !error) {
		for (const auto &action : runtime->plan_spec["actions"]) {
			auto id = action["operator_id"].get<int64_t>();
			if (runtime->finished.insert(id).second)
				runtime->adaptive->Finish(id);
		}
	}
	// Emit one terminal report even on failure so partial model work is visible
	// and cannot be mistaken for a successfully completed query.
	if (runtime && runtime->file.is_open()) {
		// Report evaluated semantic inputs, even if a partially filled output
		// chunk was abandoned after an empty join branch or LIMIT. Keep the
		// execution counters unchanged until execution has actually ended.
		for (auto &entry : runtime->counts)
			entry.second = runtime->published_counts[entry.first];
		for (const auto &entry : runtime->published_counts)
			runtime->counts[entry.first] = entry.second;
		if (runtime->sampling) {
			try {
				uint64_t completed = 0;
				for (const auto &entry : runtime->counts)
					completed += entry.second.first;
				runtime->Log({{"event", "sampling_stage_progress"},
				              {"query_id", query_id},
				              {"completed_inputs", completed},
				              {"requests", runtime->llm_submissions},
				              {"budget_exhausted", runtime->budget_exhausted.load()}});
			} catch (...) {
			}
		}
		for (auto &entry : runtime->counts) {
			// Cleanup telemetry must never hide the query's original failure.
			try {
				auto &samples = runtime->latencies[entry.first];
				auto total = runtime->latency_totals[entry.first];
				std::sort(samples.begin(), samples.end());
				auto percentile = [&](double fraction) -> Json {
					if (samples.empty() || total.first > samples.size())
						return nullptr;
					return samples[size_t(std::ceil(fraction * samples.size())) - 1];
				};
				runtime->Log(
				    {{"llm_calls", runtime->request_counts[entry.first]},
				     {"discarded_requests", runtime->discarded_requests[entry.first]},
				     {"result_reuses", runtime->result_reuses[entry.first] + runtime->response_cache_hits[entry.first]},
				     {"selectivity",
				      entry.second.first ? Json(double(entry.second.second) / entry.second.first) : Json(nullptr)},
				     {"latency_avg_ms", total.first ? Json(total.second / total.first) : Json(nullptr)},
				     {"latency_percentiles_complete", total.first == samples.size()},
				     {"latency_p50_ms", percentile(0.50)},
				     {"latency_p95_ms", percentile(0.95)},
				     {"event", "operator_summary"},
				     {"query_id", query_id},
				     {"operator_id", entry.first},
				     {"input_rows", entry.second.first},
				     {"output_rows", entry.second.second},
				     {"processing_seconds", runtime->OperatorProcessingSeconds(entry.first)},
				     {"processing_time_basis", "union_of_operator_active_and_http_intervals"},
				     {"query_failed", bool(error)}});
			} catch (...) {
			}
		}
		if (!runtime->plan_spec.is_null()) {
			try {
				Json operators = Json::array();
				for (auto &action : runtime->plan_spec["actions"]) {
					auto id = action["operator_id"].get<int64_t>();
					auto count = runtime->counts[id];
					operators.push_back(
					    {{"operator_id", id},
					     {"input_rows_or_pairs", count.first},
					     {"output_rows", count.second},
					     {"processing_seconds", runtime->OperatorProcessingSeconds(id)},
					     {"processing_time_basis", "union_of_operator_active_and_http_intervals"},
					     {"llm_calls", runtime->request_counts[id]},
					     {"discarded_requests", runtime->discarded_requests[id]},
					     {"result_reuses", runtime->result_reuses[id] + runtime->response_cache_hits[id]},
					     {"started", runtime->started.count(id) != 0},
					     {"request_latency_total_ms", runtime->latency_totals[id].second},
					     {"selectivity", count.first ? Json(double(count.second) / count.first) : Json(nullptr)}});
				}
				runtime->Log(
				    {{"event", "execution_report"},
				     {"version", 1},
				     {"execution_mode", "data_chunk_pipeline"},
				     {"semantic_join_mode",
				      runtime->query_ir.value("semantic_join_mode", "stream_first_left_then_replay")},
				     {"semantic_response_order",
				      runtime->sampling || (runtime->adaptive && !runtime->adaptive->ChunkMode()) ? "input_order"
				                                                                                  : "ready_first"},
				     {"query_id", query_id},
				     {"status", error ? "error" : "success"},
				     {"plan_spec", runtime->plan_spec},
				     {"actual_start_order", runtime->start_order},
				     {"operators", operators},
				     {"runtime_state", runtime->adaptive ? runtime->adaptive->Report() : Json(nullptr)},
				     {"runtime_observations", runtime->adaptive ? runtime->adaptive->Observations() : Json(nullptr)},
				     {"execution_wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
				                                                                     runtime->execution_start)
				                               .count()},
				     {"accuracy", nullptr},
				     {"cache_metrics", nullptr}});
			} catch (...) {
				if (!error)
					throw;
			}
		}
	}
	runtime.reset();
}

QueryRuntime &QueryState::Ensure(ClientContext &context, const SemanticBindData &binding) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	// The first semantic operator creates the shared runtime; later operators must
	// agree on the exact query and optimizer decision.
	ValidateExecutionSettings(context);
	if (!runtime) {
		models = binding.models;
		query_id = binding.execution_context.query_id;
		runtime = make_uniq<QueryRuntime>(
		    binding, GetSetting(context, "semantic_telemetry"), GetSetting(context, "semantic_runtime"),
		    std::stoull(GetSetting(context, "semantic_parallelism")), transport_pool,
		    GetSetting(context, "semantic_unrestricted") == "true",
		    std::stoull(GetSetting(context, "semantic_max_concurrent_requests_per_model")),
		    GetSetting(context, "semantic_enforce_model_operator_order") == "true");
		transport_pool = runtime->client.Connections();
		runtime->client.ConfigureResultCache(GetSetting(context, "semantic_response_cache"));
		// Reuse typed results and pending requests before scheduler admission.
		// Both caches are bounded; verification keeps its independent calls.
		runtime->reuse_results = GetSetting(context, "semantic_request_reuse") == "true" &&
		                         !runtime->client.ResultCacheVerificationEnabled();
	} else if (models != binding.models || query_id != binding.execution_context.query_id) {
		throw InvalidInputException("Inconsistent semantic configuration within query");
	}
	return *runtime;
}

QueryRuntime &QueryState::Get(ClientContext &context, const SemanticBindData &binding) {
	Ensure(context, binding);
	std::lock_guard<std::recursive_mutex> guard(runtime->mutex);
	if (!runtime->plan_spec.is_null()) {
		if (!runtime->operator_ids.count(binding.node.semantic.operator_id) ||
		    runtime->finished.count(binding.node.semantic.operator_id))
			throw InvalidInputException("Semantic pipeline used an unknown or finished PlanSpec operator");
	}
	// Start means the first Execute invocation, not physical-state allocation.
	// Operators whose input is empty may never Execute and have no start event.
	if (runtime->started.insert(binding.node.semantic.operator_id).second) {
		runtime->start_order.push_back(binding.node.semantic.operator_id);
		if (runtime->file.is_open())
			runtime->Log({{"event", "operator_start"},
			              {"query_id", query_id},
			              {"operator_id", binding.node.semantic.operator_id},
			              {"operator_kind", semantic::OperatorName(binding.node.kind)},
			              {"model_id", runtime->adaptive ? runtime->adaptive->Current(
			                                                   binding.node.semantic.operator_id)["current"]["model_id"]
			                                             : Json(binding.node.semantic.model_id)},
			              {"next_request_seq", runtime->sequence + 1}});
	}
	return *runtime;
}

void QueryState::Finish(ClientContext &context, const SemanticBindData &binding) {
	auto &query = Ensure(context, binding);
	std::lock_guard<std::recursive_mutex> guard(query.mutex);
	if (query.finished.insert(binding.node.semantic.operator_id).second && query.adaptive)
		query.adaptive->Finish(binding.node.semantic.operator_id);
}

namespace {
Json InputJSON(const Value &value) {
	if (value.IsNull())
		return nullptr;
	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return value.GetValue<bool>();
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
		return value.GetValue<int64_t>();
	case LogicalTypeId::DOUBLE: {
		auto number = value.GetValue<double>();
		if (!std::isfinite(number))
			throw InvalidInputException("Non-finite semantic input number");
		return number;
	}
	case LogicalTypeId::VARCHAR:
		return value.GetValue<string>();
	case LogicalTypeId::LIST: {
		Json result = Json::array();
		for (const auto &child : ListValue::GetChildren(value))
			result.push_back(InputJSON(child));
		return result;
	}
	case LogicalTypeId::STRUCT: {
		Json result = Json::object();
		const auto &fields = StructType::GetChildTypes(value.type());
		const auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < fields.size(); ++i)
			result[fields[i].first] = InputJSON(children[i]);
		return result;
	}
	default:
		throw InvalidInputException("Unsupported semantic input type");
	}
}
}

string SampleRowJSON(const Value &value) {
	std::function<Json(const Value &)> convert = [&](const Value &item) -> Json {
		if (item.IsNull())
			return nullptr;
		if (item.type().id() == LogicalTypeId::STRUCT) {
			Json result = Json::object();
			auto &fields = StructType::GetChildTypes(item.type());
			auto &children = StructValue::GetChildren(item);
			for (idx_t i = 0; i < fields.size(); ++i)
				result[fields[i].first] = convert(children[i]);
			return result;
		}
		if (item.type().id() == LogicalTypeId::LIST) {
			Json result = Json::array();
			for (auto &child : ListValue::GetChildren(item))
				result.push_back(convert(child));
			return result;
		}
		if (item.type().id() == LogicalTypeId::FLOAT)
			return item.GetValue<double>();
		if (item.type().id() == LogicalTypeId::BOOLEAN || item.type().id() == LogicalTypeId::TINYINT ||
		    item.type().id() == LogicalTypeId::SMALLINT || item.type().id() == LogicalTypeId::INTEGER ||
		    item.type().id() == LogicalTypeId::BIGINT || item.type().id() == LogicalTypeId::DOUBLE ||
		    item.type().id() == LogicalTypeId::VARCHAR)
			return InputJSON(item);
		// SQL-only types (dates, decimals, huge integers) round-trip through their
		// typed temporary columns. This does not widen semantic input contracts.
		return item.ToString();
	};
	return convert(value).dump();
}

semantic::Row ReadRow(const Value &value) {
	semantic::Row result;
	auto &fields = StructType::GetChildTypes(value.type());
	const auto *children = value.IsNull() ? nullptr : &StructValue::GetChildren(value);
	for (idx_t i = 0; i < fields.size(); i++) {
		semantic::RowValue converted;
		if (children && !(*children)[i].IsNull()) {
			const auto &item = (*children)[i];
			switch (item.type().id()) {
			case LogicalTypeId::BOOLEAN:
				converted = semantic::RowValue(item.GetValue<bool>());
				break;
			case LogicalTypeId::INTEGER:
			case LogicalTypeId::BIGINT:
				converted = semantic::RowValue(item.GetValue<int64_t>());
				break;
			case LogicalTypeId::DOUBLE:
				converted = semantic::RowValue(item.GetValue<double>());
				break;
			case LogicalTypeId::VARCHAR:
				converted = semantic::RowValue(item.GetValue<string>());
				break;
			case LogicalTypeId::LIST:
			case LogicalTypeId::STRUCT:
				converted = semantic::RowValue::FromJSON(InputJSON(item).dump());
				break;
			default:
				throw InvalidInputException("Unsupported semantic input type");
			}
		}
		result.emplace(fields[i].first, std::move(converted));
	}
	return result;
}

namespace {

Value ParseField(const Json &value, const semantic::Field &field) {
	auto kind = field.type;
	auto type = ToLogicalType(field);
	if (value.is_null())
		return Value(type);
	if (kind == semantic::FieldType::VARCHAR && value.is_string())
		return Value(value.get<string>());
	if (kind == semantic::FieldType::BOOLEAN && value.is_boolean())
		return Value::BOOLEAN(value.get<bool>());
	if ((kind == semantic::FieldType::INTEGER || kind == semantic::FieldType::BIGINT) && value.is_number_integer()) {
		if (value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))
			throw InvalidInputException("Semantic integer overflow");
		auto number = value.get<int64_t>();
		if (kind == semantic::FieldType::BIGINT)
			return Value::BIGINT(number);
		if (number >= std::numeric_limits<int32_t>::min() && number <= std::numeric_limits<int32_t>::max())
			return Value::INTEGER(int32_t(number));
	}
	if (kind == semantic::FieldType::DOUBLE && value.is_number()) {
		auto number = value.get<double>();
		if (std::isfinite(number))
			return Value::DOUBLE(number);
	}
	auto rendered = value.dump();
	if (rendered.size() > 512)
		rendered.resize(512);
	throw InvalidInputException("Semantic output does not match declared %s (JSON type=%s, value=%s)", type.ToString(),
	                            value.type_name(), rendered);
}

}

Value ParseOutput(const string &raw, const SemanticBindData &binding) {
	auto json = semantic::llm_detail::ParseJSON(raw);
	if (binding.node.kind != semantic::NodeKind::SEMANTIC_PROJECTION) {
		if (!json.is_boolean())
			throw InvalidInputException("Semantic predicate requires JSON true or false");
		return Value::BOOLEAN(json.get<bool>());
	}
	auto &schema = binding.node.semantic.output_schema.fields;
	if (!json.is_object())
		throw InvalidInputException("Semantic projection requires a JSON object");
	if (schema.size() != 1)
		throw InvalidInputException("Semantic projection requires exactly one output field");
	const auto &field = schema[0];
	if (!json.contains(field.name))
		throw InvalidInputException("Missing semantic output key: %s", field.name);
	return Value::STRUCT({{field.name, ParseField(json.at(field.name), field)}});
}

}
}
