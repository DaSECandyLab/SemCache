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
	const auto &final_event = record["event"];
	if (final_event == "planning_decision" || final_event == "runtime_chunk_completed" ||
	    final_event == "runtime_update" || final_event == "runtime_plan_update_applied" ||
	    final_event == "operator_summary" || final_event == "sampling_stage_progress" ||
	    final_event == "execution_report")
		file.flush();
	if (!file)
		throw InvalidInputException("Semantic telemetry write failed");
}


}
}
