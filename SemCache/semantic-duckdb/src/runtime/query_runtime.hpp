#pragma once

// Query-lifetime resources shared by all semantic physical operators: model
// registry, transport, scheduler, counters and append-only telemetry.

#include "duckdb/main/client_context_state.hpp"
#include "llm/http_llm_client.hpp"
#include "llm/json_utils.hpp"
#include "llm/request_scheduler.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "llm/query_result_cache.hpp"
#include "prompt/row_serializer.hpp"
#include "prompt/prompt_renderer.hpp"
#include "runtime/runtime_plan_state.hpp"
#include "runtime/runtime_progress.hpp"
#include <fstream>
#include <atomic>
#include <chrono>
#include <future>
#include <deque>
#include <utility>

namespace duckdb {
namespace semantic_extension {

using Json = semantic::llm_detail::Json;

using SamplingBudgetExhausted = semantic::SamplingBudgetExhausted;

struct InFlightSemanticResult {
	std::mutex mutex;
	bool ready = false;
	uint64_t sequence = 0;
	std::shared_future<semantic::LLMResponse> response;
	std::vector<std::function<void()>> waiters;
	std::function<Value()> publish;
	bool published = false;
	bool dependency_publication = false;
	Value value;
	std::exception_ptr error;
	// Called only under the query's CPU-state mutex, including by a reuser
	// that wakes before the original owner. Publish the source observation once.
	Value Publish(bool dependency = false) {
		if (!published) {
			dependency_publication = dependency;
			try {
				value = publish();
			} catch (...) {
				error = std::current_exception();
			}
			published = true;
			publish = {};
		}
		if (error)
			std::rethrow_exception(error);
		return value;
	}
	void Wake() {
		std::vector<std::function<void()>> callbacks;
		{
			std::lock_guard<std::mutex> guard(mutex);
			ready = true;
			callbacks.swap(waiters);
		}
		for (auto &callback : callbacks) {
			try {
				callback();
			} catch (...) {
				// Scheduler readiness callbacks must not prevent another blocked
				// consumer from receiving this already-published completion.
			}
		}
	}
	void Listen(std::function<void()> callback) {
		{
			std::lock_guard<std::mutex> guard(mutex);
			if (!ready) {
				waiters.push_back(std::move(callback));
				return;
			}
		}
		callback();
	}
};

struct QueryRuntime {
	std::recursive_mutex mutex, submission_mutex;
	std::mutex log_mutex, timing_mutex;
	struct OperatorIntervals {
		// Disjoint, coalesced steady-clock intervals; never retain raw row timings.
		std::map<int64_t, int64_t> intervals;
		uint64_t union_ns = 0;
	};
	std::map<int64_t, OperatorIntervals> operator_intervals;
	std::ofstream file;
	semantic::HttpLLMClient client;
	semantic::RequestScheduler scheduler;
	std::map<int64_t, std::pair<uint64_t, unique_ptr<semantic::PromptRenderer>>> prompts;
	SemanticResultCache results;
	std::map<SemanticResultCache::Bucket *, std::map<string, std::shared_ptr<InFlightSemanticResult>>> in_flight;
	bool reuse_results = true;
	std::map<std::pair<int64_t, uint64_t>, SemanticResultCache::Bucket *> result_bindings;
	std::map<int64_t, uint64_t> request_counts, result_reuses;
	// Transport cache hits still pass through the existing operator completion
	// path. Track their admission identities separately from paid HTTP sends.
	bool next_request_cached = false;
	uint64_t cached_reserved = 0;
	std::map<int64_t, uint64_t> response_cache_hits;
	std::set<uint64_t> cached_request_sequences;
	// Only paid requests that have not published a semantic_row remain here.
	// Keep identities, never prompts or responses; normal/failed publication
	// erases each entry so memory follows bounded outstanding pipeline work.
	struct UnconsumedRequest {
		int64_t operator_id;
		string model_id;
	};
	std::map<uint64_t, UnconsumedRequest> unconsumed_requests;
	std::map<int64_t, uint64_t> discarded_requests;
	uint64_t sequence = 0;
	uint64_t llm_request_limit = UINT64_MAX, llm_submissions = 0, llm_reserved = 0;
	size_t max_inflight_per_model = 1;
	bool sampling = false;
	std::atomic<bool> budget_exhausted {false};
	void CheckRequestAllowance();
	void ReserveRequestAllowance();
	std::map<int64_t, std::pair<uint64_t, uint64_t>> counts;
	// Published semantic results can precede delivery of a physical output
	// chunk, including when another branch terminates the pipeline early.
	std::map<int64_t, std::pair<uint64_t, uint64_t>> published_counts;
	std::map<int64_t, std::vector<double>> latencies;
	std::map<int64_t, std::pair<uint64_t, double>> latency_totals;
	std::set<int64_t> operator_ids;
	std::set<int64_t> started;
	std::set<int64_t> finished;
	Json plan_spec;
	Json query_ir;
	Json pending_events = Json::array();
	std::future<Json> reoptimization;
	uint64_t reopt_interval = 0, last_reopt_sequence = 0;
	uint64_t completed_chunks = 0;
	unique_ptr<semantic::RuntimeProgress> progress;
	struct ChunkUsage {
		uint64_t responses = 0, known = 0, cached = 0, tokens = 0;
		uint64_t prompt_cache_hits = 0, prompt_cache_known_hits = 0, prompt_cache_tokens = 0;
	};
	std::map<int64_t, ChunkUsage> chunk_usage;
	std::map<int64_t, Json> operator_statistics;
	struct PendingPlanUpdate {
		Json updates, checkpoint;
		bool counted = false;
	};
	std::deque<PendingPlanUpdate> pending_plan_updates;
	Json reoptimization_checkpoint, pending_update_checkpoint;
	uint64_t runtime_switches = 0;
	string telemetry_path;
	int reopt_timeout_ms = 0;
	bool pending_complete = true;
	unique_ptr<semantic::RuntimePlanState> adaptive;
	std::vector<int64_t> start_order;
	std::chrono::steady_clock::time_point execution_start = std::chrono::steady_clock::now();

	QueryRuntime(const SemanticBindData &binding, const string &path, const string &runtime_options,
	             size_t parallelism = 1, std::shared_ptr<semantic::HttpLLMClient::ConnectionPool> pool = nullptr,
	             bool unrestricted = false, size_t max_inflight_per_model = 1,
	             bool enforce_model_operator_order = false);
	~QueryRuntime();
	void Log(const Json &event);
	void RecordOperatorInterval(int64_t operator_id, int64_t lo, int64_t hi);
	double OperatorProcessingSeconds(int64_t operator_id);
	void Reoptimize(ClientContext &context);
	Json RuntimeStatistics() const;
	void BeginChunk(ClientContext &context, int64_t id, uint64_t first, uint64_t rows);
	void EndChunk(ClientContext &context, int64_t id);
	SemanticResultCache::Bucket &ResultBucket(semantic::LLMRequest &request, uint64_t generation);
};

struct QueryState : ClientContextState {
	std::recursive_mutex mutex;
	unique_ptr<QueryRuntime> runtime;
	// Reuse sockets between sampling statements, without sharing query budgets,
	// result caches or request identities across statements.
	std::shared_ptr<semantic::HttpLLMClient::ConnectionPool> transport_pool;
	string models;
	string query_id;

	void QueryBegin(ClientContext &context) override;
	void QueryEnd(ClientContext &context, optional_ptr<ErrorData> error) override;
	QueryRuntime &Get(ClientContext &context, const SemanticBindData &binding);
	QueryRuntime &Ensure(ClientContext &context, const SemanticBindData &binding);
	void Finish(ClientContext &context, const SemanticBindData &binding);
};

semantic::Row ReadRow(const Value &value);
string SampleRowJSON(const Value &value);
Value ParseOutput(const string &raw, const SemanticBindData &binding);

}
}
