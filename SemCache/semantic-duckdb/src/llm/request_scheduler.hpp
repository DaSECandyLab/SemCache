#pragma once

// Query-owned transport workers. Normal query execution uses FIFO per physical
// model, with optional plan-operator priority and a bounded in-flight window.
// Bounded Beam sampling can explicitly opt into unrestricted request workers
// because its requests are independent observations.
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_map>
#include "llm/llm_client.hpp"

namespace semantic {
// Shared FIFO admission for physical HTTP sends across every model and query
// scheduler in this process. Unset SEMCACHE_GLOBAL_HTTP_CONCURRENCY keeps the
// existing per-model policy. Cache hits never enter this queue; retries do.
class SharedRequestAdmission {
public:
	SharedRequestAdmission();
	~SharedRequestAdmission();
	void Release();
	SharedRequestAdmission(const SharedRequestAdmission &) = delete;
	SharedRequestAdmission &operator=(const SharedRequestAdmission &) = delete;

private:
	bool admitted = false;
};

class RequestScheduler {
public:
	RequestScheduler(std::string query_id, LLMClient &client, std::ostream *audit_log = nullptr, size_t parallelism = 1,
	                 std::mutex *audit_mutex = nullptr, bool unrestricted = false, size_t max_inflight_per_model = 1,
	                 bool enforce_model_operator_order = false);
	~RequestScheduler();
	// Register the validated plan order before admitting any requests. Among
	// queued requests for the same physical model, earlier operators take
	// precedence over enqueue timing only when enforce_model_operator_order is
	// enabled. Requests from different operators may overlap within that model's
	// in-flight limit in both modes; other models are independent.
	void SetOperatorOrder(const std::vector<int64_t> &operators);
	std::shared_future<LLMResponse> Submit(LLMRequest request, std::function<void()> ready = {});
	LLMResponse Infer(LLMRequest request);
	void Shutdown();

private:
	struct Work {
		LLMRequest request;
		std::string model_key;
		size_t operator_rank = 0;
		std::exception_ptr preparation_error;
		std::promise<LLMResponse> promise;
		std::function<void()> ready;
	};
	std::pair<size_t, uint64_t> QueueKey(const Work &work) const {
		return {enforce_model_operator_order ? work.operator_rank : 0, work.request.request_seq};
	}
	void Worker();
	const std::string query_id;
	LLMClient &client;
	std::ostream *audit_log;
	std::mutex own_audit_mutex;
	std::mutex &audit_mutex;
	std::mutex mutex;
	std::condition_variable changed;
	// Sampling is FIFO. Execution optionally ranks within each physical model;
	// selecting its next request scans model heads, never the whole backlog.
	std::deque<std::shared_ptr<Work>> queue;
	using ModelQueue = std::map<std::pair<size_t, uint64_t>, std::shared_ptr<Work>>;
	std::unordered_map<std::string, ModelQueue> model_queues;
	std::unordered_map<std::string, size_t> outstanding_models;
	std::unordered_map<int64_t, size_t> operator_ranks;
	std::unordered_map<std::string, size_t> active_model_requests;
	std::vector<std::thread> workers;
	const bool unrestricted;
	const size_t max_inflight_per_model;
	const bool enforce_model_operator_order;
	size_t active_requests = 0;
	uint64_t next_seq = 1;
	bool failed = false, stopping = false;
};
}
