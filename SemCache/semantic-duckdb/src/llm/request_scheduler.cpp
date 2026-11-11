#include "llm/request_scheduler.hpp"
#include "llm/http_llm_client.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <unistd.h>
#include "llm/json_utils.hpp"
#include "prompt/row_serializer.hpp"

namespace semantic {
namespace {
struct SharedRequestQueue {
	SharedRequestQueue() {
		const char *value = std::getenv("SEMCACHE_GLOBAL_HTTP_CONCURRENCY");
		if (!value || !*value)
			return;
		char *end = nullptr;
		const long parsed = std::strtol(value, &end, 10);
		if (!end || *end || parsed < 1 || parsed > 1024)
			throw ValidationError("SEMCACHE_GLOBAL_HTTP_CONCURRENCY must be between 1 and 1024");
		limit = size_t(parsed);
		const char *path = std::getenv("SEMCACHE_GLOBAL_HTTP_AUDIT");
		if (path)
			audit_path = path;
	}
	std::mutex mutex;
	std::condition_variable changed;
	size_t limit = 0, active = 0, peak = 0;
	uint64_t next_ticket = 0, serving_ticket = 0;
	std::string audit_path;
};

SharedRequestQueue &GlobalRequestQueue() {
	static SharedRequestQueue queue;
	return queue;
}
}

SharedRequestAdmission::SharedRequestAdmission() {
	auto &queue = GlobalRequestQueue();
	if (!queue.limit)
		return;
	std::unique_lock<std::mutex> guard(queue.mutex);
	const auto ticket = queue.next_ticket++;
	queue.changed.wait(guard, [&] { return ticket == queue.serving_ticket && queue.active < queue.limit; });
	++queue.serving_ticket;
	++queue.active;
	admitted = true;
	if (queue.active > queue.peak) {
		queue.peak = queue.active;
		if (!queue.audit_path.empty()) {
			std::ofstream audit(queue.audit_path, std::ios::app);
			audit << "{\"pid\":" << getpid() << ",\"limit\":" << queue.limit << ",\"peak\":" << queue.peak << "}\n";
		}
	}
	queue.changed.notify_all();
}

SharedRequestAdmission::~SharedRequestAdmission() {
	Release();
}

void SharedRequestAdmission::Release() {
	if (!admitted)
		return;
	auto &queue = GlobalRequestQueue();
	std::lock_guard<std::mutex> guard(queue.mutex);
	--queue.active;
	admitted = false;
	queue.changed.notify_all();
}

RequestScheduler::RequestScheduler(std::string id, LLMClient &client, std::ostream *log, size_t parallelism,
                                   std::mutex *shared_audit_mutex, bool unrestricted, size_t max_inflight_per_model,
                                   bool enforce_model_operator_order)
    : query_id(std::move(id)), client(client), audit_log(log),
      audit_mutex(shared_audit_mutex ? *shared_audit_mutex : own_audit_mutex), unrestricted(unrestricted),
      max_inflight_per_model(max_inflight_per_model), enforce_model_operator_order(enforce_model_operator_order) {
	if (query_id.empty())
		throw ValidationError("scheduler query_id must not be empty");
	QuoteText(query_id);
	if (!parallelism || parallelism > 1024)
		throw ValidationError("semantic_parallelism must be between 1 and 1024");
	if (!max_inflight_per_model || max_inflight_per_model > 1024)
		throw ValidationError("semantic_max_concurrent_requests_per_model must be between 1 and 1024");
}
RequestScheduler::~RequestScheduler() {
	Shutdown();
}
void RequestScheduler::SetOperatorOrder(const std::vector<int64_t> &operators) {
	std::lock_guard<std::mutex> guard(mutex);
	if (next_seq != 1)
		throw ValidationError("operator order must be fixed before request admission");
	std::unordered_map<int64_t, size_t> ranks;
	for (size_t i = 0; i < operators.size(); ++i)
		if (!ranks.emplace(operators[i], i).second)
			throw ValidationError("operator order contains duplicate operators");
	operator_ranks = std::move(ranks);
}
void RequestScheduler::Shutdown() {
	std::deque<std::shared_ptr<Work>> cancelled;
	{
		std::lock_guard<std::mutex> guard(mutex);
		stopping = true;
		cancelled.swap(queue);
		for (auto &model : model_queues)
			for (auto &entry : model.second)
				cancelled.push_back(std::move(entry.second));
		model_queues.clear();
		for (const auto &work : cancelled) {
			auto found = outstanding_models.find(work->model_key);
			if (--found->second == 0)
				outstanding_models.erase(found);
		}
	}
	for (auto &work : cancelled) {
		work->promise.set_exception(std::make_exception_ptr(
		    LLMError(work->request, "scheduler", 0, "query stopped before transport admission")));
		if (work->ready) {
			try {
				work->ready();
			} catch (...) {
			}
		}
	}
	changed.notify_all();
	for (auto &worker : workers)
		if (worker.joinable())
			worker.join();
	workers.clear();
}
std::shared_future<LLMResponse> RequestScheduler::Submit(LLMRequest request, std::function<void()> ready) {
	std::lock_guard<std::mutex> guard(mutex);
	if (failed || stopping)
		throw LLMError(request, "scheduler", 0, "query already failed; no further requests admitted");
	if (request.query_id != query_id || request.request_seq != 0)
		throw LLMError(request, "scheduler", 0, "query mismatch or caller-assigned request_seq");
	if (next_seq == std::numeric_limits<uint64_t>::max())
		throw LLMError(request, "scheduler", 0, "request sequence exhausted");
	auto work = std::make_shared<Work>();
	const auto rank = operator_ranks.find(request.operator_id);
	work->operator_rank = rank == operator_ranks.end() ? operator_ranks.size() : rank->second;
	request.request_seq = next_seq++;
	try {
		work->model_key = client.SchedulingKey(request);
	} catch (...) {
		work->model_key = request.model_id;
		work->preparation_error = std::current_exception();
	}
	work->request = std::move(request);
	work->ready = std::move(ready);
	auto future = work->promise.get_future().share();
	if (unrestricted)
		queue.push_back(work);
	else
		model_queues[work->model_key].emplace(QueueKey(*work), work);
	++outstanding_models[work->model_key];
	// Normal execution has a bounded number of workers per physical model. Beam
	// sampling explicitly removes that lane cap: each admitted request gets a
	// worker, while the sampling ledger remains the independent request bound.
	size_t desired_workers = queue.size() + active_requests;
	if (!unrestricted) {
		desired_workers = 0;
		for (const auto &model : outstanding_models)
			desired_workers += std::min(model.second, max_inflight_per_model);
	}
	while (workers.size() < desired_workers)
		workers.emplace_back(&RequestScheduler::Worker, this);
	changed.notify_one();
	return future;
}
LLMResponse RequestScheduler::Infer(LLMRequest request) {
	return Submit(std::move(request)).get();
}
void RequestScheduler::Worker() {
	for (;;) {
		std::shared_ptr<Work> work;
		size_t concurrency, model_concurrency;
		bool rejected;
		{
			std::unique_lock<std::mutex> guard(mutex);
			auto eligible = [&]() -> std::shared_ptr<Work> {
				if (unrestricted)
					return queue.empty() ? nullptr : queue.front();
				std::shared_ptr<Work> selected;
				for (const auto &model : model_queues) {
					if (model.second.empty())
						continue;
					const auto active = active_model_requests.find(model.first);
					if (!failed && active != active_model_requests.end() && active->second >= max_inflight_per_model)
						continue;
					const auto &head = model.second.begin()->second;
					if (!selected || head->request.request_seq < selected->request.request_seq)
						selected = head;
				}
				return selected;
			};
			changed.wait(guard, [&] {
				work = eligible();
				return work || (stopping && queue.empty() && model_queues.empty());
			});
			if (!work)
				return;
			if (unrestricted) {
				queue.pop_front();
			} else {
				auto model = model_queues.find(work->model_key);
				model->second.erase(QueueKey(*work));
				// Keep the lane while it has in-flight work. A sliding window
				// frequently empties its queue between two submissions.
			}
			rejected = failed;
			++active_model_requests[work->model_key];
			concurrency = ++active_requests;
			model_concurrency = active_model_requests.at(work->model_key);
		}
		auto &request = work->request;
		llm_detail::Json event {
		    {"query_id", query_id},         {"operator_id", request.operator_id}, {"request_seq", request.request_seq},
		    {"model_id", request.model_id}, {"concurrency", concurrency},         {"retry_count", 0}};
		event["operator_rank"] = work->operator_rank;
		event["model_concurrency"] = model_concurrency;
		auto write = [&](const std::string &phase) {
			if (!audit_log)
				return;
			std::lock_guard<std::mutex> guard(audit_mutex);
			event["event"] = phase;
			*audit_log << event.dump() << '\n';
			audit_log->flush();
			if (!*audit_log)
				throw ValidationError("scheduler audit log write failed");
		};
		std::exception_ptr error;
		bool budget_exhausted = false;
		LLMResponse response;
		try {
			if (rejected)
				throw LLMError(request, "scheduler", 0, "query already failed; no further requests admitted");
			if (work->preparation_error)
				std::rethrow_exception(work->preparation_error);
			client.ConfigureRequest(request);
			ValidateRequest(request);
			event["temperature"] = request.temperature;
			event["max_tokens"] = request.max_tokens;
			response = client.InferWithAdmission(request, [&] { write("submit"); });
			if (!response.result_cache_hit)
				write("complete");
		} catch (const SamplingBudgetExhausted &) {
			// Sampling exhaustion is a normal bounded-observation outcome. Keep
			// its type intact so the semantic operator can stop the current input
			// chunk and publish the paid prefix instead of failing the query.
			error = std::current_exception();
			budget_exhausted = true;
		} catch (const LLMError &) {
			error = std::current_exception();
		} catch (const std::exception &ex) {
			error = std::make_exception_ptr(LLMError(request, "scheduler", 0, ex.what()));
		} catch (...) {
			error = std::current_exception();
		}
		if (error) {
			if (!budget_exhausted) {
				std::lock_guard<std::mutex> guard(mutex);
				failed = true;
			}
			try {
				std::rethrow_exception(error);
			} catch (const std::exception &ex) {
				try {
					event["error"] = ex.what();
					write("error");
				} catch (...) {
				}
			} catch (...) {
			}
		}
		{
			std::lock_guard<std::mutex> guard(mutex);
			auto active = active_model_requests.find(work->model_key);
			if (active != active_model_requests.end() && --active->second == 0)
				active_model_requests.erase(active);
			--active_requests;
			auto found = outstanding_models.find(work->model_key);
			if (--found->second == 0) {
				model_queues.erase(work->model_key);
				outstanding_models.erase(found);
			}
		}
		if (error)
			work->promise.set_exception(error);
		else
			work->promise.set_value(std::move(response));
		// Publish readiness before waking DuckDB's blocked task.
		if (work->ready) {
			try {
				work->ready();
			} catch (...) {
			}
		}
		if (error && !budget_exhausted)
			changed.notify_all();
		else
			changed.notify_one();
	}
}
}
