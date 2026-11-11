#pragma once


#include "duckdb/execution/expression_executor.hpp"
#include "runtime/query_runtime.hpp"
#include <utility>

namespace duckdb {
namespace semantic_extension {

// Measure only an executing source invocation, never the scheduler suspension
// between invocations. HTTP intervals enter the same union independently.
class SemanticOperatorActiveScope {
public:
	SemanticOperatorActiveScope(QueryRuntime &query, int64_t operator_id)
	    : runtime(query.file.is_open() ? &query : nullptr), operator_id(operator_id), started_ns(runtime ? Now() : 0) {
	}
	SemanticOperatorActiveScope(const SemanticOperatorActiveScope &) = delete;
	SemanticOperatorActiveScope &operator=(const SemanticOperatorActiveScope &) = delete;
	~SemanticOperatorActiveScope() noexcept {
		Stop();
	}
	void Stop() noexcept {
		if (!runtime)
			return;
		auto query = runtime;
		runtime = nullptr;
		try {
			query->RecordOperatorInterval(operator_id, started_ns, Now());
		} catch (...) {
			// Timing telemetry must not replace an execution/transport exception.
		}
	}

private:
	static int64_t Now() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
		    .count();
	}
	QueryRuntime *runtime;
	int64_t operator_id;
	int64_t started_ns;
};

struct SemanticRequestBlocked {};
struct SemanticRequestFailure : InvalidInputException {
	using InvalidInputException::InvalidInputException;
};
struct PendingSemanticRequest {
	std::shared_future<semantic::LLMResponse> response;
	// All original rows with the same immutable request share this identity.
	// It stays alive until the last mapped row has consumed its answer.
	std::shared_ptr<InFlightSemanticResult> shared_flight;
	size_t retained_bytes = 0;
	std::function<Value()> complete;
	std::function<void(std::function<void()>)> notify_when_ready;
	// Move the registration out first so repeated waits cannot add listeners.
	void SubscribeReady(std::function<void()> ready) {
		auto subscribe = std::move(notify_when_ready);
		notify_when_ready = {};
		if (subscribe)
			subscribe(std::move(ready));
	}
};

struct SemanticOperatorState : OperatorState {
	ExpressionExecutor executor;
	DataChunk arguments;
	SelectionVector selection;

	SemanticOperatorState(ExecutionContext &context, const vector<unique_ptr<Expression>> &expressions);
};

Value InferSemanticValue(ExecutionContext &context, const SemanticBindData &binding, QueryRuntime &runtime,
                         const semantic::Row &left, optional_ptr<const semantic::Row> right, uint64_t input_index,
                         uint64_t right_count = 0, uint64_t pair_index = UINT64_MAX, uint64_t left_index = UINT64_MAX,
                         uint64_t right_index = UINT64_MAX, semantic::RenderedPrompt *prepared_prompt = nullptr,
                         PendingSemanticRequest *pending = nullptr, std::function<void()> ready = {});

[[noreturn]] void FailSemanticRequest(const SemanticBindData &binding, QueryRuntime &runtime, Json &event,
                                      uint64_t sequence, const std::exception &error,
                                      const semantic::LLMResponse *confirmed_response = nullptr);

void RecordSemanticRequestFailure(QueryRuntime &runtime, Json &event, const std::exception &error,
                                  const semantic::LLMResponse *confirmed_response = nullptr);

void RecordCacheCorrection(Json &event, const semantic::LLMResponse &response);

vector<Value> InferSemanticSamples(ExecutionContext &context, const SemanticBindData &binding, QueryRuntime &runtime,
                                   DataChunk &arguments, idx_t begin, idx_t end, bool pair, uint64_t first_input);

}
}
