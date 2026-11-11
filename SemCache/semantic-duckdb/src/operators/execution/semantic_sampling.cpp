// Single-record sampling shares transport futures; only the owner publishes statistics
// and outputs, in input order. Complete query execution retains its own scheduler.
#include "operators/execution/semantic_execution.hpp"

namespace duckdb {
namespace semantic_extension {

vector<Value> InferSemanticSamples(ExecutionContext &context, const SemanticBindData &binding, QueryRuntime &runtime,
                                   DataChunk &arguments, idx_t begin, idx_t end, bool pair, uint64_t first_input) {
	if (begin >= end || !binding.sampling || runtime.adaptive)
		throw InvalidInputException("Invalid native sampling input");
	vector<std::function<Value()>> pending;
	vector<Value> results;
	std::exception_ptr error;
	for (auto row = begin; row < end; ++row) {
		PendingSemanticRequest request;
		try {
			if (context.client.IsInterrupted())
				throw InterruptException();
			auto left = ReadRow(arguments.GetValue(0, row));
			auto right = pair ? ReadRow(arguments.GetValue(1, row)) : semantic::Row {};
			auto value = InferSemanticValue(context, binding, runtime, left, pair ? &right : nullptr,
			                                first_input + row - begin, 0, UINT64_MAX, UINT64_MAX, UINT64_MAX,
			                                nullptr, &request);
			pending.push_back([value]() { return value; });
		} catch (const SemanticRequestBlocked &) {
			pending.push_back(std::move(request.complete));
		} catch (const SamplingBudgetExhausted &) {
			break;
		} catch (...) {
			error = std::current_exception();
			break;
		}
	}
	// Always join paid work, even after an error or admission-budget exhaustion.
	// Publishing on this thread keeps DuckDB values and shared statistics serial.
	bool prefix_stopped = false;
	for (auto &complete : pending) {
		try {
			auto value = complete();
			if (!prefix_stopped)
				results.push_back(std::move(value));
		} catch (const SamplingBudgetExhausted &) {
			// An incomplete response can exhaust its retry allowance after earlier
			// inputs completed. Retain that contiguous prefix, drain later sends,
			// and never shift a later answer onto the missing input's identity.
			prefix_stopped = true;
		} catch (...) {
			if (!error)
				error = std::current_exception();
		}
	}
	if (error)
		std::rethrow_exception(error);
	if (results.empty() && runtime.budget_exhausted)
		throw SamplingBudgetExhausted();
	return results;
}
}
}
