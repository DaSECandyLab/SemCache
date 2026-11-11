#pragma once
#include "duckdb/execution/physical_operator_states.hpp"
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>

namespace duckdb {
namespace semantic_extension {
template <class PendingRows>
typename PendingRows::iterator FirstReadySemanticRow(PendingRows &rows) {
	for (auto row = rows.begin(); row != rows.end(); ++row) {
		if (row->ready || !row->pending.response.valid() ||
		    row->pending.response.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
			return row;
	}
	return rows.end();
}

// Completion can race with source teardown. The gate prevents callbacks from
// touching a destroyed GlobalSourceState; task registration uses DuckDB's own
// lock and BlockSource/UnblockTasks protocol.
struct SemanticSourceState : GlobalSourceState {
	struct WakeState {
		std::mutex mutex;
		GlobalSourceState *source;
		explicit WakeState(GlobalSourceState *source) : source(source) {
		}
		void Wake() {
			std::lock_guard<std::mutex> guard(mutex);
			if (source) {
				auto source_guard = source->Lock();
				source->UnblockTasks(source_guard);
			}
		}
	};
	std::shared_ptr<WakeState> wake = std::make_shared<WakeState>(this);
	~SemanticSourceState() override {
		std::lock_guard<std::mutex> guard(wake->mutex);
		wake->source = nullptr;
	}
	std::function<void()> CompletionCallback() {
		auto state = wake;
		return [state] {
			state->Wake();
		};
	}
	// Subscribe outside the source lock: a response that is already ready may
	// invoke its callback immediately. Recheck every candidate under the lock
	// before blocking, so any completion can release a bounded window slot.
	template <class PendingRows, class UpstreamReady>
	bool AwaitAny(PendingRows &rows, UpstreamReady upstream_ready, const InterruptState &interrupt,
	              SourceResultType &result) {
		for (auto &row : rows) {
			if (!row.ready && row.pending.response.valid() &&
			    row.pending.response.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
				row.pending.SubscribeReady(CompletionCallback());
		}
		auto guard = Lock();
		if (upstream_ready() || FirstReadySemanticRow(rows) != rows.end())
			return false;
		result = BlockSource(guard, interrupt);
		return true;
	}
	// False means the response arrived before registration; callers retry the
	// pending row without returning BLOCKED. FINISHED honours PreventBlocking.
	template <class Response>
	bool Await(const std::shared_future<Response> &response, const InterruptState &interrupt,
	           SourceResultType &result) {
		auto guard = Lock();
		if (!response.valid() || response.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
			return false;
		result = BlockSource(guard, interrupt);
		return true;
	}
};
}
}
