#pragma once
#include "duckdb/execution/physical_operator.hpp"
#include <atomic>
#include <functional>

namespace duckdb {
namespace semantic_extension {
struct SemanticInput {
	explicit SemanticInput(bool prefetch = false) : prefetch(prefetch) {
	}
	mutex lock;
	unique_ptr<DataChunk> chunk;
	const bool prefetch;
	bool requested = false, published = false, done = false, stopped = false;
	std::atomic<bool> available {false};
	vector<InterruptState> producers, consumers;
	std::function<void()> consumer_ready;

	static void Wake(vector<InterruptState> &waiters) {
		for (auto &waiter : waiters)
			waiter.Callback();
	}
	SinkResultType Publish(ClientContext &context, DataChunk &input, const InterruptState &interrupt) {
		vector<InterruptState> wake_consumers;
		std::function<void()> ready;
		SinkResultType result = SinkResultType::BLOCKED;
		{
			lock_guard<mutex> guard(lock);
			if (stopped)
				return SinkResultType::FINISHED;
			if (prefetch) {
				// One owned queued chunk bounds memory while the consumer processes
				// the previous chunk. Acceptance, not consumption, releases upstream.
				if (chunk) {
					producers.push_back(interrupt);
					return SinkResultType::BLOCKED;
				}
				result = SinkResultType::NEED_MORE_INPUT;
			} else {
				if (published && requested) {
					published = false;
					return SinkResultType::NEED_MORE_INPUT;
				}
				if (published || !requested) {
					producers.push_back(interrupt);
					return SinkResultType::BLOCKED;
				}
				published = true;
				requested = false;
				producers.push_back(interrupt);
			}
			chunk = make_uniq<DataChunk>();
			chunk->Initialize(context, input.GetTypes());
			input.Copy(*chunk);
			available.store(true);
			wake_consumers.swap(consumers);
			ready = std::move(consumer_ready);
		}
		// Never call a source or producer while holding the exchange lock.
		Wake(wake_consumers);
		if (ready)
			ready();
		return result;
	}
	SourceResultType Fetch(DataChunk &output, const InterruptState &interrupt) {
		return FetchInternal(output, &interrupt, {});
	}
	SourceResultType Fetch(DataChunk &output, std::function<void()> ready) {
		return FetchInternal(output, nullptr, std::move(ready));
	}
	bool Ready() const {
		return available.load();
	}
	SourceResultType FetchInternal(DataChunk &output, const InterruptState *interrupt, std::function<void()> ready) {
		vector<InterruptState> wake_producers;
		SourceResultType result;
		{
			lock_guard<mutex> guard(lock);
			if (chunk) {
				output.Reference(*chunk);
				chunk.reset();
				available.store(done || stopped);
				if (prefetch)
					wake_producers.swap(producers);
				result = SourceResultType::HAVE_MORE_OUTPUT;
			} else if (done || stopped) {
				return SourceResultType::FINISHED;
			} else {
				requested = true;
				wake_producers.swap(producers);
				if (interrupt)
					consumers.push_back(*interrupt);
				else
					consumer_ready = std::move(ready);
				result = SourceResultType::BLOCKED;
			}
		}
		Wake(wake_producers);
		return result;
	}
	void Request() {
		vector<InterruptState> wake_producers;
		{
			lock_guard<mutex> guard(lock);
			if (!chunk && !done && !stopped) {
				requested = true;
				wake_producers.swap(producers);
			}
		}
		Wake(wake_producers);
	}
	// PREPARE/EXECUTE reuses the plan, but never its exchange or callbacks.
	void Reset() {
		lock_guard<mutex> guard(lock);
		chunk.reset();
		requested = published = done = stopped = false;
		available.store(false);
		producers.clear();
		consumers.clear();
		consumer_ready = {};
	}
	void Complete() {
		vector<InterruptState> wake_consumers;
		std::function<void()> ready;
		{
			lock_guard<mutex> guard(lock);
			done = true;
			available.store(true);
			wake_consumers.swap(consumers);
			ready = std::move(consumer_ready);
		}
		Wake(wake_consumers);
		if (ready)
			ready();
	}
	void Stop() {
		vector<InterruptState> wake_producers, wake_consumers;
		std::function<void()> ready;
		{
			lock_guard<mutex> guard(lock);
			stopped = true;
			chunk.reset();
			available.store(true);
			wake_producers.swap(producers);
			wake_consumers.swap(consumers);
			ready = std::move(consumer_ready);
		}
		Wake(wake_producers);
		Wake(wake_consumers);
		if (ready)
			ready();
	}
};
}
}
