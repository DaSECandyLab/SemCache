#pragma once

#include "llm/json_utils.hpp"
#include <functional>
#include <memory>

namespace semantic {

// Admission batches group consecutive row/pair calls, never multiple inputs in
// one LLM request. Only the serial execution thread owns and updates this state.
// A future asynchronous policy must enqueue decisions onto that thread.
class RuntimePlanState {
public:
	using Json = llm_detail::Json;
	using Logger = std::function<void(const Json &)>;
	using PaidObservation = std::function<bool(int64_t, double)>;
	enum class Status { PENDING, RUNNING, DONE };
	struct Batch {
		SemanticOperator node;
		uint64_t generation = 0, sequence = 0, first_input = 0;
	};
	RuntimePlanState(const std::vector<SemanticOperator> &nodes, const std::set<std::string> &models,
	                 const std::string &options, Logger log, PaidObservation paid_observation = {});
	std::shared_ptr<const Batch> Admit(int64_t id, uint64_t input_index);
	bool ChunkMode() const {
		return chunk_mode;
	}
	void BeginChunk(int64_t id, uint64_t first_input, uint64_t capacity);
	void EndChunk(int64_t id);
	bool AtBoundary(int64_t id) const;
	void Observe(int64_t id, bool emitted, double latency_ms, bool requested = true);
	void Finish(int64_t id);
	// Atomic compare-and-swap of future configuration only. Task, schema, topology
	// and input identities are not part of the update protocol.
	void Update(const Json &updates);
	Json Report() const;
	Json Observations() const;
	Json Current(int64_t id) const;
	static Json Action(const SemanticConfig &config);

private:
	struct Observation {
		uint64_t completed = 0, emitted = 0, requests = 0;
		double latency_ms = 0;
		Json action;
	};
	struct Entry {
		SemanticOperator node;
		Status status = Status::PENDING;
		uint64_t generation = 0, next_batch_seq = 0, completed = 0, emitted = 0;
		uint64_t batch_completed = 0, batch_emitted = 0;
		uint64_t admitted = 0, capacity = 0;
		std::shared_ptr<const Batch> active;
		std::map<uint64_t, Observation> observations;
	};
	std::map<int64_t, Entry> entries;
	std::set<std::string> models;
	uint64_t batch_size;
	bool chunk_mode = false;
	Json rules;
	std::vector<bool> fired;
	Logger log;
	PaidObservation paid_observation;
	SemanticOperator ValidateUpdate(const Json &update) const;
	void CloseBatch(int64_t id, Entry &entry);
	static const char *Name(Status status);
};
}
