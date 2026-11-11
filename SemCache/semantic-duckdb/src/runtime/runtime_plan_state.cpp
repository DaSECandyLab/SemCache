#include "runtime/runtime_plan_state.hpp"
#include <cmath>
#include <limits>

namespace semantic {
namespace {
using Json = llm_detail::Json;
void Keys(const Json &value, const std::set<std::string> &keys) {
	if (!value.is_object() || value.size() != keys.size())
		throw ValidationError("runtime configuration has missing/unknown keys");
	for (auto &key : keys)
		if (!value.contains(key))
			throw ValidationError("runtime configuration missing " + key);
}
uint64_t Number(const Json &value) {
	if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
		throw ValidationError("runtime count/id/generation must be a nonnegative integer");
	return value.get<uint64_t>();
}
int64_t Id(const Json &value) {
	auto id = Number(value);
	if (id > uint64_t(std::numeric_limits<int64_t>::max()))
		throw ValidationError("runtime operator_id overflow");
	return int64_t(id);
}
}

const char *RuntimePlanState::Name(Status status) {
	return status == Status::PENDING ? "PENDING" : status == Status::RUNNING ? "RUNNING" : "DONE";
}
RuntimePlanState::Json RuntimePlanState::Action(const SemanticConfig &config) {
	return {{"operator_id", config.operator_id},
	        {"model_id", config.model_id},
	        {"layout", llm_detail::LayoutJSON(config.layout, config.input_fields.empty())}};
}
RuntimePlanState::RuntimePlanState(const std::vector<SemanticOperator> &nodes,
                                   const std::set<std::string> &known_models, const std::string &options, Logger logger,
                                   PaidObservation paid_observation)
    : models(known_models), log(std::move(logger)), paid_observation(std::move(paid_observation)) {
	for (auto &node : nodes) {
		ValidateSemanticConfig(node.kind, node.semantic);
		if (!models.count(node.semantic.model_id))
			throw ValidationError("unknown initial runtime model");
		Entry entry;
		entry.node = node;
		if (!entries.emplace(node.semantic.operator_id, entry).second)
			throw ValidationError("duplicate runtime operator_id");
	}
	auto config = llm_detail::ParseJSON(options);
	auto required = config;
	chunk_mode = Number(config["version"]) == 2;
	if (required.contains("value_policy")) {
		auto policy = required["value_policy"];
		if (chunk_mode && policy.contains("estimated_input_rows")) {
			if (!policy["estimated_input_rows"].is_object())
				throw ValidationError("Runtime input cardinalities must be an object");
			policy.erase("estimated_input_rows");
		}
		const std::string budget = chunk_mode ? "max_switches" : "interval_requests";
		Keys(policy, {budget, "timeout_ms"});
		if (!Number(policy[budget]) || Number(policy[budget]) > (chunk_mode ? 128 : 65536) ||
		    !Number(policy["timeout_ms"]) || Number(policy["timeout_ms"]) > 600000)
			throw ValidationError("invalid runtime value policy budget");
		required.erase("value_policy");
	}
	if (chunk_mode) {
		Keys(required, {"version"});
		if (!config.contains("value_policy"))
			throw ValidationError("Chunk adaptation requires a value policy");
	} else {
		Keys(required, {"version", "batch_size", "rules"});
	}
	if (Number(config["version"]) != 1 && !chunk_mode)
		throw ValidationError("unsupported runtime configuration version");
	batch_size = chunk_mode ? 1 : Number(config["batch_size"]);
	if (!batch_size || batch_size > 65536)
		throw ValidationError("runtime batch_size must be in 1..65536");
	rules = chunk_mode ? Json::array() : config["rules"];
	if (!rules.is_array() || rules.size() > 128)
		throw ValidationError("runtime rules must be an array of at most 128 rules");
	for (auto &rule : rules) {
		const std::string metric = rule.contains("selectivity_below") ? "selectivity_below" : "latency_above_ms";
		Keys(rule, {"source_operator_id", "min_completed", metric, "updates"});
		if (!entries.count(Id(rule["source_operator_id"])) || !Number(rule["min_completed"]))
			throw ValidationError("invalid runtime rule source/min_completed");
		if (!rule[metric].is_number() || !std::isfinite(rule[metric].get<double>()) || rule[metric].get<double>() < 0 ||
		    (metric == "selectivity_below" && rule[metric].get<double>() > 1))
			throw ValidationError("invalid runtime rule threshold");
		if (!rule["updates"].is_array() || rule["updates"].empty() || rule["updates"].size() > entries.size())
			throw ValidationError("invalid runtime rule updates");
		std::set<int64_t> targets;
		for (auto &update : rule["updates"]) {
			ValidateUpdate(update);
			if (!targets.insert(Id(update["operator_id"])).second)
				throw ValidationError("duplicate runtime update target");
		}
	}
	fired.resize(rules.size(), false);
	log({{"event", "runtime_initialized"}, {"options", config}, {"state", Report()}});
}

SemanticOperator RuntimePlanState::ValidateUpdate(const Json &update) const {
	Keys(update, {"operator_id", "expected_generation", "model_id", "layout"});
	auto item = entries.find(Id(update["operator_id"]));
	if (item == entries.end())
		throw ValidationError("unknown runtime operator_id");
	Number(update["expected_generation"]);
	auto node = item->second.node;
	auto &config = node.semantic;
	config.model_id = update["model_id"].get<std::string>();
	if (!models.count(config.model_id))
		throw ValidationError("unknown runtime model_id");
	auto &layout = update["layout"];
	if (node.kind == NodeKind::SEMANTIC_JOIN) {
		Keys(layout, {"block_order", "left_field_order", "right_field_order"});
		config.layout.left_field_order = layout["left_field_order"].get<FieldOrder>();
		config.layout.right_field_order = layout["right_field_order"].get<FieldOrder>();
	} else {
		Keys(layout, {"block_order", "field_order"});
		config.layout.field_order = layout["field_order"].get<FieldOrder>();
	}
	if (!layout["block_order"].is_array())
		throw ValidationError("runtime block_order must be an array");
	config.layout.block_order.clear();
	for (auto &block : layout["block_order"]) {
		if (block == "OPERATOR_SPECIFICATION")
			config.layout.block_order.push_back(PromptBlock::OPERATOR_SPECIFICATION);
		else if (block == "TASK_DESCRIPTION")
			config.layout.block_order.push_back(PromptBlock::TASK_DESCRIPTION);
		else if (block == "DATA_BLOCK")
			config.layout.block_order.push_back(PromptBlock::DATA_BLOCK);
		else
			throw ValidationError("unknown runtime prompt block");
	}
	ValidateSemanticConfig(node.kind, config);
	if (chunk_mode) {
		const auto &b = config.layout.block_order;
		if (b != std::vector<PromptBlock>(
		             {PromptBlock::OPERATOR_SPECIFICATION, PromptBlock::TASK_DESCRIPTION, PromptBlock::DATA_BLOCK}) &&
		    b != std::vector<PromptBlock>(
		             {PromptBlock::OPERATOR_SPECIFICATION, PromptBlock::DATA_BLOCK, PromptBlock::TASK_DESCRIPTION}) &&
		    b != std::vector<PromptBlock>(
		             {PromptBlock::DATA_BLOCK, PromptBlock::OPERATOR_SPECIFICATION, PromptBlock::TASK_DESCRIPTION}))
			throw ValidationError("chunk runtime layouts are limited to OTD, ODT and DOT");
	}
	return node;
}

void RuntimePlanState::Update(const Json &updates) {
	if (!updates.is_array() || updates.empty() || updates.size() > entries.size())
		throw ValidationError("invalid runtime updates");
	std::map<int64_t, SemanticOperator> validated;
	// Validate the entire transaction before publishing any future configuration.
	for (auto &update : updates) {
		auto node = ValidateUpdate(update);
		auto id = node.semantic.operator_id;
		auto &entry = entries.at(id);
		if (entry.status == Status::DONE)
			throw ValidationError("cannot update DONE semantic operator");
		if (chunk_mode && entry.active)
			throw ValidationError("chunk implementation updates require an input chunk boundary");
		if (Number(update["expected_generation"]) != entry.generation)
			throw ValidationError("stale runtime config_generation");
		if (entry.generation == std::numeric_limits<uint64_t>::max())
			throw ValidationError("runtime generation exhausted");
		if (!validated.emplace(id, std::move(node)).second)
			throw ValidationError("duplicate runtime update target");
	}
	Json changes = Json::array();
	for (auto &item : validated) {
		auto &entry = entries.at(item.first);
		auto previous = Current(item.first);
		entry.node = std::move(item.second);
		++entry.generation;
		changes.push_back(
		    {{"before", previous}, {"after", Current(item.first)}, {"effective_from_batch_seq", entry.next_batch_seq}});
	}
	log({{"event", "runtime_update"}, {"changes", changes}});
}

std::shared_ptr<const RuntimePlanState::Batch> RuntimePlanState::Admit(int64_t id, uint64_t input_index) {
	auto &entry = entries.at(id);
	if (chunk_mode) {
		if (!entry.active || entry.status != Status::RUNNING || entry.admitted >= entry.capacity ||
		    input_index != entry.active->first_input + entry.admitted)
			throw ValidationError("runtime chunk admission order violation");
		++entry.admitted;
		return entry.active;
	}
	if (entry.status == Status::DONE || input_index != entry.completed)
		throw ValidationError("runtime admission order violation");
	if (!entry.active) {
		entry.status = Status::RUNNING;
		auto batch = std::make_shared<Batch>();
		// 批次准入时复制配置；之后的 generation 更新只影响新批次，
		// 已准入但尚未发出的行/候选对仍使用这个快照。
		batch->node = entry.node;
		batch->generation = entry.generation;
		batch->sequence = entry.next_batch_seq++;
		batch->first_input = input_index;
		entry.active = batch;
		entry.batch_completed = entry.batch_emitted = 0;
		log({{"event", "batch_admitted"},
		     {"operator_id", id},
		     {"batch_seq", batch->sequence},
		     {"config_generation", batch->generation},
		     {"first_input", input_index},
		     {"capacity", batch_size},
		     {"action", Action(batch->node.semantic)}});
	}
	return entry.active;
}

void RuntimePlanState::BeginChunk(int64_t id, uint64_t first_input, uint64_t capacity) {
	if (!chunk_mode)
		return;
	auto &entry = entries.at(id);
	if (entry.active || entry.status == Status::DONE || first_input != entry.completed || !capacity)
		throw ValidationError("Invalid runtime chunk boundary");
	entry.status = Status::RUNNING;
	auto batch = std::make_shared<Batch>();
	batch->node = entry.node;
	batch->generation = entry.generation;
	batch->sequence = entry.next_batch_seq++;
	batch->first_input = first_input;
	entry.active = batch;
	entry.capacity = capacity;
	entry.admitted = entry.batch_completed = entry.batch_emitted = 0;
	log({{"event", "batch_admitted"},
	     {"operator_id", id},
	     {"batch_seq", batch->sequence},
	     {"config_generation", batch->generation},
	     {"first_input", first_input},
	     {"capacity", capacity},
	     {"boundary", "duckdb_data_chunk"},
	     {"action", Action(batch->node.semantic)}});
}

void RuntimePlanState::EndChunk(int64_t id) {
	if (!chunk_mode)
		return;
	auto &entry = entries.at(id);
	if (!entry.active || entry.batch_completed != entry.admitted)
		throw ValidationError("Runtime chunk has unfinished inputs");
	CloseBatch(id, entry);
}

bool RuntimePlanState::AtBoundary(int64_t id) const {
	return !entries.at(id).active;
}

void RuntimePlanState::CloseBatch(int64_t id, Entry &entry) {
	if (!entry.active)
		return;
	log({{"event", "batch_completed"},
	     {"operator_id", id},
	     {"batch_seq", entry.active->sequence},
	     {"config_generation", entry.active->generation},
	     {"completed", entry.batch_completed},
	     {"emitted", entry.batch_emitted}});
	entry.active.reset();
}

void RuntimePlanState::Observe(int64_t id, bool emitted, double latency_ms, bool requested) {
	auto &entry = entries.at(id);
	if (!entry.active || entry.status != Status::RUNNING || !std::isfinite(latency_ms) || latency_ms < 0)
		throw ValidationError("runtime observation without an active batch or valid latency");
	if (requested && paid_observation)
		requested = paid_observation(id, latency_ms);
	++entry.completed;
	++entry.batch_completed;
	entry.emitted += emitted;
	entry.batch_emitted += emitted;
	// Keep model/layout generations separate: mixing old and new measurements
	// would feed a model-specific reoptimizer the wrong selectivity and latency.
	auto &observed = entry.observations[entry.active->generation];
	if (!observed.completed)
		observed.action = Action(entry.active->node.semantic);
	++observed.completed;
	observed.requests += requested;
	observed.emitted += emitted;
	if (requested)
		observed.latency_ms += latency_ms;
	// Evaluate after each completed call, even while the admission batch still
	// contains future calls. Updates must leave that batch's snapshot untouched.
	for (size_t i = 0; i < rules.size(); ++i) {
		auto &rule = rules[i];
		if (fired[i] || Id(rule["source_operator_id"]) != id || observed.completed < Number(rule["min_completed"]))
			continue;
		const double completed = static_cast<double>(observed.completed);
		const double selectivity = static_cast<double>(observed.emitted) / completed;
		const double latency = observed.requests ? observed.latency_ms / observed.requests : 0;
		if (rule.contains("selectivity_below")
		        ? selectivity >= rule["selectivity_below"].get<double>()
		        : !observed.requests || latency <= rule["latency_above_ms"].get<double>())
			continue;
		fired[i] = true;
		log({{"event", "reoptimization_trigger"},
		     {"rule_id", i},
		     {"source_operator_id", id},
		     {"config_generation", entry.active->generation},
		     {"action", observed.action},
		     {"completed", observed.completed},
		     {"emitted", observed.emitted},
		     {"selectivity", selectivity},
		     {"llm_calls", observed.requests},
		     {"latency_avg_ms", observed.requests ? Json(latency) : Json(nullptr)},
		     {"accuracy", nullptr},
		     {"cache_metrics", nullptr}});
		try {
			Update(rule["updates"]);
		} catch (const ValidationError &error) {
			// A stale or DONE-target decision is rejected atomically and once; it
			// cannot cause a replay, a partial switch or a hidden retry.
			log({{"event", "runtime_update_rejected"}, {"rule_id", i}, {"reason", error.what()}});
		}
	}
	if (!chunk_mode && entry.batch_completed == batch_size)
		CloseBatch(id, entry);
}

void RuntimePlanState::Finish(int64_t id) {
	auto &entry = entries.at(id);
	if (entry.status == Status::DONE)
		throw ValidationError("runtime operator finalized twice");
	CloseBatch(id, entry);       // A final partial batch is valid, including across DataChunks.
	entry.status = Status::DONE; // Empty operators finish without admitting any batch.
	log({{"event", "runtime_operator_done"}, {"operator_id", id}, {"state", Current(id)}});
}

RuntimePlanState::Json RuntimePlanState::Current(int64_t id) const {
	auto &entry = entries.at(id);
	auto current = Action(entry.node.semantic);
	current["config_generation"] = entry.generation;
	return {{"operator_id", id},
	        {"status", Name(entry.status)},
	        {"current", current},
	        {"next_batch_seq", entry.next_batch_seq},
	        {"completed", entry.completed},
	        {"emitted", entry.emitted}};
}
RuntimePlanState::Json RuntimePlanState::Report() const {
	Json result = Json::array();
	for (auto &item : entries)
		result.push_back(Current(item.first));
	return result;
}
RuntimePlanState::Json RuntimePlanState::Observations() const {
	Json result = Json::array();
	for (auto &item : entries) {
		for (auto &generation : item.second.observations) {
			auto &observed = generation.second;
			const double completed = static_cast<double>(observed.completed);
			result.push_back(
			    {{"operator_id", item.first},
			     {"config_generation", generation.first},
			     {"action", observed.action},
			     {"completed", observed.completed},
			     {"emitted", observed.emitted},
			     {"llm_calls", observed.requests},
			     {"selectivity", static_cast<double>(observed.emitted) / completed},
			     {"latency_avg_ms", observed.requests ? Json(observed.latency_ms / observed.requests) : Json(nullptr)},
			     {"accuracy", nullptr},
			     {"cache_metrics", nullptr}});
		}
	}
	return result;
}
}
