#include "optimizer/semantic_optimizer.hpp"
#include "optimizer/semantic_optimizer_internal.hpp"
#include "optimizer/optimizer_client.hpp"
#include "llm/model_registry.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "operators/plan/rewrite/semantic_predicates.hpp"
#include "operators/plan/semantic_plan_tree.hpp"
#include "runtime/result_capture.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace duckdb {
namespace semantic_extension {
namespace optimizer_detail {

int64_t Id(const Json &value) {
	if (!value.is_number_integer() ||
	    (value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
	    value.get<int64_t>() < 0)
		throw BinderException("PlanSpec IDs/version must be nonnegative integers");
	return value.get<int64_t>();
}

} // namespace optimizer_detail

using namespace optimizer_detail;
namespace {
void Keys(const Json &object, std::initializer_list<const char *> keys) {
	if (!object.is_object() || object.size() != keys.size())
		throw BinderException("PlanSpec has missing or unknown keys");
	for (auto key : keys)
		if (!object.contains(key))
			throw BinderException("PlanSpec missing key %s", key);
}
Json SamplingDecision(const Json &query, const string &text) {
	auto setting = semantic::llm_detail::ParseJSON(text);
	auto keys = setting;
	keys.erase("llm_request_limit");
	keys.erase("result_cache_probe");
	Keys(keys, {"model_id", "layout", "candidate_pairs"});
	if (!setting["model_id"].is_string() || !setting["candidate_pairs"].is_boolean() ||
	    query["semantic_operators"].size() != 1)
		throw BinderException("Native sampling requires one explicitly configured semantic stage");
	auto &op = query["semantic_operators"][0];
	Json spec {{"version", 1},
	           {"kind", "PlanSpec"},
	           {"query_id", query["query_id"]},
	           {"actions", Json::array({{{"operator_id", op["operator_id"]},
	                                     {"model_id", setting["model_id"]},
	                                     {"layout", setting["layout"]}}})}};
	if (!setting["candidate_pairs"].get<bool>())
		return spec;
	// Selected pairs use two private flat tables linked by their unique
	// sample identity. Explicitly schedule that relational equality before the
	// semantic predicate; SQL conjunct order alone does not guarantee this.
	if (op["kind"] != "SEMANTIC_JOIN" || !query.contains("tree_regions") || query["tree_regions"].size() != 1)
		throw BinderException("Native selected-pair stage requires one bound semantic join region");
	auto &region = query["tree_regions"][0];
	// The selected-pair identity equality is a complete relational join.
	// The semantic operator consumes only its output pairs, never a Cartesian
	// state with a separately scheduled identity predicate.
	if (region["nodes"].size() != 2 || region["base_nodes"].size() != 2)
		throw BinderException("Native selected-pair stage requires one atomic identity join and two flat inputs");
	const Json *join = nullptr, *semantic = nullptr;
	for (auto &entry : query["operators"]) {
		if (entry["type"] == "COMPARISON_JOIN" &&
		    std::find(region["nodes"].begin(), region["nodes"].end(), entry["node_id"]) != region["nodes"].end())
			join = &entry;
		if (entry["node_id"] == op["node_id"])
			semantic = &entry;
	}
	if (!join || !semantic || (*join)["children"].size() != 2 ||
	    (*semantic)["children"] != Json::array({(*join)["node_id"]}))
		throw BinderException("Native selected-pair identity join has an invalid semantic consumer");
	spec["tree_choices"] =
	    Json::array({{{"region_id", region["region_id"]},
	                  {"root", (*semantic)["node_id"]},
	                  {"actions", Json::array({{{"node_id", (*join)["node_id"]}, {"inputs", (*join)["children"]}},
	                                           {{"node_id", (*semantic)["node_id"]},
	                                            {"inputs", (*semantic)["children"]}}})}}});
	return spec;
}

void Collect(LogicalOperator &op, std::map<int64_t, SemanticLogicalOperator *> &nodes) {
	if (auto sem = Semantic(op)) {
		if (!nodes.emplace(sem->OperatorId(), sem).second)
			throw BinderException("DuckDB duplicated a semantic operator_id");
	}
	for (auto &child : op.children)
		Collect(*child, nodes);
}

void ValidateOrder(const Json &groups, const std::vector<int64_t> &order) {
	// Reordering is legal only inside an advertised group; groups themselves keep
	// their dependency order across relational boundaries.
	size_t offset = 0;
	for (auto &group : groups) {
		std::set<int64_t> remaining;
		for (auto &id : group)
			remaining.insert(Id(id));
		for (size_t i = 0; i < group.size(); ++i) {
			if (offset >= order.size() || !remaining.erase(order[offset++]))
				throw BinderException("PlanSpec order violates semantic dependencies or supported reorder regions");
		}
	}
	if (offset != order.size())
		throw BinderException("PlanSpec contains extra semantic actions");
}

void Reorder(unique_ptr<LogicalOperator> &op, const std::vector<int64_t> &order) {
	// Filter-chain movement preserves input schema and binding identity.
	if (Filter(*op)) {
		std::map<int64_t, unique_ptr<LogicalOperator>> filters;
		auto current = std::move(op);
		while (Filter(*current)) {
			auto next = std::move(current->children[0]);
			current->children.clear();
			const auto id = Semantic(*current)->OperatorId();
			filters.emplace(id, std::move(current));
			current = std::move(next);
		}
		Reorder(current, order);
		for (auto id : order) {
			auto item = filters.find(id);
			if (item == filters.end())
				continue;
			item->second->children.push_back(std::move(current));
			current = std::move(item->second);
		}
		op = std::move(current);
		return;
	}
	for (auto &child : op->children)
		Reorder(child, order);
}
} // namespace

void ApplySemanticDecision(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	// Recheck identities and schemas after DuckDB's native optimizer: a PlanSpec
	// must never be applied to a plan that changed underneath it.
	auto state = input.context.registered_state->GetOrCreate<SemanticPlanningState>("semantic.planning");
	if (state->query.is_null())
		return;
	auto query = std::move(state->query);
	state->query = nullptr;
	if (state->model_plan)
		plan = std::move(state->model_plan);
	std::map<int64_t, SemanticLogicalOperator *> nodes;
	Collect(*plan, nodes);
	plan->ResolveOperatorTypes();
	if (nodes.size() != query["semantic_operators"].size())
		throw BinderException("DuckDB removed or duplicated a semantic operator before PlanSpec application");
	for (auto &entry : query["semantic_operators"]) {
		auto item = nodes.find(Id(entry["operator_id"]));
		if (item == nodes.end())
			throw BinderException("Semantic operator_id lost during native optimization");
		auto &binding = item->second->binding->Cast<SemanticBindData>();
		if (binding.node.semantic.task_description != entry["task"].get<string>())
			throw BinderException("Semantic template changed during native optimization");
		Json types = Json::array();
		for (auto &expression : item->second->expressions)
			types.push_back(expression->return_type.ToString());
		if (types != entry["input_types"])
			throw BinderException("Semantic input schema changed during native optimization");
	}
	Value timeout;
	input.context.TryGetCurrentSetting("semantic_optimizer_timeout_ms", timeout);
	auto ms = timeout.GetValue<int64_t>();
	if (ms < 1 || ms > 600000)
		throw BinderException("semantic_optimizer_timeout_ms must be in 1..600000");
	auto sampling_action = GetSetting(input.context, "semantic_sampling_action");
	Json spec;
	if (sampling_action.empty()) {
		semantic::UnixSocketOptimizerClient client(GetSetting(input.context, "semantic_optimizer_socket"), int(ms));
		spec = client.Optimize(query);
	} else {
		spec = SamplingDecision(query, sampling_action);
	}
	if (spec.is_object() && spec.size() == 1 && spec.contains("error"))
		throw BinderException("Python optimizer rejected QueryIR: %s", spec["error"].get<string>());
	if (spec.contains("tree_choices") && spec.contains("predicate_order"))
		Keys(spec, {"version", "kind", "query_id", "actions", "predicate_order", "tree_choices"});
	else if (spec.contains("tree_choices"))
		Keys(spec, {"version", "kind", "query_id", "actions", "tree_choices"});
	else if (spec.contains("predicate_order"))
		Keys(spec, {"version", "kind", "query_id", "actions", "predicate_order"});
	else
		Keys(spec, {"version", "kind", "query_id", "actions"});
	if (Id(spec["version"]) != 1 || spec["kind"] != "PlanSpec" || spec["query_id"] != query["query_id"] ||
	    !spec["actions"].is_array())
		throw BinderException("PlanSpec version/kind/query identity mismatch");
	std::vector<int64_t> order;
	std::map<int64_t, semantic::SemanticConfig> decisions;
	for (auto &action : spec["actions"]) {
		Keys(action, {"operator_id", "layout", "model_id"});
		auto id = Id(action["operator_id"]);
		if (!nodes.count(id) || decisions.count(id))
			throw BinderException("PlanSpec unknown or duplicate semantic operator_id");
		auto &binding = nodes.at(id)->binding->Cast<SemanticBindData>();
		auto config = binding.node.semantic;
		config.model_id = action["model_id"].get<string>();
		semantic::ModelRegistry::Parse(binding.models).Resolve(config.model_id);
		auto &layout = action["layout"];
		config.layout.block_order.clear();
		if (binding.node.kind == semantic::NodeKind::SEMANTIC_JOIN) {
			Keys(layout, {"block_order", "left_field_order", "right_field_order"});
			config.layout.left_field_order = layout["left_field_order"].get<semantic::FieldOrder>();
			config.layout.right_field_order = layout["right_field_order"].get<semantic::FieldOrder>();
		} else {
			Keys(layout, {"block_order", "field_order"});
			config.layout.field_order = layout["field_order"].get<semantic::FieldOrder>();
		}
		if (!layout["block_order"].is_array())
			throw BinderException("PlanSpec block_order must be an array");
		for (auto &value : layout["block_order"]) {
			bool found = false;
			for (auto block : {semantic::PromptBlock::OPERATOR_SPECIFICATION, semantic::PromptBlock::TASK_DESCRIPTION,
			                   semantic::PromptBlock::DATA_BLOCK}) {
				if (value == semantic::BlockName(block)) {
					config.layout.block_order.push_back(block);
					found = true;
				}
			}
			if (!found)
				throw BinderException("PlanSpec contains unknown prompt block");
		}
		semantic::ValidateSemanticConfig(binding.node.kind, config);
		decisions.emplace(id, config);
		order.push_back(id);
	}
	if (decisions.size() != nodes.size())
		throw BinderException("PlanSpec has missing semantic actions");
	if (!spec.contains("tree_choices"))
		ValidateOrder(query["order_groups"], order);
	auto query_types = plan->types;
	auto query_bindings = plan->GetColumnBindings();
	if (spec.contains("tree_choices")) {
		if (!query.contains("tree_regions"))
			throw BinderException("Query has no model tree regions");
		ApplyModelTrees(plan, query["tree_regions"], spec["tree_choices"]);
	}
	if (spec.contains("predicate_order")) {
		if (!query.contains("predicate_regions"))
			throw BinderException("Query has no movable predicate regions");
		ApplyPredicateOrder(plan, query["predicate_regions"], spec["predicate_order"]);
		for (auto &region : spec["predicate_order"]) {
			std::vector<int64_t> semantic_order, selected_order;
			for (auto &nid : region["nodes"])
				for (auto &node : query["operators"])
					if (node["node_id"] == nid && node.contains("semantic_operator_id"))
						semantic_order.push_back(Id(node["semantic_operator_id"]));
			for (auto id : order)
				if (std::find(semantic_order.begin(), semantic_order.end(), id) != semantic_order.end())
					selected_order.push_back(id);
			if (semantic_order != selected_order)
				throw BinderException("Predicate order disagrees with semantic stage order");
		}
	}
	if (spec.contains("tree_choices")) {
		if (ModelExecutionOrder(*plan) != order)
			throw BinderException("Model tree disagrees with semantic execution order");
	} else {
		Json current_groups = Json::array();
		if (!query.contains("tree_regions"))
			Regions(*plan, current_groups);
		else
			for (auto id : ModelExecutionOrder(*plan))
				current_groups.push_back(Json::array({id}));
		ValidateOrder(current_groups, order);
	}
	plan->ResolveOperatorTypes();
	auto result_types = plan->types;
	auto result_bindings = plan->GetColumnBindings();
	// Every stage needs the full initial semantic configuration catalog so it can
	// update PENDING operators before their physical Execute is first called.
	std::vector<semantic::SemanticOperator> runtime_nodes;
	for (auto id : order) {
		auto node = nodes.at(id)->binding->Cast<SemanticBindData>().node;
		node.semantic = decisions.at(id);
		runtime_nodes.push_back(std::move(node));
	}
	const auto decision_json = spec.dump();
	const auto query_ir_json = query.dump();
	for (auto &entry : decisions) {
		// PlanSpec fixes topology and per-row dependencies. DuckDB pipelines may
		// interleave requests from consecutive operators across DataChunks.
		auto &binding = nodes.at(entry.first)->binding->Cast<SemanticBindData>();
		binding.node.semantic = entry.second;
		binding.execution_context.operators = runtime_nodes;
		binding.decision_json = decision_json;
		binding.query_ir_json = query_ir_json;
	}
	if (!spec.contains("tree_choices"))
		Reorder(plan, order);
	if (query.contains("tree_regions"))
		FuseModelJoins(plan);
	plan->ResolveOperatorTypes();
	if (result_types != plan->types || result_bindings != plan->GetColumnBindings())
		throw BinderException("PlanSpec application changed query output schema/bindings");
	if (query_types != plan->types || query_bindings != plan->GetColumnBindings())
		throw BinderException("Model tree changed original SQL output schema/bindings");
	CaptureAggregateInput(plan, GetSetting(input.context, "semantic_capture_aggregate_input"));
	ColumnBindingResolver::Verify(*plan);
}
}
}
