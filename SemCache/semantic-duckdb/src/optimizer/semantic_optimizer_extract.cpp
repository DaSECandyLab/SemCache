// Convert DuckDB's logical plan into a stable optimizer-facing QueryIR, validate
// the returned PlanSpec, then apply only realizable semantic physical choices.
#include "optimizer/semantic_optimizer.hpp"
#include "optimizer/semantic_optimizer_internal.hpp"
#include "optimizer/semantic_dependencies.hpp"
#include "optimizer/optimizer_client.hpp"
#include "llm/model_registry.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "runtime/result_capture.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "operators/plan/semantic_rewriter.hpp"
#include "operators/plan/rewrite/semantic_predicates.hpp"
#include "operators/plan/semantic_plan_tree.hpp"
#include "prompt/prompt_renderer.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/string_stats.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/common/types/uuid.hpp"
#include <algorithm>
#include <limits>

namespace duckdb {
namespace semantic_extension {
namespace optimizer_detail {

SemanticLogicalOperator *Semantic(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR)
		return nullptr;
	if (op.GetName() != "SEMANTIC_FILTER" && op.GetName() != "SEMANTIC_PROJECTION" && op.GetName() != "SEMANTIC_JOIN")
		return nullptr;
	return &static_cast<SemanticLogicalOperator &>(op);
}
bool Filter(LogicalOperator &op) {
	return Semantic(op) && op.GetName() == "SEMANTIC_FILTER";
}
// Adjacent Filter regions reorder predicates on one input; other semantic stages form singleton regions.
void Regions(LogicalOperator &op, Json &groups) {
	if (Filter(op)) {
		Json group = Json::array();
		auto current = &op;
		while (Filter(*current)) {
			group.push_back(Semantic(*current)->OperatorId());
			current = current->children[0].get();
		}
		Regions(*current, groups);
		std::reverse(group.begin(), group.end());
		groups.push_back(group);
		return;
	}
	idx_t branches = 0;
	for (auto &child : op.children) {
		auto before = groups.size();
		Regions(*child, groups);
		branches += groups.size() != before;
	}
	if (branches > 1)
		throw BinderException(
		    "QueryIR v1 does not schedule independent semantic branches; use one dependent semantic chain");
	if (auto sem = Semantic(op))
		groups.push_back(Json::array({sem->OperatorId()}));
}

} // namespace optimizer_detail

using namespace optimizer_detail;
namespace {
using Json = semantic::llm_detail::Json;

struct Markers : LogicalOperatorVisitor {
	bool found = false;
	void VisitExpression(unique_ptr<Expression> *expr) override {
		// VisitExpression<T> stops at the first T, so struct_extract/project
		// wrappers hide nested semantic functions. Inspect every expression.
		std::function<void(const Expression &)> visit = [&](const Expression &expression) {
			if (expression.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				auto &fn = expression.Cast<BoundFunctionExpression>();
				if (fn.function.name == "semantic_filter" || fn.function.name == "semantic_project" ||
				    fn.function.name == "semantic_join") {
					found = true;
				}
			}
			ExpressionIterator::EnumerateChildren(expression, visit);
		};
		visit(**expr);
	}
};

Json Columns(LogicalOperator &op) {
	Json result = Json::array();
	auto bindings = op.GetColumnBindings();
	for (idx_t i = 0; i < bindings.size(); ++i)
		result.push_back(
		    {{"binding", {bindings[i].table_index, bindings[i].column_index}}, {"type", op.types.at(i).ToString()}});
	return result;
}

void Extract(LogicalOperator &op, ClientContext &context, Json &query, idx_t &next) {
	// IDs describe this serialized tree only. Semantic operator IDs remain the
	// stable handles used by optimizer actions and execution telemetry.
	const auto id = next++;
	Json children = Json::array();
	for (auto &child : op.children) {
		children.push_back(next);
		Extract(*child, context, query, next);
	}
	Json expressions = Json::array(), required = Json::array();
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *pointer) {
		auto &expression = *pointer;
		expressions.push_back(expression->ToString());
		std::function<void(const Expression &)> dependencies = [&](const Expression &expr) {
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF ||
			    expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY)
				throw BinderException("QueryIR requires resolved input field dependencies");
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &ref = expr.Cast<BoundColumnRefExpression>();
				if (ref.depth)
					throw BinderException("QueryIR v1 does not support correlated semantic queries");
				required.push_back({ref.binding.table_index, ref.binding.column_index});
			}
			ExpressionIterator::EnumerateChildren(expr, dependencies);
		};
		dependencies(*expression);
	});
	Json entry {{"node_id", id},
	            {"type", op.GetName()},
	            {"children", children},
	            {"expressions", expressions},
	            {"required_columns", required},
	            {"output_schema", Columns(op)},
	            {"estimated_rows", nullptr}};
	entry["input_schema"] = Json::array();
	for (auto &child : op.children)
		entry["input_schema"].push_back(Columns(*child));
	entry["parameters"] = Json::object();
	for (auto &parameter : op.ParamsToString())
		entry["parameters"][parameter.first] = parameter.second;
	entry["estimated_rows"] = op.EstimateCardinality(context);
	entry["statistics"] = {{"cardinality_basis", "DuckDB logical EstimateCardinality"}};
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE)
		entry["cte_index"] = op.Cast<LogicalMaterializedCTE>().table_index;
	else if (op.type == LogicalOperatorType::LOGICAL_CTE_REF)
		entry["cte_index"] = op.Cast<LogicalCTERef>().cte_index;
	SamplingSQL(op, entry);
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		Json relation {{"node_id", id},
		               {"relation_id", get.table_index},
		               {"function", get.function.name},
		               {"table", nullptr},
		               {"schema", nullptr},
		               {"catalog", nullptr},
		               {"estimated_rows", nullptr},
		               {"max_rows", nullptr},
		               {"columns", Json::array()}};
		if (get.function.cardinality) {
			auto stats = get.function.cardinality(context, get.bind_data.get());
			if (stats && stats->has_estimated_cardinality)
				relation["estimated_rows"] = stats->estimated_cardinality;
			if (stats && stats->has_max_cardinality)
				relation["max_rows"] = stats->max_cardinality;
		}
		if (auto table = get.GetTable()) {
			relation["table"] = table->name;
			relation["schema"] = table->schema.name;
			relation["catalog"] = table->catalog.GetName();
			vector<string> fields {"row_number() OVER ()::VARCHAR __sample_id"};
			auto bindings = get.GetColumnBindings();
			for (idx_t slot = 0; slot < bindings.size(); ++slot) {
				auto &column = get.GetColumnIds()[slot];
				if (column.IsRowIdColumn() || !column.IsVirtualColumn())
					fields.push_back(Name(get.GetColumnName(column)) + " AS " + Name(FieldName(bindings[slot])));
			}
			relation["source_sql"] = "SELECT " + StringUtil::Join(fields, ",") + " FROM " + Name(table->schema.name) +
			                         "." + Name(table->name);
		}
		const auto &column_ids = get.GetColumnIds();
		for (idx_t slot = 0; slot < column_ids.size(); ++slot) {
			const auto &column_index = column_ids[slot];
			const auto column = column_index.GetPrimaryIndex();
			if (column_index.IsVirtualColumn() && !column_index.IsRowIdColumn())
				continue;
			Json field {{"name", get.GetColumnName(column_index)},
			            {"binding", Json::array({get.table_index, slot})},
			            {"type", get.GetColumnType(column_index).ToString()},
			            {"can_have_null", nullptr},
			            {"null_fraction", nullptr},
			            {"distinct_count", nullptr},
			            {"min", nullptr},
			            {"max", nullptr},
			            {"summary", nullptr}};
			if (!column_index.IsVirtualColumn() && (get.function.statistics || get.function.statistics_extended)) {
				unique_ptr<BaseStatistics> stats;
				if (get.function.statistics_extended) {
					TableFunctionGetStatisticsInput input(get.bind_data.get(), column_index);
					stats = get.function.statistics_extended(context, input);
				} else {
					stats = get.function.statistics(context, get.bind_data.get(), column);
				}
				if (stats) {
					field["can_have_null"] = stats->CanHaveNull();
					field["summary"] = stats->ToString();
					const auto distinct = stats->GetDistinctCount();
					if (distinct || relation["max_rows"] == 0)
						field["distinct_count"] = distinct;
					if (relation["max_rows"] != 0) {
						if (!stats->CanHaveNull())
							field["null_fraction"] = 0.0;
						else if (!stats->CanHaveNoNull())
							field["null_fraction"] = 1.0;
					}
					if (stats->GetStatsType() == StatisticsType::NUMERIC_STATS) {
						if (NumericStats::HasMin(*stats))
							field["min"] = NumericStats::Min(*stats).ToString();
						if (NumericStats::HasMax(*stats))
							field["max"] = NumericStats::Max(*stats).ToString();
					}
					if (stats->GetStatsType() == StatisticsType::STRING_STATS &&
					    StringStats::HasMaxStringLength(*stats))
						field["max_string_bytes"] = StringStats::MaxStringLength(*stats);
				}
			}
			relation["columns"].push_back(field);
		}
		entry["estimated_rows"] = relation["estimated_rows"];
		query["relations"].push_back(relation);
	}
	if (auto sem = Semantic(op)) {
		auto &binding = sem->binding->Cast<SemanticBindData>();
		binding.execution_context.query_id = query["query_id"].get<string>();
		auto &config = binding.node.semantic;
		config.operator_id = int64_t(query["semantic_operators"].size() + 1);
		Json schema = Json::array(), inputs = Json::array(), fields = Json::array();
		for (auto &field : config.output_schema.fields)
			schema.push_back({{"name", field.name}, {"type", semantic::TypeName(field)}});
		for (auto &expr : op.expressions) {
			inputs.push_back(expr->return_type.ToString());
			// SQL inputs are bound struct_pack expressions. Export each original
			// field, not the enclosing STRUCT type, for the shared field encoder.
			auto &pack = expr->Cast<BoundFunctionExpression>();
			for (auto &field : pack.children) {
				Json entry {
				    {"name", field->alias}, {"type", field->return_type.ToString()}, {"expression", field->ToString()}};
				entry["sample_expression"] = SampleExpression(*field);
				if (field->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &ref = field->Cast<BoundColumnRefExpression>();
					entry["binding"] = {ref.binding.table_index, ref.binding.column_index};
					entry["value_key"] = FieldName(ref.binding);
				} else
					entry["value_key"] = "__semantic_input:" + std::to_string(sem->OperatorId()) + ":" + field->alias;
				fields.push_back(std::move(entry));
			}
		}
		entry["semantic_operator_id"] = sem->OperatorId();
		Json pair_columns;
		if (op.GetName() == "SEMANTIC_JOIN" && op.children.size() == 1) {
			pair_columns = Json::array();
			auto &source = ModelPairSource(*op.children[0]);
			for (auto &side : source.children) {
				Json columns = Json::array();
				for (auto &binding : side->GetColumnBindings())
					columns.push_back(FieldName(binding));
				pair_columns.push_back(std::move(columns));
			}
			entry["pair_columns"] = pair_columns;
		}
		Json candidate {
		    {"operator_id", sem->OperatorId()},
		    {"node_id", id},
		    {"kind", op.GetName()},
		    {"task", config.task_description},
		    {"operator_specification", semantic::OperatorSpecification(binding.node.kind)},
		    {"input_types", inputs},
		    {"input_field_schema", fields},
		    {"output_schema", schema},
		    {"required_columns", required},
		    {"input_fields", config.input_fields},
		    {"left_input_fields", config.left_input_fields},
		    {"right_input_fields", config.right_input_fields},
		    {"models", semantic::ModelRegistry::Parse(binding.models).KnownModelIds()},
		    {"layout_domain",
		     {{"block_orders",
		       {{"OPERATOR_SPECIFICATION", "TASK_DESCRIPTION", "DATA_BLOCK"},
		        {"OPERATOR_SPECIFICATION", "DATA_BLOCK", "TASK_DESCRIPTION"},
		        {"DATA_BLOCK", "OPERATOR_SPECIFICATION", "TASK_DESCRIPTION"}}},
		      {"field_permutations", true},
		      {"swap_join_sides", false}}},
		    {"default_layout", semantic::llm_detail::LayoutJSON(config.layout, config.layout.field_order.empty())},
		    {"request_granularity", "one_row_or_pair"},
		    {"semantic_statistics", nullptr},
		    {"memory_constraints", nullptr}};
		if (!pair_columns.is_null())
			candidate["pair_columns"] = pair_columns;
		if (entry.contains("sampling_sql"))
			candidate["sampling_sql"] = entry["sampling_sql"];
		query["semantic_operators"].push_back(candidate);
	}
	query["operators"].push_back(entry);
}

} // namespace

void ExtractSemanticQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto state = input.context.registered_state->GetOrCreate<SemanticPlanningState>("semantic.planning");
	state->query = nullptr;
	state->model_plan.reset();
	Markers markers;
	markers.VisitOperator(*plan);
	if (!markers.found)
		return;
	// Canonicalize before native predicate movement/CSE. Structural regions keep
	// the bound tree opaque until the model's choices have been validated.
	RewriteSemanticOperators(input, plan);
	NormalizeModelSPJ(plan);
	PreserveModelPredicates(plan);
	plan->ResolveOperatorTypes();
	const auto runtime_options = GetSetting(input.context, "semantic_runtime");
	const bool chunk_adaptation =
	    !runtime_options.empty() && semantic::llm_detail::ParseJSON(runtime_options).value("version", 1) == 2;
	Json query {
	    {"version", 1},
	    {"kind", "QueryIR"},
	    {"execution_mode", "data_chunk_pipeline"},
	    {"semantic_join_mode", "stream_first_left_then_replay"},
	    {"query_id", UUID::ToString(UUID::GenerateRandomUUID())},
	    {"duckdb_commit", "08e34c447bae34eaee3723cac61f2878b6bdf787"},
	    {"relations", Json::array()},
	    {"operators", Json::array()},
	    {"semantic_operators", Json::array()},
	    {"global_system_prefix", semantic::GlobalSystemPrefix()},
	    {"order_groups", Json::array()},
	    {"dependencies", Json::array()},
	    {"environment",
	     {{"duckdb_threads", std::stoll(GetSetting(input.context, "threads"))},
	      {"semantic_response_order", GetSetting(input.context, "semantic_sampling_action").empty() &&
	                                          (runtime_options.empty() || chunk_adaptation)
	                                      ? "ready_first"
	                                      : "input_order"},
	      {"scheduler_concurrency", GetSetting(input.context, "semantic_unrestricted") == "true"
	                                    ? Json(std::stoll(GetSetting(input.context, "semantic_parallelism")))
	                                    : Json(nullptr)},
	      {"scheduler_policy", GetSetting(input.context, "semantic_unrestricted") == "true"
	                               ? Json("unrestricted_sampling")
	                               : Json(GetSetting(input.context, "semantic_enforce_model_operator_order") == "true"
	                                          ? "ordered_model_lanes"
	                                          : "fifo_model_lanes")},
	      {"scheduler_enforce_model_operator_order",
	       GetSetting(input.context, "semantic_unrestricted") != "true" &&
	           GetSetting(input.context, "semantic_enforce_model_operator_order") == "true"},
	      {"scheduler_max_inflight_per_model",
	       GetSetting(input.context, "semantic_unrestricted") == "true"
	           ? Json(nullptr)
	           : Json(std::stoll(GetSetting(input.context, "semantic_max_concurrent_requests_per_model")))},
	      {"available_memory", nullptr}}}};
	idx_t next = 0;
	Extract(*plan, input.context, query, next);
	auto trees = ModelTreeRegions(*plan, query);
	if (!trees.empty())
		query["tree_regions"] = trees;
	auto regions = PredicateRegions(*plan);
	if (!trees.empty()) {
		std::set<idx_t> members;
		for (auto &tree : trees)
			for (auto &id : tree["nodes"])
				members.insert(Id(id));
		regions.erase(std::remove_if(regions.begin(), regions.end(),
		                             [&](const Json &r) {
			                             for (auto &id : r["nodes"])
				                             if (members.count(Id(id)))
					                             return true;
			                             return false;
		                             }),
		              regions.end());
	}
	if (!regions.empty())
		query["predicate_regions"] = std::move(regions);
	if (trees.empty())
		Regions(*plan, query["order_groups"]);
	else
		for (auto id : ModelExecutionOrder(*plan))
			query["order_groups"].push_back(Json::array({id}));
	query["dependencies"] = SemanticDependencies(query);
	state->query = std::move(query);
	if (!trees.empty() || !GetSetting(input.context, "semantic_capture_aggregate_input").empty()) {
		// Preserve the complete input projection for recording before native
		// aggregate pruning removes row identity and other non-group fields.
		auto frozen = FreezeModelPlan(*plan);
		state->model_plan = std::move(plan);
		plan = std::move(frozen);
	}
}

}
}
