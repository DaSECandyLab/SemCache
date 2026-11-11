// Convert DuckDB's logical plan into a stable optimizer-facing QueryIR, validate
// the returned PlanSpec, then apply only realizable semantic physical choices.
#include "optimizer/semantic_optimizer.hpp"
#include "optimizer/semantic_dependencies.hpp"
#include "optimizer/optimizer_client.hpp"
#include "llm/model_registry.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "runtime/result_capture.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "operators/plan/semantic_rewriter.hpp"
#include "operators/plan/semantic_predicates.hpp"
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
namespace {
using Json = semantic::llm_detail::Json;

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

string FieldName(const ColumnBinding &binding) {
	return Json::array({binding.table_index, binding.column_index}).dump();
}
string Name(const string &value) {
	return KeywordHelper::WriteOptionallyQuoted(value, '\"');
}
string SampleExpression(const Expression &expression) {
	auto copy = expression.Copy();
	std::function<void(Expression &)> rewrite = [&](Expression &expr) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &ref = expr.Cast<BoundColumnRefExpression>();
			ref.alias = Name(FieldName(ref.binding));
		} else {
			// Bound casts render their child with GetName(), which prefers an
			// alias to the expression SQL. Projection lineage can leave an alias
			// such as has_evidence on a struct_extract child; that name is not a
			// column of the sampling boundary. Discard display aliases on this
			// copy so every nested expression renders its bound input instead.
			expr.alias.clear();
		}
		ExpressionIterator::EnumerateChildren(expr, rewrite);
	};
	rewrite(*copy);
	return copy->ToString();
}
bool SafeSamplingCast(const BoundCastExpression &cast) {
	auto source = cast.child->return_type;
	auto target = cast.return_type;
	if (source == target)
		return true;
	// A bound constant can be checked once without relaxing safety for casts
	// of arbitrary input rows (including overflowing integer/decimal values).
	if (cast.child->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		Value converted;
		string error;
		return cast.child->Cast<BoundConstantExpression>().value.DefaultTryCastAs(target, converted, &error);
	}
	auto width = [](LogicalTypeId type, bool &is_unsigned) {
		is_unsigned = false;
		switch (type) {
		case LogicalTypeId::TINYINT:
			return 8;
		case LogicalTypeId::SMALLINT:
			return 16;
		case LogicalTypeId::INTEGER:
			return 32;
		case LogicalTypeId::BIGINT:
			return 64;
		case LogicalTypeId::HUGEINT:
			return 128;
		case LogicalTypeId::UTINYINT:
			is_unsigned = true;
			return 8;
		case LogicalTypeId::USMALLINT:
			is_unsigned = true;
			return 16;
		case LogicalTypeId::UINTEGER:
			is_unsigned = true;
			return 32;
		case LogicalTypeId::UBIGINT:
			is_unsigned = true;
			return 64;
		case LogicalTypeId::UHUGEINT:
			is_unsigned = true;
			return 128;
		default:
			return 0;
		}
	};
	bool source_unsigned, target_unsigned;
	auto source_width = width(source.id(), source_unsigned);
	auto target_width = width(target.id(), target_unsigned);
	// Integer-to-double conversion can round, but cannot throw. The sampled
	// predicate uses the same bound cast, so rounding does not alter its SQL.
	if ((source_width || source.id() == LogicalTypeId::DECIMAL) && target.id() == LogicalTypeId::DOUBLE)
		return true;
	if (source_width && target_width)
		return source_unsigned == target_unsigned ? target_width >= source_width
		                                          : source_unsigned && !target_unsigned && target_width > source_width;
	return source.id() == LogicalTypeId::FLOAT && target.id() == LogicalTypeId::DOUBLE;
}
// Certify only explicitly understood, deterministic, nonthrowing bound ASTs.
// Check each node and recurse: DuckDB conservatively marks every ABS/modulo
// overload as fallible, including the nonthrowing floating-point overloads.
bool SamplingRowLocal(const Expression &expression, const vector<ColumnBinding> &bindings, vector<string> &required) {
	if (expression.IsVolatile() || !expression.IsConsistent() || expression.IsAggregate() || expression.IsWindow())
		return false;
	switch (expression.GetExpressionClass()) {
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_COMPARISON:
	case ExpressionClass::BOUND_CONJUNCTION:
		break;
	case ExpressionClass::BOUND_CAST:
		if (!SafeSamplingCast(expression.Cast<BoundCastExpression>()))
			return false;
		break;
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &ref = expression.Cast<BoundColumnRefExpression>();
		if (ref.depth || std::find(bindings.begin(), bindings.end(), ref.binding) == bindings.end())
			return false;
		auto name = FieldName(ref.binding);
		if (std::find(required.begin(), required.end(), name) == required.end())
			required.push_back(name);
		break;
	}
	case ExpressionClass::BOUND_OPERATOR:
		if (expression.type != ExpressionType::OPERATOR_NOT && expression.type != ExpressionType::OPERATOR_IS_NULL &&
		    expression.type != ExpressionType::OPERATOR_IS_NOT_NULL &&
		    expression.type != ExpressionType::OPERATOR_COALESCE)
			return false;
		break;
	case ExpressionClass::BOUND_FUNCTION: {
		auto &fn = expression.Cast<BoundFunctionExpression>();
		auto type = expression.return_type.id();
		if (type != LogicalTypeId::FLOAT && type != LogicalTypeId::DOUBLE)
			return false;
		if (!((fn.function.name == "abs" && fn.children.size() == 1) ||
		      (fn.function.name == "%" && fn.children.size() == 2)))
			return false;
		for (auto &child : fn.children)
			if (child->return_type != expression.return_type)
				return false;
		break;
	}
	default:
		return false;
	}
	bool safe = true;
	ExpressionIterator::EnumerateChildren(
	    expression, [&](const Expression &child) { safe &= SamplingRowLocal(child, bindings, required); });
	return safe;
}
// Trusted stage templates are generated from bound expressions. The Python
// sampler executes these over bounded temporary inputs, never over a complete
// candidate plan and never by parsing human-readable EXPLAIN expressions.
void SamplingSQL(LogicalOperator &op, Json &entry) {
	auto bindings = op.GetColumnBindings();
	string sql;
	if (op.type == LogicalOperatorType::LOGICAL_FILTER || IsModelPredicate(op)) {
		vector<string> predicates;
		for (auto &expr : op.expressions)
			predicates.push_back("(" + SampleExpression(*expr) + ")");
		sql = "SELECT * FROM $INPUT WHERE " + StringUtil::Join(predicates, " AND ");
	} else if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		vector<string> fields {"__sample_id"};
		for (idx_t i = 0; i < op.expressions.size(); ++i)
			fields.push_back(SampleExpression(*op.expressions[i]) + " AS " + Name(FieldName(bindings[i])));
		sql = "SELECT " + StringUtil::Join(fields, ",") + " FROM $INPUT";
	} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = op.Cast<LogicalAnyJoin>();
		if (join.join_type != JoinType::INNER)
			return;
		entry["join_type"] = "INNER";
		sql = "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,"
		      "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l JOIN $RIGHT r ON " +
		      SampleExpression(*join.condition);
	} else if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
	           op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		string condition = "TRUE";
		if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &join = op.Cast<LogicalComparisonJoin>();
			if (join.join_type != JoinType::INNER && join.join_type != JoinType::SEMI &&
			    join.join_type != JoinType::LEFT)
				return;
			entry["join_type"] = JoinTypeToString(join.join_type);
			vector<string> predicates;
			for (auto &c : join.conditions)
				predicates.push_back("(" + SampleExpression(*c.left) + " " + ExpressionTypeToOperator(c.comparison) +
				                     " " + SampleExpression(*c.right) + ")");
			condition = StringUtil::Join(predicates, " AND ");
			if (join.join_type == JoinType::SEMI) {
				// Existence preserves each left tuple once, even when several
				// right tuples match. Its identity/lineage belongs only to the left.
				entry["sampling_sql"] = "SELECT l.* FROM $LEFT l SEMI JOIN $RIGHT r ON " + condition;
				return;
			}
			if (join.join_type == JoinType::LEFT) {
				// Request-count planning only EXPLAINs this bound relation. Keep
				// the null-extending join intact instead of estimating an inner join.
				entry["sampling_sql"] =
				    "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,"
				    "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l LEFT JOIN $RIGHT r ON " +
				    condition;
				return;
			}
		}
		sql = "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,"
		      "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l JOIN $RIGHT r ON " +
		      condition;
	} else if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
		if (op.Cast<LogicalDistinct>().distinct_type != DistinctType::DISTINCT)
			return;
		vector<string> fields;
		for (auto &binding : bindings)
			fields.push_back(Name(FieldName(binding)));
		// The transport identity is not part of SQL DISTINCT's equality key.
		sql = "SELECT row_number() OVER ()::VARCHAR __sample_id,* FROM (SELECT DISTINCT " +
		      StringUtil::Join(fields, ",") + " FROM $INPUT)";
	} else if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op.Cast<LogicalAggregate>();
		if (!agg.grouping_functions.empty() || agg.grouping_sets.size() > 1)
			return;
		vector<string> fields, groups;
		for (auto &expr : agg.groups) {
			groups.push_back(SampleExpression(*expr));
			fields.push_back(groups.back() + " AS " + Name(FieldName(bindings[fields.size()])));
		}
		for (auto &expr : agg.expressions)
			fields.push_back(SampleExpression(*expr) + " AS " + Name(FieldName(bindings[fields.size()])));
		sql = "SELECT row_number() OVER ()::VARCHAR __sample_id,* FROM (SELECT " + StringUtil::Join(fields, ",") +
		      " FROM $INPUT" + (groups.empty() ? "" : " GROUP BY " + StringUtil::Join(groups, ",")) + ")";
		entry["scalar_aggregate"] = groups.empty();
	} else if (op.type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		vector<string> orders;
		for (auto &order : op.Cast<LogicalOrder>().orders)
			orders.push_back(SampleExpression(*order.expression) + " " + order.GetOrderModifier());
		sql = "SELECT * FROM $INPUT ORDER BY " + StringUtil::Join(orders, ",");
	} else if (op.type == LogicalOperatorType::LOGICAL_LIMIT) {
		auto &limit = op.Cast<LogicalLimit>();
		if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE ||
		    (limit.offset_val.Type() != LimitNodeType::CONSTANT_VALUE &&
		     limit.offset_val.Type() != LimitNodeType::UNSET))
			return;
		entry["limit"] = limit.limit_val.GetConstantValue();
		entry["offset"] = limit.offset_val.Type() == LimitNodeType::UNSET ? 0 : limit.offset_val.GetConstantValue();
		sql = "SELECT * FROM $INPUT LIMIT " + std::to_string(entry["limit"].get<idx_t>()) + " OFFSET " +
		      std::to_string(entry["offset"].get<idx_t>());
	} else if (auto sem = Semantic(op)) {
		auto &binding = sem->binding->Cast<SemanticBindData>();
		vector<string> args {KeywordHelper::WriteQuoted(binding.node.semantic.task_description, '\'')};
		for (auto &expr : op.expressions)
			for (auto &field : expr->Cast<BoundFunctionExpression>().children)
				args.push_back(SampleExpression(*field));
		if (op.GetName() == "SEMANTIC_PROJECTION") {
			args.push_back("NULL::" + sem->generated_type.ToString());
			sql = "SELECT *,semantic_project(" + StringUtil::Join(args, ",") + ") AS " +
			      Name(FieldName(bindings.back())) + " FROM $INPUT";
		} else {
			string call = (op.GetName() == "SEMANTIC_FILTER" ? "semantic_filter(" : "semantic_join(") +
			              StringUtil::Join(args, ",") + ")";
			sql = (op.children.size() == 2 || op.GetName() == "SEMANTIC_JOIN")
			          ? "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,"
			            "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l JOIN $RIGHT r ON " +
			                call
			          : "SELECT * FROM $INPUT WHERE " + call;
		}
	}
	if (!sql.empty())
		entry["sampling_sql"] = sql;
	if ((op.type == LogicalOperatorType::LOGICAL_FILTER || IsModelPredicate(op)) && op.children.size() == 1 &&
	    !op.expressions.empty()) {
		vector<string> required;
		bool safe = true;
		auto local = op.children[0]->GetColumnBindings();
		for (auto &expression : op.expressions)
			safe &= SamplingRowLocal(*expression, local, required);
		if (safe) {
			entry["sampling_row_local"] = true;
			entry["sampling_row_local_sql"] = sql;
			entry["sampling_required_fields"] = required;
		}
	}
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

void Keys(const Json &object, std::initializer_list<const char *> keys) {
	if (!object.is_object() || object.size() != keys.size())
		throw BinderException("PlanSpec has missing or unknown keys");
	for (auto key : keys)
		if (!object.contains(key))
			throw BinderException("PlanSpec missing key %s", key);
}
int64_t Id(const Json &value) {
	if (!value.is_number_integer() ||
	    (value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
	    value.get<int64_t>() < 0)
		throw BinderException("PlanSpec IDs/version must be nonnegative integers");
	return value.get<int64_t>();
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
}

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
	    {"semantic_join_mode", chunk_adaptation ? "duckdb_candidate_pair_chunks" : "stream_first_left_then_replay"},
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
