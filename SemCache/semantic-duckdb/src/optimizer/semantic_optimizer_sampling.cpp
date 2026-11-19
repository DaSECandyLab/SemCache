#include "optimizer/semantic_optimizer_internal.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "operators/plan/rewrite/semantic_predicates.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include <algorithm>
#include <functional>

namespace duckdb {
namespace semantic_extension {
namespace optimizer_detail {

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

namespace {
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

} // namespace

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

} // namespace optimizer_detail
} // namespace semantic_extension
} // namespace duckdb
