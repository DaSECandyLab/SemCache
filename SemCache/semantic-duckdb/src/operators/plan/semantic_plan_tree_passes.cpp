#include "operators/plan/semantic_plan_tree.hpp"
#include "operators/plan/semantic_plan_tree_internal.hpp"
#include "operators/plan/rewrite/semantic_projection_lineage.hpp"
#include "operators/plan/rewrite/semantic_predicates.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include <functional>
#include <map>
#include <utility>

namespace duckdb {
namespace semantic_extension {
namespace plan_tree_detail {
void ExecutionOrder(LogicalOperator &op, std::vector<int64_t> &order) {
	if (op.children.size() == 2 &&
	    (op.GetName().find("JOIN") != string::npos || op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT)) {
		ExecutionOrder(*op.children[1], order);
		ExecutionOrder(*op.children[0], order);
	} else
		for (auto &child : op.children)
			ExecutionOrder(*child, order);
	if (Semantic(op))
		order.push_back(static_cast<SemanticLogicalOperator &>(op).OperatorId());
}

} // namespace plan_tree_detail

using namespace plan_tree_detail;
namespace {
void RemoveAliases(unique_ptr<LogicalOperator> &plan, bool parent_movable,
                   std::map<std::pair<idx_t, idx_t>, ColumnBinding> &aliases) {
	bool movable = Movable(*plan);
	for (auto &child : plan->children)
		RemoveAliases(child, movable, aliases);
	if (!parent_movable || plan->type != LogicalOperatorType::LOGICAL_PROJECTION)
		return;
	for (auto &expr : plan->expressions)
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
		    expr->Cast<BoundColumnRefExpression>().depth)
			return;
	auto fields = plan->GetColumnBindings();
	for (idx_t i = 0; i < fields.size(); ++i)
		aliases[{fields[i].table_index, fields[i].column_index}] =
		    plan->expressions[i]->Cast<BoundColumnRefExpression>().binding;
	plan = std::move(plan->children[0]);
}
void RebindAliases(LogicalOperator &op, const std::map<std::pair<idx_t, idx_t>, ColumnBinding> &aliases) {
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *expr) {
		std::function<void(Expression &)> visit = [&](Expression &value) {
			if (value.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &ref = value.Cast<BoundColumnRefExpression>();
				auto found = aliases.find({ref.binding.table_index, ref.binding.column_index});
				while (found != aliases.end()) {
					ref.binding = found->second;
					found = aliases.find({ref.binding.table_index, ref.binding.column_index});
				}
			}
			ExpressionIterator::EnumerateChildren(value, visit);
		};
		visit(**expr);
	});
	for (auto &child : op.children)
		RebindAliases(*child, aliases);
}
void NormalizeAtoms(unique_ptr<LogicalOperator> &plan) {
	for (auto &child : plan->children)
		NormalizeAtoms(child);
	if (plan->type == LogicalOperatorType::LOGICAL_FILTER && plan->expressions.empty() &&
	    plan->Cast<LogicalFilter>().projection_map.empty()) {
		// Marker lowering leaves identity filters. They add meaningless search
		// permutations and execution stages; preserve filters that project columns.
		plan = std::move(plan->children[0]);
		return;
	}
	// A relational join and all of its ON conditions form one search action.
	// Expanding it into a cross product plus independent filters exposes an
	// unfiltered Cartesian intermediate to ranking and allows conditions to lag.
	if (plan->type == LogicalOperatorType::LOGICAL_FILTER && Movable(*plan) && plan->expressions.size() > 1) {
		auto filter = std::move(plan);
		plan = std::move(filter->children[0]);
		for (auto &expr : filter->expressions) {
			auto atom = make_uniq<LogicalFilter>(std::move(expr));
			atom->children.push_back(std::move(plan));
			plan = std::move(atom);
		}
	}
}
void PushRelationalFilters(unique_ptr<LogicalOperator> &plan) {
	for (auto &child : plan->children)
		PushRelationalFilters(child);
	if (plan->type != LogicalOperatorType::LOGICAL_FILTER || !Movable(*plan) || plan->children.size() != 1)
		return;
	auto &child = *plan->children[0];
	// A matched-pair semantic join must keep its relational join intact, but
	// must not trap independent WHERE predicates above that boundary. Both
	// semantic predicates preserve their input bindings and only remove rows.
	if (!Semantic(child) || child.children.size() != 1 ||
	    (child.GetName() != "SEMANTIC_FILTER" && child.GetName() != "SEMANTIC_JOIN"))
		return;
	auto fields = child.children[0]->GetColumnBindings();
	for (auto &expr : plan->expressions)
		if (expr->IsVolatile() || !Available(*expr, fields))
			return;
	auto filter = std::move(plan);
	auto predicate = std::move(filter->children[0]);
	filter->children[0] = std::move(predicate->children[0]);
	predicate->children[0] = std::move(filter);
	plan = std::move(predicate);
	PushRelationalFilters(plan->children[0]);
}

} // namespace

void NormalizeModelSPJ(unique_ptr<LogicalOperator> &plan) {
	std::map<std::pair<idx_t, idx_t>, ColumnBinding> aliases;
	RemoveAliases(plan, false, aliases);
	RebindAliases(*plan, aliases);
	NormalizeAtoms(plan);
	PushRelationalFilters(plan);
	LiftModelUnaryProjections(plan);
}
void FuseModelJoins(unique_ptr<LogicalOperator> &plan) {
	for (auto &child : plan->children)
		FuseModelJoins(child);
	if ((!IsModelPredicate(*plan) && plan->type != LogicalOperatorType::LOGICAL_FILTER) ||
	    plan->expressions.size() != 1 ||
	    plan->expressions[0]->GetExpressionClass() != ExpressionClass::BOUND_COMPARISON)
		return;
	auto &child = *plan->children[0];
	if (child.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT &&
	    child.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
		return;
	if (child.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	    child.Cast<LogicalComparisonJoin>().join_type != JoinType::INNER)
		return;
	auto &expr = plan->expressions[0]->Cast<BoundComparisonExpression>();
	if (References(*expr.left).empty() || References(*expr.right).empty())
		return;
	auto left = child.children[0]->GetColumnBindings(), right = child.children[1]->GetColumnBindings();
	bool direct = Available(*expr.left, left) && Available(*expr.right, right);
	bool flipped = Available(*expr.left, right) && Available(*expr.right, left);
	if (!direct && !flipped)
		return;
	JoinCondition condition;
	condition.left = std::move(direct ? expr.left : expr.right);
	condition.right = std::move(direct ? expr.right : expr.left);
	condition.comparison = direct ? expr.type : FlipComparisonExpression(expr.type);
	auto joined = std::move(plan->children[0]);
	if (joined->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		auto comparison = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
		comparison->children = std::move(joined->children);
		comparison->conditions.push_back(std::move(condition));
		plan = std::move(comparison);
	} else {
		joined->Cast<LogicalComparisonJoin>().conditions.push_back(std::move(condition));
		plan = std::move(joined);
	}
	plan->ResolveOperatorTypes();
}
std::vector<int64_t> ModelExecutionOrder(LogicalOperator &plan) {
	std::vector<int64_t> result;
	ExecutionOrder(plan, result);
	return result;
}

} // namespace semantic_extension
} // namespace duckdb
