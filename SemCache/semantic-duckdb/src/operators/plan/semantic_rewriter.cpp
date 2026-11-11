// Logical-plan normalization and marker lowering. This is the point where a
// semantic SQL expression becomes a first-class DuckDB operator boundary.
#include "operators/plan/semantic_rewriter.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "operators/execution/semantic_operator_factory.hpp"
#include "operators/plan/semantic_plan_tree.hpp"
#include <algorithm>

namespace duckdb {
namespace semantic_extension {
namespace {

BoundFunctionExpression *GetMarker(Expression &expression) {
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION)
		return nullptr;
	auto &function = expression.Cast<BoundFunctionExpression>();
	if (function.function.name == "semantic_filter" || function.function.name == "semantic_project" ||
	    function.function.name == "semantic_join")
		return &function;
	return nullptr;
}

void RejectMarker(Expression &expression) {
	if (GetMarker(expression))
		throw BinderException("Unsupported semantic marker placement; use a WHERE conjunct or SELECT projection");
	ExpressionIterator::EnumerateChildren(expression, [](Expression &child) { RejectMarker(child); });
}

bool HasJoinBoundary(const LogicalOperator &plan) {
	if (plan.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT)
		return true;
	if (plan.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
		return plan.Cast<LogicalComparisonJoin>().join_type == JoinType::INNER;
	if (plan.type == LogicalOperatorType::LOGICAL_ANY_JOIN)
		return plan.Cast<LogicalAnyJoin>().join_type == JoinType::INNER;
	return plan.children.size() == 1 && HasJoinBoundary(*plan.children[0]);
}

unique_ptr<LogicalOperator> CreateLogicalOperator(unique_ptr<LogicalOperator> child, BoundFunctionExpression &marker,
                                                  idx_t table_index) {
	if (marker.function.name == "semantic_filter")
		return CreateLogicalSemanticFilter(std::move(child), marker, table_index);
	if (marker.function.name == "semantic_project")
		return CreateLogicalSemanticProjection(std::move(child), marker, table_index);
	return CreateLogicalSemanticJoin(std::move(child), marker, table_index);
}

void RestoreSemanticCrossProduct(unique_ptr<LogicalOperator> &plan) {
	// DuckDB may turn CROSS JOIN + WHERE into LogicalAnyJoin before extension
	// optimization. Restore the explicit binary boundary needed by our operator.
	if (plan->type != LogicalOperatorType::LOGICAL_ANY_JOIN)
		return;
	auto &join = plan->Cast<LogicalAnyJoin>();
	bool semantic_join = false;
	ExpressionIterator::VisitExpression<BoundFunctionExpression>(*join.condition,
	                                                             [&](const BoundFunctionExpression &expr) {
		                                                             if (expr.function.name == "semantic_join")
			                                                             semantic_join = true;
	                                                             });
	if (!semantic_join)
		return;
	if (join.join_type != JoinType::INNER)
		throw BinderException("Semantic SQL supports inner semantic joins only");

	auto filter = make_uniq<LogicalFilter>(std::move(join.condition));
	if (join.HasProjectionMap()) {
		auto left_count = join.children[0]->GetColumnBindings().size();
		auto right_count = join.children[1]->GetColumnBindings().size();
		if (join.left_projection_map.empty()) {
			for (idx_t i = 0; i < left_count; i++)
				filter->projection_map.push_back(i);
		} else {
			for (auto i : join.left_projection_map)
				filter->projection_map.push_back(i);
		}
		if (join.right_projection_map.empty()) {
			for (idx_t i = 0; i < right_count; i++)
				filter->projection_map.push_back(left_count + i);
		} else {
			for (auto i : join.right_projection_map)
				filter->projection_map.push_back(left_count + i);
		}
	}
	filter->children.push_back(
	    make_uniq<LogicalCrossProduct>(std::move(join.children[0]), std::move(join.children[1])));
	plan = std::move(filter);
}

void Rewrite(unique_ptr<LogicalOperator> &plan, Binder &binder, ClientContext &context) {
	// Rewrite children first so nested semantic stages retain bottom-up dataflow.
	for (auto &child : plan->children)
		Rewrite(child, binder, context);

	// DuckDB v1.5.4 absorbs CROSS JOIN + WHERE into LogicalAnyJoin.
	RestoreSemanticCrossProduct(plan);

	if (plan->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = plan->Cast<LogicalFilter>();
		// Bind the binary task before unary markers wrap its boundary. Position
		// selection happens later, after all tasks become model-visible operators.
		std::stable_partition(filter.expressions.begin(), filter.expressions.end(),
		                      [](const unique_ptr<Expression> &expr) {
			                      auto marker = GetMarker(*expr);
			                      return marker && marker->function.name == "semantic_join";
		                      });
		for (idx_t i = 0; i < filter.expressions.size();) {
			auto marker = GetMarker(*filter.expressions[i]);
			if (marker && marker->function.name != "semantic_project") {
				// Preserve a bound relational ON condition as an atomic join;
				// the semantic predicate consumes its matching row pairs.
				if (marker->function.name == "semantic_join" && !HasJoinBoundary(*filter.children[0])) {
					throw BinderException(
					    "semantic_join requires attributes from both left and right of a direct inner join");
				}
				for (idx_t j = 1; j < marker->children.size(); j++)
					RejectMarker(*marker->children[j]);
				PrepareSemanticSQL(context, *filter.children[0], *marker);
				filter.children[0] =
				    CreateLogicalOperator(std::move(filter.children[0]), *marker, binder.GenerateTableIndex());
				filter.expressions.erase_at(i);
			} else {
				++i;
			}
		}
	}

	if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		std::function<void(unique_ptr<Expression> &)> lower = [&](unique_ptr<Expression> &expression) {
			auto marker = GetMarker(*expression);
			if (marker && marker->function.name == "semantic_project") {
				for (idx_t j = 1; j < marker->children.size(); j++)
					RejectMarker(*marker->children[j]);
				auto table_index = binder.GenerateTableIndex();
				auto type = expression->return_type;
				auto alias = expression->alias;
				PrepareSemanticSQL(context, *plan->children[0], *marker);
				plan->children[0] = CreateLogicalSemanticProjection(std::move(plan->children[0]), *marker, table_index);
				expression = make_uniq<BoundColumnRefExpression>(alias, type, ColumnBinding(table_index, 0));
				return;
			}
			ExpressionIterator::EnumerateChildren(*expression, lower);
		};
		for (auto &expression : plan->expressions)
			lower(expression);
	}

	for (auto &expression : plan->expressions)
		RejectMarker(*expression);
}

bool IsSemanticLogicalOperator(const LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR)
		return false;
	auto name = op.GetName();
	return name == "SEMANTIC_FILTER" || name == "SEMANTIC_PROJECTION" || name == "SEMANTIC_JOIN";
}

struct MarkerGuard : LogicalOperatorVisitor {
	// After rewriting, no executable marker scalar may remain. IDs also must be
	// unique so optimizer actions cannot address two physical operations at once.
	std::set<int64_t> ids;

	void VisitOperator(LogicalOperator &op) override {
		if (IsSemanticLogicalOperator(op)) {
			auto &semantic_operator = static_cast<SemanticLogicalOperator &>(op);
			if (semantic_operator.OperatorId() != 0 && !ids.insert(semantic_operator.OperatorId()).second)
				throw BinderException("Each semantic operator_id must occur once in SQL; reference its generated "
				                      "column through a subquery");
		}
		LogicalOperatorVisitor::VisitOperator(op);
	}

	void VisitExpression(unique_ptr<Expression> *expression) override {
		RejectMarker(**expression);
	}
};

}

void RewriteSemanticOperators(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	Rewrite(plan, input.optimizer.binder, input.context);
	MarkerGuard guard;
	guard.VisitOperator(*plan);
}

}
}
