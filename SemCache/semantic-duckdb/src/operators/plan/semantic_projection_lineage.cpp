#include "operators/plan/semantic_projection_lineage.hpp"
#include "operators/plan/semantic_predicates.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <algorithm>
#include <functional>
#include <map>

namespace duckdb {
namespace semantic_extension {
namespace {

bool IsFilter(LogicalOperator &op) {
	if (op.children.size() != 1)
		return false;
	if (op.type == LogicalOperatorType::LOGICAL_FILTER)
		return op.Cast<LogicalFilter>().projection_map.empty();
	return IsModelPredicate(op) ||
	       (op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR && op.GetName() == "SEMANTIC_FILTER");
}

bool IsProjection(LogicalOperator &op) {
	return op.children.size() == 1 && op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
	       op.GetName() == "SEMANTIC_PROJECTION";
}

bool Available(const Expression &expression, const vector<ColumnBinding> &fields, const vector<LogicalType> &types) {
	if (fields.size() != types.size())
		return false;
	bool available = true;
	std::function<void(const Expression &)> visit = [&](const Expression &expr) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF ||
		    expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY)
			available = false;
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &ref = expr.Cast<BoundColumnRefExpression>();
			auto field = std::find(fields.begin(), fields.end(), ref.binding);
			if (ref.depth || field == fields.end() || types[field - fields.begin()] != ref.return_type)
				available = false;
		}
		ExpressionIterator::EnumerateChildren(expr, visit);
	};
	visit(expression);
	return available;
}

using Replacement = std::pair<unique_ptr<Expression> *, unique_ptr<Expression>>;

bool TraceInputs(LogicalOperator &op, LogicalProjection &projection, vector<Replacement> &replacements) {
	if (projection.children.size() != 1)
		return false;
	auto &input = *projection.children[0];
	auto input_fields = input.GetColumnBindings();
	// Filtering earlier must not suppress projection errors or side effects.
	// The original projection expressions remain intact. Copies of expressions
	// such as bound struct_extract may also be evaluated inside the predicate.
	for (auto &expr : projection.expressions) {
		if (expr->IsVolatile() || expr->CanThrow() || expr->IsAggregate() || expr->IsWindow() ||
		    !Available(*expr, input_fields, input.types))
			return false;
	}
	auto projected_fields = projection.GetColumnBindings();
	bool valid = true;
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *expression) {
		auto rewritten = (*expression)->Copy();
		std::function<bool(unique_ptr<Expression> &)> rewrite = [&](unique_ptr<Expression> &expr) {
			if (expr->GetExpressionClass() == ExpressionClass::BOUND_REF ||
			    expr->GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY)
				return false;
			if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &ref = expr->Cast<BoundColumnRefExpression>();
				auto field = std::find(projected_fields.begin(), projected_fields.end(), ref.binding);
				if (ref.depth || field == projected_fields.end())
					return false;
				auto &source = *projection.expressions[field - projected_fields.begin()];
				if (source.return_type != ref.return_type)
					return false;
				auto replacement = source.Copy();
				if (!ref.alias.empty())
					replacement->alias = ref.alias;
				// Source references already belong to the projection input. Do
				// not interpret them again as projection output references.
				expr = std::move(replacement);
				return true;
			}
			bool children_valid = true;
			ExpressionIterator::EnumerateChildren(
			    *expr, [&](unique_ptr<Expression> &child) { children_valid &= rewrite(child); });
			return children_valid;
		};
		if (!rewrite(rewritten) || !Available(*rewritten, input_fields, input.types)) {
			valid = false;
			return;
		}
		replacements.emplace_back(expression, std::move(rewritten));
	});
	return valid;
}

using BindingKey = std::pair<idx_t, idx_t>;
using Aliases = std::map<BindingKey, ColumnBinding>;

void RebindInputs(LogicalOperator &op, const Aliases &aliases) {
	if (aliases.empty())
		return;
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *expression) {
		std::function<void(Expression &)> visit = [&](Expression &expr) {
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &ref = expr.Cast<BoundColumnRefExpression>();
				auto found = aliases.find({ref.binding.table_index, ref.binding.column_index});
				if (!ref.depth && found != aliases.end())
					ref.binding = found->second;
			}
			ExpressionIterator::EnumerateChildren(expr, visit);
		};
		visit(**expression);
	});
}

Aliases Lift(unique_ptr<LogicalOperator> &plan) {
	const auto fields = plan->GetColumnBindings();
	const auto types = plan->types;
	for (auto &child : plan->children)
		RebindInputs(*plan, Lift(child));
	const bool project = IsProjection(*plan);
	if ((project || IsFilter(*plan)) && plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		vector<Replacement> replacements;
		if (TraceInputs(*plan, plan->children[0]->Cast<LogicalProjection>(), replacements)) {
			// Prepare every replacement before changing the tree. The ordinary
			// projection retains its expressions, aliases and original table index.
			for (auto &entry : replacements)
				*entry.first = std::move(entry.second);
			auto unary = std::move(plan);
			ColumnBinding generated_field;
			if (project)
				generated_field = unary->GetColumnBindings().back();
			auto projection = std::move(unary->children[0]);
			if (project) {
				// A semantic projection appends one STRUCT. Carry it through the
				// ordinary projection; its new boundary binding is propagated only
				// to ancestors, never back into this expression or its producer.
				projection->expressions.push_back(
				    make_uniq<BoundColumnRefExpression>(unary->types.back(), generated_field));
				projection->types.push_back(unary->types.back());
			}
			unary->children[0] = std::move(projection->children[0]);
			auto input_types = unary->children[0]->types;
			if (project)
				input_types.push_back(unary->types.back());
			unary->types = std::move(input_types);
			projection->children[0] = std::move(unary);
			plan = std::move(projection);
			// Keep moving across pure projections, stopping at every other
			// boundary. Rebind all expressions, including the newly carried field.
			RebindInputs(*plan, Lift(plan->children[0]));
		}
	}
	const auto result = plan->GetColumnBindings();
	if (fields.size() != result.size() || types != plan->types)
		throw InternalException("Projection lineage changed boundary column count or types");
	Aliases aliases;
	if (fields == result)
		return aliases;
	for (idx_t i = 0; i < fields.size(); ++i) {
		auto entry = aliases.emplace(BindingKey(fields[i].table_index, fields[i].column_index), result[i]);
		if (!entry.second && entry.first->second != result[i])
			throw InternalException("Projection lineage changed an ambiguous column binding");
	}
	for (auto entry = aliases.begin(); entry != aliases.end();) {
		if (entry->first == BindingKey(entry->second.table_index, entry->second.column_index))
			entry = aliases.erase(entry);
		else
			++entry;
	}
	return aliases;
}

}

void LiftModelUnaryProjections(unique_ptr<LogicalOperator> &plan) {
	plan->ResolveOperatorTypes();
	Lift(plan);
}

}
}
