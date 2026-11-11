#include "operators/plan/semantic_predicates.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace duckdb {
namespace semantic_extension {
namespace {
using Json = semantic::llm_detail::Json;

bool UnarySemantic(const LogicalOperator &op) {
	return op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR && op.children.size() == 1 &&
	       (op.GetName() == "SEMANTIC_FILTER" || op.GetName() == "SEMANTIC_PROJECTION");
}
bool RegionNode(const LogicalOperator &op) {
	return UnarySemantic(op) || IsModelPredicate(op);
}
bool HasUnarySemantic(const LogicalOperator &op) {
	if (UnarySemantic(op))
		return true;
	return op.type == LogicalOperatorType::LOGICAL_FILTER && op.children.size() == 1 &&
	       HasUnarySemantic(*op.children[0]);
}
bool Available(const Expression &expr, const vector<ColumnBinding> &bindings) {
	bool valid = true;
	std::function<void(const Expression &)> visit = [&](const Expression &child) {
		if (child.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY ||
		    child.GetExpressionClass() == ExpressionClass::BOUND_REF)
			valid = false;
		if (child.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &ref = child.Cast<BoundColumnRefExpression>();
			valid &= ref.depth == 0 && std::find(bindings.begin(), bindings.end(), ref.binding) != bindings.end();
		}
		ExpressionIterator::EnumerateChildren(child, visit);
	};
	visit(expr);
	return valid;
}
idx_t NodeId(LogicalOperator &op) {
	return IsModelPredicate(op) ? static_cast<LogicalModelPredicate &>(op).query_node_id
	                            : static_cast<SemanticLogicalOperator &>(op).query_node_id;
}
void Assign(LogicalOperator &op, idx_t &next, std::map<LogicalOperator *, idx_t> &ids) {
	// Match Extract's preorder IDs; the serialized array itself is postorder.
	ids[&op] = next;
	if (IsModelPredicate(op))
		static_cast<LogicalModelPredicate &>(op).query_node_id = next;
	else if (UnarySemantic(op))
		static_cast<SemanticLogicalOperator &>(op).query_node_id = next;
	++next;
	for (auto &child : op.children)
		Assign(*child, next, ids);
}
void Export(LogicalOperator &op, const std::map<LogicalOperator *, idx_t> &ids, Json &regions) {
	if (!RegionNode(op)) {
		for (auto &child : op.children)
			Export(*child, ids, regions);
		return;
	}
	auto current = &op;
	Json members = Json::array();
	bool predicate = false, semantic = false;
	while (RegionNode(*current)) {
		predicate |= IsModelPredicate(*current);
		semantic |= UnarySemantic(*current);
		members.push_back(ids.at(current));
		current = current->children[0].get();
	}
	if (predicate && semantic) {
		std::reverse(members.begin(), members.end());
		regions.push_back({{"region_id", ids.at(&op)}, {"nodes", members}, {"base_node_id", ids.at(current)}});
	}
	Export(*current, ids, regions);
}
void Apply(unique_ptr<LogicalOperator> &plan, idx_t region_id, const Json &members, const Json &order, bool &found) {
	if (RegionNode(*plan) && NodeId(*plan) == region_id) {
		if (found)
			throw BinderException("Duplicate predicate region");
		found = true;
		std::map<idx_t, unique_ptr<LogicalOperator>> nodes;
		while (RegionNode(*plan)) {
			auto node = std::move(plan);
			plan = std::move(node->children[0]);
			node->children.clear();
			auto id = NodeId(*node);
			if (nodes.count(id))
				throw BinderException("Duplicate predicate region node");
			nodes.emplace(id, std::move(node));
		}
		if (nodes.size() != members.size() || order.size() != members.size())
			throw BinderException("Predicate region changed or has missing nodes");
		std::set<idx_t> expected, used;
		for (auto &id : members)
			expected.insert(id.get<idx_t>());
		for (auto &value : order) {
			if (!value.is_number_unsigned() && (!value.is_number_integer() || value.get<int64_t>() < 0))
				throw BinderException("Predicate node IDs must be nonnegative integers");
			auto id = value.get<idx_t>();
			if (!expected.count(id) || !nodes.count(id) || !used.insert(id).second)
				throw BinderException("Unknown or duplicate predicate region node");
			auto node = std::move(nodes.at(id));
			auto available = plan->GetColumnBindings();
			for (auto &expr : node->expressions)
				if (!Available(*expr, available))
					throw BinderException("Predicate order consumes fields before their producer");
			node->children.push_back(std::move(plan));
			plan = std::move(node);
		}
		return;
	}
	for (auto &child : plan->children)
		Apply(child, region_id, members, order, found);
}
}

LogicalModelPredicate::LogicalModelPredicate(unique_ptr<LogicalOperator> child,
                                             vector<unique_ptr<Expression>> predicates) {
	children.push_back(std::move(child));
	expressions = std::move(predicates);
}
string LogicalModelPredicate::GetName() const {
	return "FILTER";
}
vector<ColumnBinding> LogicalModelPredicate::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}
bool LogicalModelPredicate::SupportSerialization() const {
	return false;
}
void LogicalModelPredicate::ResolveTypes() {
	types = children[0]->types;
}
PhysicalOperator &LogicalModelPredicate::CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) {
	auto &child = planner.CreatePlan(*children[0]);
	auto &result = planner.Make<PhysicalFilter>(types, std::move(expressions), child.estimated_cardinality);
	result.children.push_back(child);
	return result;
}
bool IsModelPredicate(const LogicalOperator &op) {
	return op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR && op.GetName() == "FILTER";
}
void PreserveModelPredicates(unique_ptr<LogicalOperator> &plan, bool unary_region) {
	bool extend = UnarySemantic(*plan) || plan->type == LogicalOperatorType::LOGICAL_FILTER;
	bool eligible = plan->type == LogicalOperatorType::LOGICAL_FILTER && (unary_region || HasUnarySemantic(*plan));
	for (auto &child : plan->children)
		PreserveModelPredicates(child, extend && (unary_region || UnarySemantic(*plan)));
	if (!eligible)
		return;
	auto &filter = plan->Cast<LogicalFilter>();
	if (!filter.projection_map.empty() || filter.expressions.empty())
		return;
	for (auto &expr : filter.expressions)
		if (!Available(*expr, filter.children[0]->GetColumnBindings()))
			return;
	plan = make_uniq<LogicalModelPredicate>(std::move(filter.children[0]), std::move(filter.expressions));
}
Json PredicateRegions(LogicalOperator &plan) {
	std::map<LogicalOperator *, idx_t> ids;
	idx_t next = 0;
	Assign(plan, next, ids);
	Json result = Json::array();
	Export(plan, ids, result);
	return result;
}
void ApplyPredicateOrder(unique_ptr<LogicalOperator> &plan, const Json &regions, const Json &orders) {
	if (!orders.is_array() || orders.size() != regions.size())
		throw BinderException("Missing predicate regions");
	std::set<idx_t> used;
	for (auto &entry : orders) {
		if (!entry.is_object() || entry.size() != 2 || !entry.contains("region_id") || !entry.contains("nodes") ||
		    !entry["region_id"].is_number_integer() || entry["region_id"].get<int64_t>() < 0 ||
		    !entry["nodes"].is_array())
			throw BinderException("Invalid predicate order contract");
		auto id = entry["region_id"].get<idx_t>();
		if (!used.insert(id).second)
			throw BinderException("Duplicate predicate region order");
		auto region = std::find_if(regions.begin(), regions.end(), [&](const Json &r) { return r["region_id"] == id; });
		if (region == regions.end())
			throw BinderException("Unknown predicate region order");
		bool found = false;
		Apply(plan, id, (*region)["nodes"], entry["nodes"], found);
		if (!found)
			throw BinderException("Predicate region disappeared before model decision");
	}
}
}
}
