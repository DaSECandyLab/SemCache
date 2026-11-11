#include "operators/plan/semantic_plan_tree.hpp"
#include "operators/plan/semantic_projection_lineage.hpp"
#include "operators/plan/semantic_predicates.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace duckdb {
namespace semantic_extension {
namespace {
using Json = semantic::llm_detail::Json;
using Ids = std::map<LogicalOperator *, idx_t>;

bool Semantic(const LogicalOperator &op) {
	return op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
	       (op.GetName() == "SEMANTIC_FILTER" || op.GetName() == "SEMANTIC_PROJECTION" ||
	        op.GetName() == "SEMANTIC_JOIN");
}
vector<ColumnBinding> References(const Expression &expression) {
	vector<ColumnBinding> result;
	std::function<void(const Expression &)> visit = [&](const Expression &expr) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF ||
		    expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY)
			throw BinderException("Model tree requires resolved input field dependencies");
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &ref = expr.Cast<BoundColumnRefExpression>();
			if (ref.depth)
				throw BinderException("Model tree does not support correlated fields");
			result.push_back(ref.binding);
		}
		ExpressionIterator::EnumerateChildren(expr, visit);
	};
	visit(expression);
	return result;
}
bool Available(const Expression &expr, const vector<ColumnBinding> &fields) {
	for (auto &binding : References(expr))
		if (std::find(fields.begin(), fields.end(), binding) == fields.end())
			return false;
	return true;
}
bool Movable(LogicalOperator &op) {
	// Matched-pair semantic joins participate in structural search as unary
	// predicates. Bound pair-source roles, exported below, constrain their inputs
	// without freezing independent filters on the other side of the join.
	bool supported = Semantic(op) || IsModelPredicate(op) || op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
	if (op.type == LogicalOperatorType::LOGICAL_FILTER)
		supported = op.Cast<LogicalFilter>().projection_map.empty();
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();
		supported = join.join_type == JoinType::INNER && !join.HasProjectionMap() && !join.predicate &&
		            join.duplicate_eliminated_columns.empty();
	}
	if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = op.Cast<LogicalAnyJoin>();
		supported = join.join_type == JoinType::INNER && !join.HasProjectionMap() && join.condition;
	}
	// Reordering is governed by bound input dependencies, not expression error
	// or volatility flags. In particular, a cast inside one predicate must not
	// freeze the other predicates above a semantic join.
	return supported && !op.children.empty();
}
bool Freezable(LogicalOperator &op) {
	// These nodes require native lowering passes; do not silently advertise
	// structural search when this implementation cannot preserve their execution.
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN || op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE)
		return false;
	for (auto &child : op.children)
		if (!Freezable(*child))
			return false;
	return true;
}
void Assign(LogicalOperator &op, idx_t &next, Ids &ids) {
	ids[&op] = next++;
	for (auto &child : op.children)
		Assign(*child, next, ids);
}
void Gather(LogicalOperator &op, const Ids &ids, Json &members, Json &bases, bool &join) {
	if (!Movable(op)) {
		bases.push_back(ids.at(&op));
		return;
	}
	members.push_back(ids.at(&op));
	join |= op.children.size() == 2;
	for (auto &child : op.children)
		Gather(*child, ids, members, bases, join);
}
Json Role(const vector<ColumnBinding> &refs) {
	Json result = Json::array();
	for (auto &ref : refs)
		result.push_back({ref.table_index, ref.column_index});
	return result;
}
void Roles(LogicalOperator &op, Json &node) {
	if (Semantic(op) && op.GetName() == "SEMANTIC_JOIN" && op.children.size() == 1) {
		node["candidate_pair_columns"] = Json::array({Role(References(*op.expressions[0])),
		                                             Role(References(*op.expressions[1]))});
		return;
	}
	if (op.children.size() != 2)
		return;
	vector<ColumnBinding> left, right;
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		for (auto &cond : op.Cast<LogicalComparisonJoin>().conditions) {
			auto l = References(*cond.left), r = References(*cond.right);
			left.insert(left.end(), l.begin(), l.end());
			right.insert(right.end(), r.begin(), r.end());
		}
		node["join_commutative"] = true;
	} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto refs = References(*op.Cast<LogicalAnyJoin>().condition);
		auto l = op.children[0]->GetColumnBindings(), r = op.children[1]->GetColumnBindings();
		for (auto &ref : refs) {
			if (std::find(l.begin(), l.end(), ref) != l.end())
				left.push_back(ref);
			if (std::find(r.begin(), r.end(), ref) != r.end())
				right.push_back(ref);
		}
		node["join_commutative"] = true;
	} else if (Semantic(op)) {
		left = References(*op.expressions[0]);
		right = References(*op.expressions[1]);
		node["join_commutative"] = false;
	} else
		node["join_commutative"] = true;
	node["left_required_columns"] = Role(left);
	node["right_required_columns"] = Role(right);
}
using NodeIndex = std::map<idx_t, LogicalOperator *>;
using EntryIndex = std::map<idx_t, Json *>;
void Export(LogicalOperator &op, const Ids &ids, const NodeIndex &nodes, const EntryIndex &entries, Json &regions) {
	if (Movable(op)) {
		Json members = Json::array(), bases = Json::array();
		bool join = false;
		Gather(op, ids, members, bases, join);
		bool semantic = false;
		for (auto &member : members)
			semantic |= Semantic(*nodes.at(member.get<idx_t>()));
		// Unary semantic chains and independent branches use the same bound-field
		// legality as join regions. Even a singleton must retain its own region,
		// rather than imposing one global schedule on unrelated branches.
		if (join || semantic) {
			regions.push_back({{"region_id", ids.at(&op)}, {"nodes", members}, {"base_nodes", bases}});
			Json pair_targets = Json::array();
			// Visit query-local IDs in tree order, never allocator-address order.
			// Indexed entries also avoid repeatedly scanning the entire QueryIR.
			for (auto &member : members) {
				auto id = member.get<idx_t>();
				auto &node = *nodes.at(id);
				if (Semantic(node) && node.GetName() == "SEMANTIC_JOIN" && node.children.size() == 2)
					pair_targets.push_back(id);
			}
			for (auto &member : members) {
				auto id = member.get<idx_t>();
				auto &entry = *entries.at(id);
				Roles(*nodes.at(id), entry);
				if (entry["type"] == "FILTER" && !pair_targets.empty())
					entry["pair_targets"] = pair_targets;
			}
			for (auto &base : bases)
				Export(*nodes.at(base.get<idx_t>()), ids, nodes, entries, regions);
			return;
		}
	}
	for (auto &child : op.children)
		Export(*child, ids, nodes, entries, regions);
}

class FrozenPlan : public LogicalExtensionOperator {
public:
	explicit FrozenPlan(LogicalOperator &plan) : fields(plan.GetColumnBindings()), output_types(plan.types) {
	}
	vector<ColumnBinding> fields;
	vector<LogicalType> output_types;
	string GetName() const override {
		return "SEMANTIC_MODEL_PENDING";
	}
	vector<ColumnBinding> GetColumnBindings() override {
		return fields;
	}
	bool SupportSerialization() const override {
		return false;
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &) override {
		throw InternalException("Unapplied model tree reached physical planning");
	}

protected:
	void ResolveTypes() override {
		types = output_types;
	}
};
class BindingOrder : public LogicalExtensionOperator {
public:
	BindingOrder(unique_ptr<LogicalOperator> child, vector<ColumnBinding> fields, vector<LogicalType> output_types)
	    : fields(std::move(fields)), output_types(std::move(output_types)) {
		children.push_back(std::move(child));
		for (idx_t i = 0; i < this->fields.size(); ++i)
			expressions.push_back(make_uniq<BoundColumnRefExpression>(this->output_types[i], this->fields[i]));
	}
	vector<ColumnBinding> fields;
	vector<LogicalType> output_types;
	string GetName() const override {
		return "MODEL_BINDING_ORDER";
	}
	vector<ColumnBinding> GetColumnBindings() override {
		return fields;
	}
	bool SupportSerialization() const override {
		return false;
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		auto &child = planner.CreatePlan(*children[0]);
		auto &result = planner.Make<PhysicalProjection>(types, std::move(expressions), child.estimated_cardinality);
		result.children.push_back(child);
		return result;
	}

protected:
	void ResolveTypes() override {
		types = output_types;
	}
};
idx_t Id(const Json &value) {
	if (!value.is_number_integer() || value.get<int64_t>() < 0)
		throw BinderException("Model tree IDs must be nonnegative integers");
	return value.get<idx_t>();
}
void Detach(unique_ptr<LogicalOperator> plan, const Ids &ids, const std::set<idx_t> &members,
            std::map<idx_t, unique_ptr<LogicalOperator>> &nodes, std::map<idx_t, unique_ptr<LogicalOperator>> &roots) {
	auto id = ids.at(plan.get());
	if (!members.count(id)) {
		roots.emplace(id, std::move(plan));
		return;
	}
	for (auto &child : plan->children)
		Detach(std::move(child), ids, members, nodes, roots);
	plan->children.clear();
	nodes.emplace(id, std::move(plan));
}
void CheckInputs(LogicalOperator &node) {
	if (node.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = node.Cast<LogicalComparisonJoin>();
		auto left = node.children[0]->GetColumnBindings(), right = node.children[1]->GetColumnBindings();
		bool direct = true, flipped = true;
		for (auto &cond : join.conditions) {
			direct &= Available(*cond.left, left) && Available(*cond.right, right);
			flipped &= Available(*cond.left, right) && Available(*cond.right, left);
		}
		if (!direct && !flipped)
			throw BinderException("Model join predicate is not available on its input sides");
		if (!direct)
			for (auto &cond : join.conditions) {
				std::swap(cond.left, cond.right);
				cond.comparison = FlipComparisonExpression(cond.comparison);
			}
	} else if (Semantic(node) && node.GetName() == "SEMANTIC_JOIN") {
		auto &source = node.children.size() == 1 ? ModelPairSource(*node.children[0]) : node;
		// Matched-pair execution evaluates both bound expressions on the joined
		// row. Input labels do not constrain the relational join's partitioning.
		bool candidate_pipeline = node.children.size() == 1 && source.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
		if (node.children.size() == 1)
			for (auto current = node.children[0].get(); current != &source; current = current->children[0].get())
				candidate_pipeline |= current->GetName() == "SEMANTIC_FILTER" || current->GetName() == "MODEL_BINDING_ORDER";
		if (candidate_pipeline) {
			auto fields = node.children[0]->GetColumnBindings();
			for (auto &expr : node.expressions)
				if (!Available(*expr, fields))
					throw BinderException("Model semantic join uses an unavailable bound pair field");
			return;
		}
		for (idx_t i = 0; i < 2; ++i)
			if (!Available(*node.expressions[i], source.children[i]->GetColumnBindings()))
				throw BinderException("Model semantic join changed bound input roles");
	} else {
		vector<ColumnBinding> available;
		for (auto &child : node.children) {
			auto fields = child->GetColumnBindings();
			available.insert(available.end(), fields.begin(), fields.end());
		}
		LogicalOperatorVisitor::EnumerateExpressions(node, [&](unique_ptr<Expression> *expr) {
			if (!Available(**expr, available))
				throw BinderException("Model tree uses a field before its producer");
		});
	}
}
void Apply(unique_ptr<LogicalOperator> &plan, const Ids &ids, const Json &region, const Json &choice, bool &found) {
	if (ids.count(plan.get()) && ids.at(plan.get()) == Id(region["region_id"])) {
		found = true;
		auto fields = plan->GetColumnBindings();
		auto types = plan->types;
		std::set<idx_t> members;
		for (auto &n : region["nodes"])
			members.insert(Id(n));
		std::map<idx_t, unique_ptr<LogicalOperator>> nodes, roots;
		std::map<idx_t, idx_t> arities;
		std::map<idx_t, idx_t> pending;
		for (auto &item : ids)
			if (members.count(item.second))
				arities[item.second] = item.first->children.size();
		Detach(std::move(plan), ids, members, nodes, roots);
		if (choice["actions"].size() != members.size())
			throw BinderException("Model tree has missing operators");
		for (auto &action : choice["actions"]) {
			if (!action.is_object() || action.size() != (action.contains("pair_for") ? 3 : 2) ||
			    !action.contains("node_id") || !action.contains("inputs") || !action["inputs"].is_array())
				throw BinderException("Invalid model tree action");
			auto id = Id(action["node_id"]);
			if (!nodes.count(id))
				throw BinderException("Unknown, duplicate or wrong-arity model tree action");
			auto count = action["inputs"].size();
			bool candidate = action.contains("pair_for");
			idx_t target = candidate ? Id(action["pair_for"]) : 0;
			bool paired = Semantic(*nodes.at(id)) && nodes.at(id)->GetName() == "SEMANTIC_JOIN" && count == 1 &&
			              pending.count(Id(action["inputs"][0])) && pending.at(Id(action["inputs"][0])) == id;
			if (candidate) {
				if (count != 2 ||
				    (nodes.at(id)->type != LogicalOperatorType::LOGICAL_FILTER && !IsModelPredicate(*nodes.at(id))) ||
				    !nodes.count(target) || !Semantic(*nodes.at(target)) ||
				    nodes.at(target)->GetName() != "SEMANTIC_JOIN")
					throw BinderException("Invalid candidate-pair target");
			} else if (count != arities[id] && !paired)
				throw BinderException("Wrong-arity model tree action");
			auto node = std::move(nodes.at(id));
			nodes.erase(id);
			idx_t inherited = 0;
			bool has_pending = false;
			for (auto &input : action["inputs"]) {
				auto root = Id(input);
				if (!roots.count(root))
					throw BinderException("Model tree consumes a missing or already consumed input");
				if (pending.count(root)) {
					if (count != 1 ||
					    (!paired && node->type != LogicalOperatorType::LOGICAL_FILTER && !IsModelPredicate(*node) &&
					     !(Semantic(*node) && node->GetName() == "SEMANTIC_FILTER")))
						throw BinderException("Invalid candidate-pair consumer");
					inherited = pending.at(root);
					has_pending = true;
					pending.erase(root);
				}
				node->children.push_back(std::move(roots.at(root)));
				roots.erase(root);
			}
			if (candidate) {
				for (idx_t side = 0; side < 2; ++side) {
					auto fields = node->children[side]->GetColumnBindings();
					bool referenced = false;
					for (auto &expr : node->expressions)
						for (auto &ref : References(*expr))
							referenced |= std::find(fields.begin(), fields.end(), ref) != fields.end();
					if (!referenced || !Available(*nodes.at(target)->expressions[side], fields))
						throw BinderException("Candidate predicate must use both bound join sides");
				}
				auto cross = make_uniq<LogicalCrossProduct>(std::move(node->children[0]), std::move(node->children[1]));
				node->children.clear();
				node->children.push_back(std::move(cross));
				pending[id] = target;
			} else if (has_pending && !paired)
				pending[id] = inherited;
			CheckInputs(*node);
			node->ResolveOperatorTypes();
			roots.emplace(id, std::move(node));
		}
		if (!nodes.empty() || !pending.empty() || roots.size() != 1 || !roots.count(Id(choice["root"])))
			throw BinderException("Model tree must use all operators and inputs exactly once");
		plan = std::move(roots.begin()->second);
		auto actual = plan->GetColumnBindings();
		if (actual.size() != fields.size())
			throw BinderException("Model tree changed boundary fields");
		for (idx_t i = 0; i < fields.size(); ++i) {
			auto pos = std::find(actual.begin(), actual.end(), fields[i]);
			if (pos == actual.end() || plan->types[pos - actual.begin()] != types[i])
				throw BinderException("Model tree changed boundary field identities or types");
		}
		if (actual != fields)
			plan = make_uniq<BindingOrder>(std::move(plan), std::move(fields), std::move(types));
		return;
	}
	for (auto &child : plan->children)
		Apply(child, ids, region, choice, found);
}
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
}

Json ModelTreeRegions(LogicalOperator &plan, Json &query) {
	Json regions = Json::array();
	if (!Freezable(plan))
		return regions;
	Ids ids;
	idx_t next = 0;
	Assign(plan, next, ids);
	NodeIndex nodes;
	EntryIndex entries;
	for (auto &item : ids)
		nodes.emplace(item.second, item.first);
	for (auto &entry : query["operators"])
		entries.emplace(entry["node_id"].get<idx_t>(), &entry);
	Export(plan, ids, nodes, entries, regions);
	return regions;
}
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
LogicalOperator &ModelPairSource(LogicalOperator &plan) {
	auto current = &plan;
	while ((IsModelPredicate(*current) || current->type == LogicalOperatorType::LOGICAL_FILTER ||
	        (Semantic(*current) && current->GetName() == "SEMANTIC_FILTER") ||
	        current->GetName() == "MODEL_BINDING_ORDER") &&
	       current->children.size() == 1)
		current = current->children[0].get();
	if ((current->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT &&
	     current->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	     current->type != LogicalOperatorType::LOGICAL_ANY_JOIN) ||
	    current->children.size() != 2)
		throw BinderException("Candidate-pair source requires a binary join below filters or binding-order wrappers; "
		                      "found %s with %llu inputs", current->GetName(),
		                      static_cast<unsigned long long>(current->children.size()));
	return *current;
}
unique_ptr<LogicalOperator> FreezeModelPlan(LogicalOperator &plan) {
	return make_uniq<FrozenPlan>(plan);
}
void ApplyModelTrees(unique_ptr<LogicalOperator> &plan, const Json &regions, const Json &choices) {
	if (!choices.is_array() || choices.size() != regions.size())
		throw BinderException("Missing model tree regions");
	Ids ids;
	idx_t next = 0;
	Assign(*plan, next, ids);
	std::set<idx_t> used;
	for (auto &choice : choices) {
		if (!choice.is_object() || choice.size() != 3 || !choice.contains("region_id") || !choice.contains("actions") ||
		    !choice.contains("root") || !choice["actions"].is_array())
			throw BinderException("Invalid model tree contract");
		auto id = Id(choice["region_id"]);
		if (!used.insert(id).second)
			throw BinderException("Duplicate model tree region");
		auto region = std::find_if(regions.begin(), regions.end(), [&](const Json &r) { return r["region_id"] == id; });
		if (region == regions.end())
			throw BinderException("Unknown model tree region");
		bool found = false;
		Apply(plan, ids, *region, choice, found);
		if (!found)
			throw BinderException("Model tree region disappeared");
	}
}
std::vector<int64_t> ModelExecutionOrder(LogicalOperator &plan) {
	std::vector<int64_t> result;
	ExecutionOrder(plan, result);
	return result;
}
}
}
