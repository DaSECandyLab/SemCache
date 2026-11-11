#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "llm/json_utils.hpp"

namespace duckdb {
namespace semantic_extension {

// Preserve a relational predicate for the learned planner: DuckDB must not
// absorb it into a scan or choose its position before the model sees it.
class LogicalModelPredicate : public LogicalExtensionOperator {
public:
	LogicalModelPredicate(unique_ptr<LogicalOperator> child, vector<unique_ptr<Expression>> predicates);
	idx_t query_node_id = DConstants::INVALID_INDEX;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	bool SupportSerialization() const override;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

protected:
	void ResolveTypes() override;
};

bool IsModelPredicate(const LogicalOperator &op);
void PreserveModelPredicates(unique_ptr<LogicalOperator> &plan, bool unary_region = false);
semantic::llm_detail::Json PredicateRegions(LogicalOperator &plan);
void ApplyPredicateOrder(unique_ptr<LogicalOperator> &plan, const semantic::llm_detail::Json &regions,
                         const semantic::llm_detail::Json &orders);

}
}
