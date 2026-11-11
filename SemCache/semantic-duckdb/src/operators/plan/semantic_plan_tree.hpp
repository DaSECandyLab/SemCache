#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "llm/json_utils.hpp"

namespace duckdb {
namespace semantic_extension {

// Bound expressions remain in C++; the model selects their input tree.
void NormalizeModelSPJ(unique_ptr<LogicalOperator> &plan);
semantic::llm_detail::Json ModelTreeRegions(LogicalOperator &plan, semantic::llm_detail::Json &query);
unique_ptr<LogicalOperator> FreezeModelPlan(LogicalOperator &plan);
void ApplyModelTrees(unique_ptr<LogicalOperator> &plan, const semantic::llm_detail::Json &regions,
                     const semantic::llm_detail::Json &choices);
std::vector<int64_t> ModelExecutionOrder(LogicalOperator &plan);
void FuseModelJoins(unique_ptr<LogicalOperator> &plan);
LogicalOperator &ModelPairSource(LogicalOperator &plan);

}
}
