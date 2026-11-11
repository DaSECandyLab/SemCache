#pragma once


#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
namespace semantic_extension {

struct SemanticBindData;

unique_ptr<LogicalOperator> CreateLogicalSemanticFilter(unique_ptr<LogicalOperator> child,
                                                        BoundFunctionExpression &marker, idx_t table_index);
unique_ptr<LogicalOperator> CreateLogicalSemanticProjection(unique_ptr<LogicalOperator> child,
                                                            BoundFunctionExpression &marker, idx_t table_index);
unique_ptr<LogicalOperator> CreateLogicalSemanticJoin(unique_ptr<LogicalOperator> child,
                                                      BoundFunctionExpression &marker, idx_t table_index);
PhysicalOperator &OrderSamplingInput(PhysicalPlanGenerator &planner, LogicalOperator &logical_child,
                                     PhysicalOperator &physical_child, const SemanticBindData &binding);
PhysicalOperator &CreatePhysicalSemanticPairFilter(PhysicalPlanGenerator &planner, vector<LogicalType> types,
                                                   PhysicalOperator &child, unique_ptr<FunctionData> binding,
                                                   vector<unique_ptr<Expression>> expressions);

}
}
