#pragma once

// Common DuckDB LogicalExtensionOperator carrying a validated semantic binding
// and the bound expressions used to construct model input rows.

#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "operators/binding/semantic_bind.hpp"

namespace duckdb {
namespace semantic_extension {

class SemanticLogicalOperator : public LogicalExtensionOperator {
public:
	SemanticLogicalOperator(unique_ptr<LogicalOperator> child, BoundFunctionExpression &marker, idx_t table_index);

	unique_ptr<FunctionData> binding;
	LogicalType generated_type;
	idx_t table_index;
	idx_t query_node_id = DConstants::INVALID_INDEX;

	int64_t OperatorId() const;
	bool SupportSerialization() const override;
};

}
}
