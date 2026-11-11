#include "operators/plan/semantic_logical_operator.hpp"

namespace duckdb {
namespace semantic_extension {

SemanticLogicalOperator::SemanticLogicalOperator(unique_ptr<LogicalOperator> child, BoundFunctionExpression &marker,
                                                 idx_t table_index)
    : binding(marker.bind_info->Copy()), generated_type(marker.return_type), table_index(table_index) {
	children.push_back(std::move(child));
	for (idx_t i = 1; i < marker.children.size(); i++)
		expressions.push_back(std::move(marker.children[i]));
}

int64_t SemanticLogicalOperator::OperatorId() const {
	return binding->Cast<SemanticBindData>().node.semantic.operator_id;
}

bool SemanticLogicalOperator::SupportSerialization() const {
	return false;
}

}
}
