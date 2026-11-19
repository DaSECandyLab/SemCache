#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include <vector>

namespace duckdb {
namespace semantic_extension {
namespace plan_tree_detail {

bool Semantic(const LogicalOperator &op);
vector<ColumnBinding> References(const Expression &expression);
bool Available(const Expression &expr, const vector<ColumnBinding> &fields);
bool Movable(LogicalOperator &op);
void ExecutionOrder(LogicalOperator &op, std::vector<int64_t> &order);

} // namespace plan_tree_detail
} // namespace semantic_extension
} // namespace duckdb
