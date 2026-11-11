#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
namespace semantic_extension {
// Attach only after QueryIR and PlanSpec validation, so recording is not a search action.
void CaptureAggregateInput(unique_ptr<LogicalOperator> &plan, const string &path);
}
}
