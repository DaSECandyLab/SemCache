#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
namespace semantic_extension {

// Retain ordinary projections while exposing independent filters and semantic
// projections to structural planning. Trace typed, pure, nonthrowing expressions
// to their real inputs and carry generated fields through the original aliases.
void LiftModelUnaryProjections(unique_ptr<LogicalOperator> &plan);

}
}
