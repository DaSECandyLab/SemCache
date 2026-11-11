#pragma once


#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {
namespace semantic_extension {

void RewriteSemanticOperators(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);

}
}
