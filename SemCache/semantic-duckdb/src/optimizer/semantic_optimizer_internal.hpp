#pragma once

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "llm/json_utils.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include <cstdint>

namespace duckdb {
namespace semantic_extension {
namespace optimizer_detail {

using Json = semantic::llm_detail::Json;

SemanticLogicalOperator *Semantic(LogicalOperator &op);
bool Filter(LogicalOperator &op);
void Regions(LogicalOperator &op, Json &groups);
int64_t Id(const Json &value);
string FieldName(const ColumnBinding &binding);
string Name(const string &value);
string SampleExpression(const Expression &expression);
void SamplingSQL(LogicalOperator &op, Json &entry);

} // namespace optimizer_detail
} // namespace semantic_extension
} // namespace duckdb
