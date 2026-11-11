#pragma once

// Binder-owned configuration copied into the custom logical and physical
// operators. It bridges bound SQL expressions and validated semantic configuration.

#include "duckdb/function/scalar_function.hpp"
#include "plan/semantic_config.hpp"

namespace duckdb {
class LogicalOperator;
class BoundFunctionExpression;
namespace semantic_extension {

struct SemanticBindData : FunctionData {
	semantic::SemanticExecutionContext execution_context;
	semantic::SemanticOperator node;

	string models;
	string query_ir_json;

	string decision_json;
	bool sampling = false;
	bool result_cache_probe = false;
	uint64_t llm_request_limit = UINT64_MAX;

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
	bool SupportStatementCache() const override;
};

string GetSetting(ClientContext &context, const string &name);

void ValidateExecutionSettings(ClientContext &context);

LogicalType ToLogicalType(semantic::FieldType type);
LogicalType ToLogicalType(const semantic::Field &field);

void ExecuteUnloweredMarker(DataChunk &input, ExpressionState &state, Vector &result);

unique_ptr<FunctionData> BindSemanticSQL(ClientContext &context, ScalarFunction &function,
                                         vector<unique_ptr<Expression>> &arguments);

void PrepareSemanticSQL(ClientContext &context, LogicalOperator &child, BoundFunctionExpression &marker);

}
}
