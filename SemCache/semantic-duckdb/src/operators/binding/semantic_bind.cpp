// Bind-time validation prevents a partially executed query from discovering a
// plan/schema/model mismatch only after model requests have been issued.
#include "operators/binding/semantic_bind.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {
namespace semantic_extension {

unique_ptr<FunctionData> SemanticBindData::Copy() const {
	auto result = make_uniq<SemanticBindData>();
	result->execution_context = execution_context;
	result->node = node;
	result->models = models;
	result->query_ir_json = query_ir_json;
	result->decision_json = decision_json;
	result->sampling = sampling;
	result->result_cache_probe = result_cache_probe;
	result->llm_request_limit = llm_request_limit;
	return std::move(result);
}

bool SemanticBindData::Equals(const FunctionData &other) const {
	auto &rhs = other.Cast<SemanticBindData>();
	return models == rhs.models && node.kind == rhs.node.kind &&
	       node.semantic.task_description == rhs.node.semantic.task_description &&
	       node.semantic.operator_id == rhs.node.semantic.operator_id && decision_json == rhs.decision_json &&
	       sampling == rhs.sampling && result_cache_probe == rhs.result_cache_probe &&
	       llm_request_limit == rhs.llm_request_limit;
}

bool SemanticBindData::SupportStatementCache() const {
	// Settings and external optimizer decisions may change between executions;
	// caching this bind data would silently reuse stale physical choices.
	return false;
}

string GetSetting(ClientContext &context, const string &name) {
	Value value;
	if (!context.TryGetCurrentSetting(name, value) || value.IsNull())
		throw InvalidInputException("Missing setting %s", name);
	return value.ToString();
}

void ValidateExecutionSettings(ClientContext &context) {
	auto parallelism = std::stoll(GetSetting(context, "semantic_parallelism"));
	if (parallelism < 1 || parallelism > 1024)
		throw InvalidInputException("semantic_parallelism must be between 1 and 1024");
	auto per_model = std::stoll(GetSetting(context, "semantic_max_concurrent_requests_per_model"));
	if (per_model < 1 || per_model > 1024)
		throw InvalidInputException("semantic_max_concurrent_requests_per_model must be between 1 and 1024");
}

LogicalType ToLogicalType(semantic::FieldType type) {
	switch (type) {
	case semantic::FieldType::BOOLEAN:
		return LogicalType::BOOLEAN;
	case semantic::FieldType::INTEGER:
		return LogicalType::INTEGER;
	case semantic::FieldType::BIGINT:
		return LogicalType::BIGINT;
	case semantic::FieldType::DOUBLE:
		return LogicalType::DOUBLE;
	case semantic::FieldType::VARCHAR:
		return LogicalType::VARCHAR;
	case semantic::FieldType::LIST:
	case semantic::FieldType::STRUCT:
		break;
	}
	throw InvalidInputException("Unsupported semantic type");
}

LogicalType ToLogicalType(const semantic::Field &field) {
	if (field.type == semantic::FieldType::LIST)
		return LogicalType::LIST(ToLogicalType(field.children.at(0)));
	if (field.type == semantic::FieldType::STRUCT) {
		child_list_t<LogicalType> children;
		for (const auto &child : field.children)
			children.emplace_back(child.name, ToLogicalType(child));
		return LogicalType::STRUCT(std::move(children));
	}
	return ToLogicalType(field.type);
}

void ExecuteUnloweredMarker(DataChunk &, ExpressionState &, Vector &) {
	throw InvalidInputException(
	    "Semantic marker was not lowered; enable the optimizer and use supported SQL placement");
}

}
}
