#pragma once

// Semantic configuration types are independent of DuckDB execution objects.

#include "prompt/prompt_layout.hpp"
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace semantic {
class ValidationError : public std::runtime_error {
public:
	explicit ValidationError(const std::string &message) : std::runtime_error(message) {
	}
};
enum class NodeKind { SEMANTIC_FILTER, SEMANTIC_PROJECTION, SEMANTIC_JOIN };
enum class FieldType { BOOLEAN, INTEGER, BIGINT, DOUBLE, VARCHAR, LIST, STRUCT };
struct Field {
	std::string name;
	FieldType type;
	std::vector<Field> children;
	Field(std::string name, FieldType type, std::vector<Field> children = {})
	    : name(std::move(name)), type(type), children(std::move(children)) {
	}
};
using Schema = std::vector<Field>;
struct OutputSchema {
	enum class Kind { BOOLEAN, OBJECT };
	Kind kind = Kind::BOOLEAN;
	Schema fields;
};
struct SemanticConfig {
	int64_t operator_id = -1;
	std::string model_id;
	std::string task_description;
	OutputSchema output_schema;
	PromptLayout layout;
	// Logical input order fixes placeholder identities, independently of layout.
	FieldOrder input_fields;
	FieldOrder left_input_fields;
	FieldOrder right_input_fields;
};
struct SemanticOperator {
	NodeKind kind = NodeKind::SEMANTIC_FILTER;
	SemanticConfig semantic;
};
struct SemanticExecutionContext {
	std::string query_id;
	std::string global_system_prefix;
	std::vector<SemanticOperator> operators;
};
std::string TypeName(FieldType type);
std::string TypeName(const Field &field);
std::string OperatorName(NodeKind kind);
void ValidateSemanticConfig(NodeKind kind, const SemanticConfig &config);
void ValidateOutputSchema(const OutputSchema &output);
}
