// Semantic invariants are centralized here. Every JSON parser, SQL binder and
// external optimizer decision ultimately passes through these checks.
#include "plan/semantic_config.hpp"
#include "prompt/row_serializer.hpp"

namespace semantic {
std::string TypeName(FieldType type) {
	switch (type) {
	case FieldType::BOOLEAN:
		return "BOOLEAN";
	case FieldType::INTEGER:
		return "INTEGER";
	case FieldType::BIGINT:
		return "BIGINT";
	case FieldType::DOUBLE:
		return "DOUBLE";
	case FieldType::VARCHAR:
		return "VARCHAR";
	case FieldType::LIST:
		return "LIST";
	case FieldType::STRUCT:
		return "STRUCT";
	}
	throw ValidationError("unknown field type");
}
std::string TypeName(const Field &field) {
	if (field.type == FieldType::LIST) {
		if (field.children.size() != 1)
			throw ValidationError("LIST requires exactly one element type");
		return TypeName(field.children[0]) + "[]";
	}
	if (field.type == FieldType::STRUCT) {
		if (field.children.empty())
			throw ValidationError("STRUCT fields must not be empty");
		std::string result = "STRUCT(";
		std::set<std::string> names;
		for (const auto &child : field.children) {
			if (child.name.empty() || !names.insert(child.name).second)
				throw ValidationError("empty or duplicate STRUCT field name");
			if (result.size() > 7)
				result += ", ";
			result += QuoteText(child.name) + " " + TypeName(child);
		}
		return result + ")";
	}
	if (!field.children.empty())
		throw ValidationError("scalar field cannot have children");
	return TypeName(field.type);
}
std::string OperatorName(NodeKind kind) {
	switch (kind) {
	case NodeKind::SEMANTIC_FILTER:
		return "Semantic Filter";
	case NodeKind::SEMANTIC_PROJECTION:
		return "Semantic Projection";
	case NodeKind::SEMANTIC_JOIN:
		return "Semantic Join";
	default:
		throw ValidationError("unsupported semantic node kind");
	}
}
std::string BlockName(PromptBlock block) {
	switch (block) {
	case PromptBlock::OPERATOR_SPECIFICATION:
		return "OPERATOR_SPECIFICATION";
	case PromptBlock::TASK_DESCRIPTION:
		return "TASK_DESCRIPTION";
	case PromptBlock::DATA_BLOCK:
		return "DATA_BLOCK";
	}
	throw ValidationError("unknown prompt block");
}
static void CheckText(const std::string &text, const std::string &label, bool allow_empty = false) {
	if (!allow_empty && text.empty()) {
		throw ValidationError(label + " must not be empty");
	}
	QuoteText(text); // Also reject invalid UTF-8 in programmatically constructed plans.
}
static void CheckOrder(const FieldOrder &order) {
	if (order.empty()) {
		throw ValidationError("field order must not be empty");
	}
	std::set<std::string> seen;
	for (const auto &field : order) {
		CheckText(field, "field name");
		if (!seen.insert(field).second) {
			throw ValidationError("duplicate field in order: " + field);
		}
	}
}
void ValidateLayout(const PromptLayout &layout, bool pair) {
	std::set<PromptBlock> seen;
	for (auto block : layout.block_order) {
		BlockName(block);
		if (!seen.insert(block).second) {
			throw ValidationError("duplicate prompt block");
		}
	}
	if (seen.size() != 3) {
		throw ValidationError("block_order must contain exactly three prompt blocks");
	}
	if (pair) {
		if (!layout.field_order.empty()) {
			throw ValidationError("join requires separate left/right field orders");
		}
		CheckOrder(layout.left_field_order);
		CheckOrder(layout.right_field_order);
	} else {
		if (!layout.left_field_order.empty() || !layout.right_field_order.empty()) {
			throw ValidationError("unary layout cannot contain pair field orders");
		}
		CheckOrder(layout.field_order);
	}
}
static void CheckSchema(const Schema &fields) {
	if (fields.empty()) {
		throw ValidationError("schema fields must not be empty");
	}
	std::set<std::string> names;
	for (const auto &field : fields) {
		CheckText(field.name, "schema field name");
		TypeName(field);
		if (!names.insert(field.name).second) {
			throw ValidationError("duplicate schema field: " + field.name);
		}
	}
}
void ValidateOutputSchema(const OutputSchema &output) {
	if (output.kind == OutputSchema::Kind::BOOLEAN) {
		if (!output.fields.empty())
			throw ValidationError("filter/join output_schema must be boolean without fields");
		return;
	}
	if (output.fields.size() != 1)
		throw ValidationError("semantic projection requires exactly one output field");
	CheckSchema(output.fields);
	if (output.fields[0].type == FieldType::LIST || output.fields[0].type == FieldType::STRUCT)
		throw ValidationError("semantic projection requires a scalar output field");
}
void ValidateSemanticConfig(NodeKind kind, const SemanticConfig &config) {
	OperatorName(kind);
	if (config.operator_id < 0) {
		throw ValidationError("operator_id must be a nonnegative int64");
	}
	CheckText(config.model_id, "model_id");
	CheckText(config.task_description, "task_description");
	ValidateLayout(config.layout, kind == NodeKind::SEMANTIC_JOIN);
	auto inputs = [&](const FieldOrder &order, const FieldOrder &fields) {
		if (!fields.empty())
			CheckOrder(fields);
		if (std::set<std::string>(order.begin(), order.end()) != std::set<std::string>(fields.begin(), fields.end()))
			throw ValidationError("input fields must exactly match layout field order");
	};
	inputs(config.layout.field_order, config.input_fields);
	inputs(config.layout.left_field_order, config.left_input_fields);
	inputs(config.layout.right_field_order, config.right_input_fields);
	const auto count = config.input_fields.size() + config.left_input_fields.size() + config.right_input_fields.size();
	std::set<size_t> referenced;
	const auto &task = config.task_description;
	for (size_t i = 0; i < task.size(); ++i) {
		if (task[i] == '}')
			throw ValidationError("unmatched closing placeholder brace");
		if (task[i] != '{')
			continue;
		size_t start = ++i, index = 0;
		while (i < task.size() && task[i] >= '0' && task[i] <= '9') {
			index = index * 10 + size_t(task[i++] - '0');
			if (index >= count)
				throw ValidationError("placeholder references nonexistent input");
		}
		if (i == start || i == task.size() || task[i] != '}' || (i > start + 1 && task[start] == '0'))
			throw ValidationError("invalid positional placeholder");
		referenced.insert(index);
	}
	if (referenced.size() != count)
		throw ValidationError("every input expression must be referenced by a placeholder");
	const auto &output = config.output_schema;
	if (kind == NodeKind::SEMANTIC_PROJECTION) {
		if (output.kind != OutputSchema::Kind::OBJECT) {
			throw ValidationError("projection output_schema must be an object");
		}
		ValidateOutputSchema(output);
	} else if (output.kind != OutputSchema::Kind::BOOLEAN || !output.fields.empty()) {
		throw ValidationError("filter/join output_schema must be boolean without fields");
	}
}
}
