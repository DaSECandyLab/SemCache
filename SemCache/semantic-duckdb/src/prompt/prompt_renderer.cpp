// Prompt serialization v4 keeps displayed field numbers dense while preserving
// logical binding identity by applying the inverse permutation to task references.
#include "prompt/prompt_renderer.hpp"
#include <algorithm>

namespace semantic {
const std::string &GlobalSystemPrefix() {
	static const std::string prefix =
	    "You are a factual reasoning assistant.\n"
	    "OPERATOR_SPECIFICATION defines the operation, TASK_DESCRIPTION states the task, and DATA_BLOCK contains the "
	    "input data.\n"
	    "Each {number} in TASK_DESCRIPTION refers to the value with the same label in DATA_BLOCK.";
	return prefix;
}
std::string OperatorSpecification(NodeKind kind) {
	switch (kind) {
	case NodeKind::SEMANTIC_FILTER:
	case NodeKind::SEMANTIC_JOIN:
		return "Your task is to determine whether the Predicate in TASK_DESCRIPTION is true.\n"
		       "Output exactly one word: true or false.\n"
		       "Do not output explanations.";
	case NodeKind::SEMANTIC_PROJECTION:
		return "Your task is to process the value requested by TASK_DESCRIPTION from DATA_BLOCK.\n"
		       "The required output type is specified in OUTPUT_TYPE.\n"
		       "For text results, return a short sentence of at most 30 characters, unless the TASK_DESCRIPTION "
		       "specifies requirements for detail or length.";
	default:
		throw ValidationError("unsupported semantic node kind");
	}
}
namespace {
void CachePart(std::string &key, const std::string &value) {
	const uint64_t size = value.size();
	for (size_t index = 0; index < sizeof(size); ++index)
		key += static_cast<char>((size >> (index * 8)) & 255);
	key += value;
}

std::string Wrap(const std::string &label, const std::string &body) {
	return "<" + label + ">\n" + body + "\n</" + label + ">";
}
std::string Data(const Row &row, const FieldOrder &order, size_t offset = 0) {
	std::string result;
	for (size_t i = 0; i < order.size(); ++i) {
		const auto &name = order[i];
		if (!result.empty())
			result += '\n';
		auto field = row.find(name);
		if (field == row.end())
			throw ValidationError("missing row field: " + name);
		result += "{" + std::to_string(offset + i) + "}: " + field->second.Serialize();
	}
	return result;
}
std::string RebindTask(const std::string &task, const std::vector<size_t> &display_to_input) {
	std::vector<size_t> input_to_display(display_to_input.size());
	for (size_t display = 0; display < display_to_input.size(); ++display)
		input_to_display.at(display_to_input[display]) = display;
	std::string result;
	// ValidateSemanticConfig has checked all braces/indices. Replace in one pass
	// over the original template: chained substitutions would corrupt swaps and
	// multi-digit indices. Never rewrite placeholders inside actual row values.
	for (size_t i = 0; i < task.size(); ++i) {
		if (task[i] != '{') {
			result += task[i];
			continue;
		}
		size_t input = 0;
		while (task[++i] != '}')
			input = input * 10 + size_t(task[i] - '0');
		result += "{" + std::to_string(input_to_display.at(input)) + "}";
	}
	return result;
}
}

PromptRenderer::PromptRenderer(const std::string &prefix, const SemanticOperator &node)
    : pair(node.kind == NodeKind::SEMANTIC_JOIN), system(prefix) {
	ValidateSemanticConfig(node.kind, node.semantic);
	if (system != GlobalSystemPrefix())
		throw ValidationError("global_system_prefix must match the shared system prefix");
	left_order = pair ? node.semantic.layout.left_field_order : node.semantic.layout.field_order;
	right_order = node.semantic.layout.right_field_order;
	auto bind = [&](const FieldOrder &inputs, const FieldOrder &order, size_t offset) {
		for (const auto &name : order)
			mapping.push_back(offset + size_t(std::find(inputs.begin(), inputs.end(), name) - inputs.begin()));
	};
	bind(pair ? node.semantic.left_input_fields : node.semantic.input_fields, left_order, 0);
	if (pair)
		bind(node.semantic.right_input_fields, right_order, left_order.size());
	const auto task = RebindTask(node.semantic.task_description, mapping);
	const auto specification = OperatorSpecification(node.kind);
	const auto output_type =
	    node.kind == NodeKind::SEMANTIC_PROJECTION ? TypeName(node.semantic.output_schema.fields.at(0)) : std::string();
	cache_prefix = "semcache-semantic-v3";
	CachePart(cache_prefix, std::to_string(int(node.kind)));
	CachePart(cache_prefix, task);
	CachePart(cache_prefix, specification);
	CachePart(cache_prefix, output_type);

	auto *text = &before_data;
	for (auto block : node.semantic.layout.block_order) {
		if (!text->empty())
			*text += "\n\n";
		if (block == PromptBlock::DATA_BLOCK) {
			*text += "<DATA_BLOCK>\n";
			after_data = "\n</DATA_BLOCK>";
			text = &after_data;
		} else {
			*text += Wrap(BlockName(block), block == PromptBlock::TASK_DESCRIPTION ? task : specification);
		}
	}
	if (!output_type.empty())
		after_data += "\n\n" + Wrap("OUTPUT_TYPE", "Return a value of DuckDB type " + output_type + ".");
}

RenderedPrompt PromptRenderer::Render(const std::string &data, bool include_hash_text, bool include_cache_key) const {
	RenderedPrompt result;
	if (include_cache_key) {
		result.semantic_cache_key = cache_prefix;
		CachePart(result.semantic_cache_key, "single");
		CachePart(result.semantic_cache_key, data);
	}
	result.display_to_input = mapping;
	result.system_text = system;
	result.user_text = before_data + data + after_data;
	if (include_hash_text)
		result.canonical_text_for_hashing = "semantic-prompt-v4\nsystem:" + std::to_string(system.size()) + "\n" +
		                                    system + "\nuser:" + std::to_string(result.user_text.size()) + "\n" +
		                                    result.user_text;
	return result;
}
RenderedPrompt PromptRenderer::RenderRow(const Row &row, bool include_hash_text, bool include_cache_key) const {
	if (pair)
		throw ValidationError("RenderRow requires filter or projection configuration");
	return Render(Data(row, left_order), include_hash_text, include_cache_key);
}
RenderedPrompt PromptRenderer::RenderPair(const Row &left, const Row &right, bool include_hash_text,
                                          bool include_cache_key) const {
	if (!pair)
		throw ValidationError("RenderPair requires join configuration");
	const auto data = Data(left, left_order) + "\n" + Data(right, right_order, left_order.size());
	return Render(data, include_hash_text, include_cache_key);
}

}
