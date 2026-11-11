#pragma once

// Turns validated semantic configuration plus one row/pair into the exact
// request text. display_to_input makes every displayed index auditable.

#include "plan/semantic_config.hpp"
#include "prompt/row_serializer.hpp"

namespace semantic {
// Shared across all queries/operators/rows in serialization v4.
const std::string &GlobalSystemPrefix();
std::string OperatorSpecification(NodeKind kind);
struct RenderedPrompt {
	std::string system_text;
	std::string user_text;
	std::string canonical_text_for_hashing;
	// Versioned binary identity of operator kind, protocol, displayed task and data.
	std::string semantic_cache_key;
	// Display placeholder index -> original logical input expression index.
	std::vector<size_t> display_to_input;
};
class PromptRenderer {
public:
	// Own a validated, immutable template; reuse it until the configuration changes.
	PromptRenderer(const std::string &global_system_prefix, const SemanticOperator &node);
	RenderedPrompt RenderRow(const Row &row, bool include_hash_text = true, bool include_cache_key = false) const;
	RenderedPrompt RenderPair(const Row &left, const Row &right, bool include_hash_text = true,
	                          bool include_cache_key = false) const;

private:
	bool pair;
	std::string system, before_data, after_data, cache_prefix;
	FieldOrder left_order, right_order;
	std::vector<size_t> mapping;
	RenderedPrompt Render(const std::string &data, bool include_hash_text, bool include_cache_key = false) const;
};
}
