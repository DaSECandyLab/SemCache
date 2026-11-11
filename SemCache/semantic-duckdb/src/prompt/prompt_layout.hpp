#pragma once

// Physical prompt choices selected by the optimizer. Field arrays contain
// bound logical names; display indices are assigned later by PromptRenderer.

#include <string>
#include <vector>

namespace semantic {
enum class PromptBlock { OPERATOR_SPECIFICATION, TASK_DESCRIPTION, DATA_BLOCK };
using FieldOrder = std::vector<std::string>;
struct PromptLayout {
	std::vector<PromptBlock> block_order;
	FieldOrder field_order;
	FieldOrder left_field_order;
	FieldOrder right_field_order;
};
std::string BlockName(PromptBlock block);
void ValidateLayout(const PromptLayout &layout, bool pair);
}
