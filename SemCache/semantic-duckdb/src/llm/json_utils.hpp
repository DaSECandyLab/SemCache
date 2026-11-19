#pragma once


#include <set>
#include <string>
#include "nlohmann/json.hpp"
#include "plan/semantic_config.hpp"

namespace semantic {
namespace llm_detail {

using Json = nlohmann::json;

inline Json LayoutJSON(const PromptLayout &layout, bool pair) {
	Json result {{"block_order", Json::array()}};
	for (auto block : layout.block_order)
		result["block_order"].push_back(BlockName(block));
	if (pair) {
		result["left_field_order"] = layout.left_field_order;
		result["right_field_order"] = layout.right_field_order;
	} else {
		result["field_order"] = layout.field_order;
	}
	return result;
}

inline Json ParseJSON(const std::string &text) {
	if (text.size() > 16 * 1024 * 1024)
		throw ValidationError("JSON exceeds 16 MiB");

	std::vector<std::set<std::string>> keys;
	return Json::parse(text, [&](int depth, Json::parse_event_t event, Json &value) {
		if (depth > 128)
			throw ValidationError("JSON nesting exceeds 128");
		if (event == Json::parse_event_t::object_start)
			keys.emplace_back();
		else if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
			throw ValidationError("duplicate JSON key");
		else if (event == Json::parse_event_t::object_end)
			keys.pop_back();
		return true;
	});
}

}
}
