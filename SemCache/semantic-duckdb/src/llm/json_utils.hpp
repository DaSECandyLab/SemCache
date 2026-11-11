#pragma once


#include <set>
#include <cctype>
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

// Some OpenAI-compatible servers return a valid structured value followed by
// a short generation artifact such as "\n\nW". Keep ParseJSON strict for
// plans and configuration; the HTTP structured-output boundary may isolate the
// first complete value before applying the normal schema/type checks.
inline std::string FirstJSONValue(const std::string &text) {
	std::size_t start = 0;
	while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])))
		++start;
	if (start == text.size())
		return {};
	const char opening = text[start];
	if (opening != '{' && opening != '[' && opening != '"') {
		std::size_t end = start;
		while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end])))
			++end;
		return text.substr(start, end - start);
	}
	if (opening == '"') {
		bool escaped = false;
		for (std::size_t i = start + 1; i < text.size(); ++i) {
			const char c = text[i];
			if (escaped) {
				escaped = false;
				continue;
			}
			if (c == '\\')
				escaped = true;
			else if (c == '"')
				return text.substr(start, i - start + 1);
		}
		return text;
	}
	int depth = 0;
	bool in_string = false, escaped = false;
	for (std::size_t i = start; i < text.size(); ++i) {
		const char c = text[i];
		if (in_string) {
			if (escaped)
				escaped = false;
			else if (c == '\\')
				escaped = true;
			else if (c == '"')
				in_string = false;
			continue;
		}
		if (c == '"')
			in_string = true;
		else if (c == '{' || c == '[')
			++depth;
		else if (c == '}' || c == ']') {
			if (--depth == 0)
				return text.substr(start, i - start + 1);
		}
	}
	return text;
}

}
}
