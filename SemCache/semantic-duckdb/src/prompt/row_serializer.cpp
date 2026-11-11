// Canonical JSON-scalar serialization is part of the model input and cache key;
// changes here therefore require an explicit prompt serialization version bump.
#include "prompt/row_serializer.hpp"
#include "plan/semantic_config.hpp"
#include "nlohmann/json.hpp"
#include <cmath>
#include <limits>
#include <utility>

namespace semantic {
RowValue::RowValue() : kind(Kind::NULL_VALUE) {
}
RowValue::RowValue(bool value) : kind(Kind::BOOLEAN), boolean_value(value) {
}
RowValue::RowValue(int64_t value) : kind(Kind::INTEGER), integer_value(value) {
}
RowValue::RowValue(double value) : kind(Kind::DOUBLE), double_value(value) {
}
RowValue::RowValue(std::string value) : kind(Kind::STRING), string_value(std::move(value)) {
}
RowValue::RowValue(const char *value) : kind(Kind::STRING) {
	if (!value) {
		throw ValidationError("null string pointer; use RowValue() for NULL");
	}
	string_value = value;
}
void ValidateUTF8(const std::string &text) {
	for (size_t offset = 0; offset < text.size();) {
		const auto first = static_cast<unsigned char>(text[offset++]);
		if (first < 0x80)
			continue;
		const size_t length = first >= 0xC2 && first <= 0xDF   ? 2
		                      : first >= 0xE0 && first <= 0xEF ? 3
		                      : first >= 0xF0 && first <= 0xF4 ? 4
		                                                       : 0;
		if (!length || length - 1 > text.size() - offset)
			throw ValidationError("invalid UTF-8 text: invalid leading byte or truncated character");
		uint32_t code = first & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
		for (size_t i = 1; i < length; ++i) {
			const auto byte = static_cast<unsigned char>(text[offset++]);
			if ((byte & 0xC0) != 0x80)
				throw ValidationError("invalid UTF-8 text: invalid continuation byte");
			code = (code << 6) | (byte & 0x3F);
		}
		if ((length == 2 && code < 0x80) || (length == 3 && code < 0x800) || (length == 4 && code < 0x10000) ||
		    code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF))
			throw ValidationError("invalid UTF-8 text: noncanonical character");
	}
}
std::string QuoteText(const std::string &text) {
	try {
		auto json = nlohmann::json(text).dump();
		auto next = json.find_first_of("<>");
		if (next == std::string::npos)
			return json;
		std::string result;
		result.reserve(json.size());
		size_t start = 0;
		while (next != std::string::npos) {
			result.append(json, start, next - start);
			result += json[next] == '<' ? "\\u003c" : "\\u003e";
			start = next + 1;
			next = json.find_first_of("<>", start);
		}
		result.append(json, start, std::string::npos);
		return result;
	} catch (const nlohmann::json::exception &error) {
		throw ValidationError(std::string("invalid UTF-8 text: ") + error.what());
	}
}
namespace {
RowValue ScalarFromJSON(const nlohmann::json &json) {
	if (json.is_null())
		return RowValue();
	if (json.is_boolean())
		return RowValue(json.get<bool>());
	if (json.is_number_integer()) {
		if (json.is_number_unsigned() && json.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))
			throw ValidationError("row JSON integer exceeds BIGINT range");
		return RowValue(json.get<int64_t>());
	}
	if (json.is_number())
		return RowValue(json.get<double>());
	if (json.is_string())
		return RowValue(json.get<std::string>());
	throw ValidationError("expected JSON scalar");
}
std::string SerializeJSON(const nlohmann::json &json) {
	if (!json.is_structured())
		return ScalarFromJSON(json).Serialize();
	std::string encoded;
	if (json.is_array()) {
		encoded = "[";
		for (const auto &item : json) {
			if (encoded.size() > 1)
				encoded += ",";
			encoded += SerializeJSON(item);
		}
		encoded += "]";
	} else {
		encoded = "{";
		for (auto it = json.begin(); it != json.end(); ++it) {
			if (encoded.size() > 1)
				encoded += ",";
			encoded += QuoteText(it.key()) + ":" + SerializeJSON(it.value());
		}
		encoded += "}";
	}
	return encoded;
}
}
RowValue RowValue::FromJSON(const std::string &text) {
	const auto json = nlohmann::json::parse(text);
	if (!json.is_structured())
		return ScalarFromJSON(json);
	RowValue result;
	result.kind = Kind::JSON;
	result.string_value = SerializeJSON(json);
	return result;
}
std::string RowValue::Serialize() const {
	switch (kind) {
	case Kind::NULL_VALUE:
		return "null";
	case Kind::BOOLEAN:
		return boolean_value ? "true" : "false";
	case Kind::INTEGER:
		return nlohmann::json(integer_value).dump();
	case Kind::DOUBLE:
		if (!std::isfinite(double_value)) {
			throw ValidationError("non-finite row number");
		}
		return nlohmann::json(double_value).dump();
	case Kind::STRING:
		return QuoteText(string_value);
	case Kind::JSON:
		return string_value;
	}
	throw ValidationError("unknown row value kind");
}
}
