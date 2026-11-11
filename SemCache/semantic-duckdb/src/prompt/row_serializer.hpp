#pragma once

// Loss-aware intermediate values used to serialize DuckDB rows deterministically
// without relying on locale-sensitive stream formatting.

#include <cstdint>
#include <map>
#include <string>

namespace semantic {
class RowValue {
public:
	enum class Kind { NULL_VALUE, BOOLEAN, INTEGER, DOUBLE, STRING, JSON };
	RowValue();
	explicit RowValue(bool value);
	explicit RowValue(int64_t value);
	explicit RowValue(double value);
	explicit RowValue(std::string value);
	explicit RowValue(const char *value);
	std::string Serialize() const;
	static RowValue FromJSON(const std::string &json);

private:
	Kind kind;
	bool boolean_value = false;
	int64_t integer_value = 0;
	double double_value = 0;
	std::string string_value;
};
using Row = std::map<std::string, RowValue>;
void ValidateUTF8(const std::string &text);
// Strict UTF-8 JSON string encoding, with < and > escaped to protect block labels.
std::string QuoteText(const std::string &text);
}
