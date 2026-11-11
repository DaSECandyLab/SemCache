#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace semcache::sampling {
using Json = nlohmann::ordered_json;
// A binary subtype is an internal typed scalar, impossible to collide with a
// user's legal SQL STRUCT/LIST/object. Prompt/SQL serialization emits its exact
// decimal integer and boundary exports recover a Python arbitrary-size int.
constexpr std::uint64_t integer_subtype = 0x53454d494e54;
inline bool exact_integer(const Json &value) {
  return value.is_binary() && value.get_binary().has_subtype() &&
         value.get_binary().subtype() == integer_subtype;
}
inline Json make_integer(const std::string &text) {
  return Json::binary(std::vector<std::uint8_t>(text.begin(), text.end()),
                      integer_subtype);
}
inline std::string integer_text(const Json &value) {
  const auto &bytes = value.get_binary();
  return {bytes.begin(), bytes.end()};
}
}
