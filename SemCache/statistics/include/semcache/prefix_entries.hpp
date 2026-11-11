#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace semcache::statistics {
using PrefixHash = std::array<unsigned char, 32>;
inline constexpr const char *prefix_entry_encoding = "sha256";

inline std::string encode_prefix_hash(const PrefixHash &hash) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string text(64, '0');
  for (std::size_t i = 0; i < hash.size(); ++i) {
    text[2 * i] = digits[hash[i] >> 4];
    text[2 * i + 1] = digits[hash[i] & 15];
  }
  return text;
}
inline PrefixHash decode_prefix_hash(std::string_view text) {
  if (text.size() != 64)
    throw std::invalid_argument("Prefix hash must have 64 hexadecimal digits");
  auto digit = [](unsigned char c) -> unsigned char {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    throw std::invalid_argument("Invalid prefix hash encoding");
  };
  PrefixHash result;
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = (digit(text[2 * i]) << 4) | digit(text[2 * i + 1]);
  return result;
}
template <class Json> bool hashed_prefix_entries(const Json &value) {
  const auto encoding = value.find("entry_encoding");
  if (encoding == value.end() || *encoding != prefix_entry_encoding)
    throw std::invalid_argument("Prefix entries require sha256 encoding; rebuild statistics");
  return true;
}
template <class Json>
std::map<PrefixHash, std::uint64_t> prefix_counts(const Json &sketch) {
  hashed_prefix_entries(sketch);
  std::map<PrefixHash, std::uint64_t> result;
  for (auto entry = sketch.at("entries").begin();
       entry != sketch.at("entries").end(); ++entry) {
    if (!entry.value().is_number_integer() || entry.value() <= 0)
      throw std::invalid_argument("Frequency must be positive");
    const auto hash = decode_prefix_hash(entry.key());
    if (!result.emplace(hash, entry.value().template get<std::uint64_t>()).second)
      throw std::invalid_argument("Duplicate prefix hash");
  }
  return result;
}
template <class Json>
std::map<PrefixHash, double> prefix_common_counts(const Json &common,
                                                const std::string &depth) {
  std::map<PrefixHash, double> result;
  if (!common.contains(depth))
    return result;
  hashed_prefix_entries(common);
  for (auto entry = common.at(depth).begin(); entry != common.at(depth).end();
       ++entry)
    result.emplace(decode_prefix_hash(entry.key()),
                   entry.value().template get<double>());
  return result;
}
}
