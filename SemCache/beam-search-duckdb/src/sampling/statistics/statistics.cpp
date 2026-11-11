#include "statistics.hpp"
#include "joint_presence.hpp"
#include "semcache/prefix_reuse.hpp"
#include "../common/exact_integer.hpp"
#include "semcache/prefix_sketch.h"
#include "semcache/prefix_entries.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mbedtls/sha256.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace semcache::sampling {
namespace {
using Hash = semcache::statistics::PrefixHash;
Hash rank(std::string_view value) {
  Hash h;
  if (mbedtls_sha256(reinterpret_cast<const unsigned char *>(value.data()),
                     value.size(), h.data(), 0))
    throw std::runtime_error("SHA-256 failed");
  return h;
}
std::string float_text(double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("Nonfinite JSON number");
  if (value == 0)
    return std::signbit(value) ? "-0.0" : "0.0";
  char buffer[128];
  auto result =
      std::to_chars(buffer, buffer + 128, value, std::chars_format::general);
  if (result.ec != std::errc())
    throw std::runtime_error("Could not serialize float");
  std::string text(buffer, result.ptr), sign;
  if (text[0] == '-') {
    sign = "-";
    text.erase(0, 1);
  }
  auto e = text.find_first_of("eE");
  int exponent = 0;
  if (e != std::string::npos) {
    exponent = std::stoi(text.substr(e + 1));
    text.resize(e);
  }
  auto dot = text.find('.');
  if (dot == std::string::npos)
    dot = text.size();
  exponent += static_cast<int>(dot) - 1;
  text.erase(std::remove(text.begin(), text.end(), '.'), text.end());
  while (text.size() > 1 && text.back() == '0')
    text.pop_back();
  if (exponent >= -4 && exponent < 16) {
    int point = exponent + 1;
    if (point <= 0)
      text = "0." + std::string(-point, '0') + text;
    else if (static_cast<std::size_t>(point) >= text.size())
      text += std::string(point - text.size(), '0') + ".0";
    else
      text.insert(point, ".");
  } else {
    if (text.size() > 1)
      text.insert(1, ".");
    auto n = std::to_string(std::abs(exponent));
    if (n.size() < 2)
      n = "0" + n;
    text += "e" + std::string(exponent < 0 ? "-" : "+") + n;
  }
  return sign + text;
}
std::string encode(const Json &value, bool sorted = true) {
  if (exact_integer(value))
    return integer_text(value);
  if (value.is_number_float())
    return float_text(value.get<double>());
  if (value.is_array()) {
    std::string out = "[";
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (i)
        out += ",";
      out += encode(value[i], sorted);
    }
    return out + "]";
  }
  if (value.is_object()) {
    std::vector<std::string> names;
    for (auto it = value.begin(); it != value.end(); ++it)
      names.push_back(it.key());
    if (sorted)
      std::sort(names.begin(), names.end());
    std::string out = "{";
    for (auto &name : names) {
      if (out.size() > 1)
        out += ",";
      out += Json(name).dump() + ":" + encode(value.at(name), sorted);
    }
    return out + "}";
  }
  return value.dump();
}
struct Sum {
  double hi = 0, lo = 0;
  void add(double x) {
    double t = hi + x;
    lo += std::abs(hi) >= std::abs(x) ? (hi - t) + x : (x - t) + hi;
    hi = t;
  }
  double get() const { return lo && std::isfinite(lo) ? hi + lo : hi; }
};
double present(double a, double count) {
  return a == 1 ? 1 : -std::expm1(count * std::log1p(-a));
}
double integral(const std::map<std::size_t, double> &values) {
  if (values.empty())
    return 0;
  auto before = values.begin();
  double total = before->second;
  for (auto it = std::next(before); it != values.end(); ++it) {
    double gap = it->first - before->first;
    total += (gap - 1) * before->second / 2 + (gap + 1) * it->second / 2;
    before = it;
  }
  return total;
}
bool compatible(const Json &a, const Json &b) {
  if (a.at("metadata").at("serializer") != b.at("metadata").at("serializer"))
    return false;
  for (auto it = a.at("sketches").begin(); it != a.at("sketches").end(); ++it)
    if (b.at("sketches").contains(it.key()))
      return true;
  return false;
}
void validate_summary(const Json &s, bool field_serializer = true) {
  const auto &m = s.at("metadata");
  if (m.at("hash_version") != "sha256-prefix-v1" ||
      m.at("domain") != "characters")
    throw std::invalid_argument("Unsupported sketch version/domain");
  if (field_serializer &&
      m.at("serializer") != "prompt-v4-first-field-value-v1")
    throw std::invalid_argument(
        "Legacy input summary: regenerate conditional profiles");
  for (const auto &sketch : s.at("sketches"))
    semcache::statistics::hashed_prefix_entries(sketch);
}
double fraction(const Hash &h) {
  int lead = 0;
  while (lead < 256 && !((h[lead / 8] >> (7 - lead % 8)) & 1))
    ++lead;
  if (lead == 256)
    return 0;
  std::uint64_t mantissa = 0;
  for (int i = 0; i < 53; ++i) {
    int bit = lead + i;
    mantissa =
        (mantissa << 1) | (bit < 256 ? ((h[bit / 8] >> (7 - bit % 8)) & 1) : 0);
  }
  int guard = lead + 53;
  bool up = guard < 256 && ((h[guard / 8] >> (7 - guard % 8)) & 1),
       sticky = false;
  for (int bit = guard + 1; bit < 256; ++bit)
    sticky |= ((h[bit / 8] >> (7 - bit % 8)) & 1) != 0;
  if (up && (sticky || (mantissa & 1)))
    ++mantissa;
  return std::ldexp(static_cast<double>(mantissa), -lead - 53);
}
struct Sample {
  double tau = 1;
  std::map<Hash, std::uint64_t> counts;
};
Sample sample(const Json &sketch) {
  Sample out;
  out.counts = semcache::statistics::prefix_counts(sketch);
  if (!sketch.at("complete").get<bool>() && !out.counts.empty()) {
    const auto largest = std::prev(out.counts.end());
    const auto threshold = largest->first;
    out.tau = fraction(threshold);
    out.counts.erase(largest);
  }
  return out;
}
struct CrossSample {
  double tau = 1;
  std::map<Hash, std::array<std::uint64_t, 2>> counts;
};
std::size_t cross_capacity(const Json &a, const Json &b) {
  std::size_t capacity = std::numeric_limits<std::size_t>::max();
  for (auto *s : {&a, &b})
    if (!s->at("complete").get<bool>()) {
      const auto &raw = s->at("capacity");
      if (!raw.is_number_integer() || raw <= 0)
        throw std::invalid_argument(
            "Sketch capacity must be a positive integer");
      capacity = std::min(capacity, raw.get<std::size_t>());
    }
  return capacity;
}
template <class Counts>
CrossSample cross_sample_counts(const Counts &left, const Counts &right,
                                std::size_t capacity) {
  // Decode fixed-size digests once, then merge sorted hashes directly. Neither
  // membership nor bottom-k thresholds hash/compare the original prefix text.
  auto l = left.begin(), r = right.begin();
  std::size_t visited = 0;
  CrossSample out;
  while (l != left.end() || r != right.end()) {
    const auto hash = r == right.end() || (l != left.end() && l->first < r->first)
                          ? l->first : r->first;
    if (++visited == capacity) {
      out.tau = fraction(hash);
      break;
    }
    const bool in_left = l != left.end() && l->first == hash,
               in_right = r != right.end() && r->first == hash;
    if (in_left && in_right)
      out.counts.emplace(hash, std::array<std::uint64_t, 2>{l->second, r->second});
    if (in_left)
      ++l;
    if (in_right)
      ++r;
  }
  return out;
}
CrossSample cross_sample(const Json &a, const Json &b) {
  const auto left = semcache::statistics::prefix_counts(a),
             right = semcache::statistics::prefix_counts(b);
  return cross_sample_counts(left, right, cross_capacity(a, b));
}
struct DecodedSketch {
  std::vector<std::pair<Hash, std::uint64_t>> counts;
  double tau = 1.;
  std::size_t sample_size = 0;
};
DecodedSketch decode_sketch(const Json &sketch) {
  semcache::statistics::hashed_prefix_entries(sketch);
  DecodedSketch result;
  const auto &entries = sketch.at("entries");
  result.counts.reserve(entries.size());
  for (auto entry = entries.begin(); entry != entries.end(); ++entry) {
    if (!entry.value().is_number_integer() || entry.value() <= 0)
      throw std::invalid_argument("Frequency must be positive");
    result.counts.emplace_back(
        semcache::statistics::decode_prefix_hash(entry.key()),
        entry.value().get<std::uint64_t>());
  }
  std::sort(result.counts.begin(), result.counts.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  if (std::adjacent_find(result.counts.begin(), result.counts.end(),
                         [](const auto &a, const auto &b) {
                           return a.first == b.first;
                         }) != result.counts.end())
    throw std::invalid_argument("Duplicate prefix hash");
  result.sample_size = result.counts.size();
  if (!sketch.at("complete").get<bool>() && result.sample_size) {
    result.tau = fraction(result.counts.back().first);
    --result.sample_size;
  }
  return result;
}
struct Fields {
  std::vector<std::vector<std::string>> values;
};
int fetch(std::size_t f, const std::size_t *indices, std::size_t count,
          const char **strings, std::size_t *sizes, void *context) {
  try {
    auto &values = static_cast<Fields *>(context)->values.at(f);
    for (std::size_t i = 0; i < count; ++i) {
      auto &s = values.at(indices[i]);
      strings[i] = s.data();
      sizes[i] = s.size();
    }
    return 0;
  } catch (...) {
    return -1;
  }
}
struct VirtualFields {
  const std::vector<NativeFieldValues> &fields;
  std::vector<std::string> batch;
};
int virtual_fetch(std::size_t f, const std::size_t *indices, std::size_t count,
                  const char **strings, std::size_t *sizes, void *context) {
  try {
    auto &source = *static_cast<VirtualFields *>(context);
    source.batch.clear();
    source.batch.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
      source.batch.push_back(source.fields.at(f).value(indices[i]));
    for (std::size_t i = 0; i < count; ++i) {
      strings[i] = source.batch[i].data();
      sizes[i] = source.batch[i].size();
    }
    return 0;
  } catch (...) {
    return -1;
  }
}
Json publish_summaries(void *handle) {
  if (!handle)
    throw std::runtime_error(sem_prefix_error());
  Json result;
  try {
    result = Json::parse(sem_adaptive_json(handle));
  } catch (...) {
    sem_adaptive_destroy(handle);
    throw;
  }
  sem_adaptive_destroy(handle);
  for (auto it = result["summaries"].begin(); it != result["summaries"].end();
       ++it)
    it.value()["metadata"]["serializer"] = "prompt-v4-first-field-value-v1";
  return result;
}
bool number(const Json &v) {
  return v.is_number() && std::isfinite(v.get<double>()) &&
         v.get<double>() >= 0;
}
}
std::string canonical(const Json &value) { return encode(value); }
std::string digest_text(const std::string &value) {
  auto h = rank(value);
  static const char *hex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (auto c : h) {
    out += hex[c >> 4];
    out += hex[c & 15];
  }
  return out;
}
std::string digest(const Json &value) { return digest_text(canonical(value)); }
std::string serialize_value(const Json &value) {
  auto text = encode(value, false);
  std::string result;
  for (char c : text) {
    if (c == '<')
      result += "\\u003c";
    else if (c == '>')
      result += "\\u003e";
    else
      result += c;
  }
  return result;
}
std::size_t character_length(const std::string &value) {
  std::size_t count = 0;
  for (unsigned char c : value)
    if ((c & 0xc0) != 0x80)
      ++count;
  return count;
}
Json build_summaries(const Json &fields, const Json &config) {
  Fields source;
  Json names = Json::array();
  std::vector<std::string> sorted;
  for (auto it = fields.begin(); it != fields.end(); ++it)
    sorted.push_back(it.key());
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::size_t> counts;
  for (auto &name : sorted) {
    names.push_back(name);
    source.values.emplace_back();
    for (auto &value : fields.at(name))
      source.values.back().push_back(value.get<std::string>());
    counts.push_back(source.values.back().size());
  }
  auto c = canonical(config), n = canonical(names);
  void *handle = sem_adaptive_build(c.c_str(), n.c_str(), counts.data(),
                                    counts.size(), fetch, &source);
  return publish_summaries(handle);
}
Json build_summaries(const std::vector<NativeFieldValues> &fields,
                     const Json &config) {
  auto sorted = fields;
  std::sort(sorted.begin(), sorted.end(),
            [](auto &a, auto &b) { return a.name < b.name; });
  VirtualFields source{sorted, {}};
  Json names = Json::array();
  std::vector<std::size_t> counts;
  for (auto &f : sorted) {
    names.push_back(f.name);
    counts.push_back(f.count);
  }
  auto c = canonical(config), n = canonical(names);
  void *handle = sem_adaptive_build(c.c_str(), n.c_str(), counts.data(),
                                    counts.size(), virtual_fetch, &source);
  return publish_summaries(handle);
}
double effective_samples(const std::vector<double> &weights) {
  Sum s, q;
  for (auto w : weights) {
    s.add(w);
    q.add(w * w);
  }
  return q.get() ? s.get() * s.get() / q.get() : 0;
}
Json survival_rate(const Json &alpha, const std::string &depth) {
  Json value = alpha;
  if (alpha.is_object()) {
    // An incomplete Join evidence record may not contain a total yet.
    // Treat that as unknown instead of indexing a partially built object.
    if (!alpha.contains("total"))
      return nullptr;
    value = alpha.at("total");
    if (!depth.empty() && alpha.contains("depths") &&
        alpha["depths"].contains(depth))
      value = alpha["depths"][depth];
  }
  if (value.is_null())
    return nullptr;
  if (!value.is_number())
    return nullptr;
  const auto probability = value.get<double>();
  // A partially observed Join can have a zero denominator.  Its derived
  // ratio is unknown, rather than a failed plan or a valid zero probability.
  // Keep that distinction so cache estimates can fall back to their existing
  // recursive prior without aborting the search.
  if (!std::isfinite(probability))
    return nullptr;
  if (probability < 0 || probability > 1) {
    // Accept only the adjacent representable endpoint roundoff. Larger
    // violations stay unknown instead of being silently made valid.
    if (probability >=
            std::nextafter(0., -std::numeric_limits<double>::infinity()) &&
        probability <=
            std::nextafter(1., std::numeric_limits<double>::infinity()))
      return std::clamp(probability, 0., 1.);
    return nullptr;
  }
  return value;
}
Json self_reuse(const Json &s, const Json &alpha, const Json &requests) {
  validate_summary(s);
  auto r = survival_rate(alpha);
  double count = s.at("count").get<double>();
  if (r.is_null())
    return count ? Json(nullptr) : Json(0.);
  const double distinct = r.get<double>() * count;
  double den = requests.is_null() ? distinct : requests.get<double>();
  if (!den)
    return 0.;
  if (!distinct)
    return nullptr;
  const double scale = den / distinct;
  std::map<std::size_t, double> values;
  for (auto it = s.at("sketches").begin(); it != s.at("sketches").end(); ++it) {
    auto p = sample(it.value());
    auto rate = survival_rate(alpha, it.key());
    if (rate.is_null() && !p.counts.empty())
      return nullptr;
    double a = rate.is_null() ? 0 : rate.get<double>();
    Sum sum;
    for (auto &entry : p.counts)
      sum.add(std::max(
          0., semcache::statistics::repeated_prefix_hits(a, entry.second) +
                  (scale - 1.) * a * entry.second));
    values[std::stoull(it.key())] = sum.get() / p.tau;
  }
  return integral(values) / den;
}
Json cross_reuse(const Json &s, const Json &t, const Json &as, const Json &at,
                 const CrossDependence &dependence, const Json &requests) {
  validate_summary(s, false);
  validate_summary(t);
  if (!compatible(s, t))
    return nullptr;
  auto r = survival_rate(at);
  double count = t.at("count").get<double>();
  if (r.is_null())
    return count ? Json(nullptr) : Json(0.);
  double den =
      requests.is_null() ? r.get<double>() * count : requests.get<double>();
  if (!den)
    return 0.;
  std::map<std::size_t, double> values;
  if (dependence.same_field) {
    for (auto it = t.at("sketches").begin(); it != t.at("sketches").end();
         ++it) {
      const auto sampled = sample(it.value());
      const auto rate = survival_rate(at, it.key());
      if (rate.is_null() && !sampled.counts.empty())
        return nullptr;
      Sum total;
      for (const auto &entry : sampled.counts)
        total.add(
            present(rate.is_null() ? 0. : rate.get<double>(), entry.second));
      values[std::stoull(it.key())] = total.get() / sampled.tau;
    }
    return integral(values) / den;
  }
  for (auto it = s.at("sketches").begin(); it != s.at("sketches").end(); ++it) {
    if (!t["sketches"].contains(it.key()))
      continue;
    auto p = cross_sample(it.value(), t["sketches"][it.key()]);
    auto a = survival_rate(as, it.key()), b = survival_rate(at, it.key());
    if ((a.is_null() || b.is_null()) && !p.counts.empty())
      return nullptr;
    Sum sum;
    const auto overlap = dependence.common
        ? semcache::statistics::prefix_common_counts(*dependence.common, it.key())
        : std::map<Hash, double>{};
    if (!a.is_null() && !b.is_null())
      for (auto &entry : p.counts)
        sum.add(
            dependence.common
                ? joint_presence(
                      a.get<double>(), b.get<double>(), entry.second[0],
                      entry.second[1],
                      overlap.count(entry.first) ? overlap.at(entry.first) : 0.)
                : present(a.get<double>(), entry.second[0]) *
                      present(b.get<double>(), entry.second[1]));
    values[std::stoull(it.key())] = sum.get() / p.tau;
  }
  return integral(values) / den;
}
struct ReuseCache::Impl {
  static std::size_t json_bytes(const Json &value) {
    std::size_t bytes = sizeof(Json) + 32;
    if (value.is_object()) {
      const auto &object = value.get_ref<const Json::object_t &>();
      bytes += object.capacity() * sizeof(Json::object_t::value_type);
      for (auto it = value.begin(); it != value.end(); ++it)
        bytes += it.key().capacity() + 1 + json_bytes(it.value());
    } else if (value.is_array()) {
      bytes += value.get_ref<const Json::array_t &>().capacity() * sizeof(Json);
      for (const auto &entry : value)
        bytes += json_bytes(entry);
    } else if (value.is_string())
      bytes += sizeof(std::string) +
               value.get_ref<const std::string &>().capacity() + 1;
    return bytes;
  }
  // Results belong to the immutable summary/pair bank, so eviction releases
  // both the decoded terms and final values. Key every changing argument by
  // value: equal population handles alone never imply equal survivals/costs.
  struct Results {
    struct Entry {
      Json key, value;
    };
    std::vector<Entry> entries;
    std::size_t next = 0;
    template <class Evaluate>
    Json get(Json key, Evaluate evaluate, std::size_t &hits,
             std::size_t &misses) {
      for (const auto &entry : entries)
        if (entry.key == key) {
          ++hits;
          return entry.value;
        }
      ++misses;
      auto value = evaluate(); // Never memoize exceptions or failed validation.
      // Eight records occupy at most 64 KiB of retained key storage. Large
      // diagnostic payloads are computed normally, not retained indefinitely.
      if (json_bytes(key) > 8192)
        return value;
      Entry entry{std::move(key), value};
      if (entries.size() < 8)
        entries.push_back(std::move(entry));
      else {
        entries[next] = std::move(entry);
        next = (next + 1) % entries.size();
      }
      return value;
    }
  };
  struct Summary {
    std::shared_ptr<const Json> owner;
    // Retain the full bottom-k set, including its threshold item. Self reuse
    // excludes that item, whereas pairwise union thresholds need it. One
    // immutable decoded vector serves both paths and every history source.
    std::map<std::string, std::shared_ptr<const DecodedSketch>> depths;
    std::size_t entries = 0;
    std::uint64_t touched = 0;
    std::size_t bytes = 0;
    Results self_results, same_field_results;
  };
  struct CrossValues {
    std::map<std::string, std::pair<double, std::vector<std::array<double, 3>>>>
        depths;
    Results results;
  };
  struct Cross {
    std::shared_ptr<const Json> source, target, common;
    std::shared_ptr<CrossValues> values;
    std::size_t entries = 0;
    std::uint64_t touched = 0;
    std::size_t bytes = 0;
  };
  std::map<const Json *, Summary> summaries;
  std::map<std::tuple<const Json *, const Json *, const Json *>, Cross> crosses;
  std::size_t summary_entries = 0, cross_entries = 0;
  std::size_t retained_bytes = 0, peak_bytes = 0;
  // Summary and cross entries share immutable JSON owners. Account each owner
  // once instead of charging the full sketch payload to every cross pair.
  struct Owner {
    std::size_t references = 0, bytes = 0;
  };
  std::map<const Json *, Owner> owners;
  using OwnerCosts = std::vector<std::pair<const Json *, std::size_t>>;
  std::uint64_t clock = 0;
  std::size_t result_hits = 0, result_misses = 0;
  std::size_t summary_decodes = 0, summary_evictions = 0, cross_evictions = 0;
  std::size_t sketch_decodes = 0, decoded_prefixes = 0;
  // A pair of 13-depth text sketches can exceed 64K entries. Keep a bounded
  // working set large enough for both Join sides instead of alternately
  // evicting/re-hashing the same immutable fields on every State.
  static constexpr std::size_t capacity = 512, max_entries = 262144;
  static constexpr std::size_t max_bytes = 64 * 1024 * 1024;
  static std::size_t count(const Json &value) {
    std::size_t result = 0;
    for (auto it = value.at("sketches").begin();
         it != value.at("sketches").end(); ++it)
      result += it.value().at("entries").size();
    return result;
  }
  std::size_t owner_size(const std::shared_ptr<const Json> &owner) const {
    const auto found = owners.find(owner.get());
    return found == owners.end() ? json_bytes(*owner) : found->second.bytes;
  }
  std::size_t additional_bytes(const OwnerCosts &costs) const {
    std::size_t bytes = 0;
    for (const auto &[owner, size] : costs)
      if (!owners.count(owner))
        bytes += size;
    return bytes;
  }
  OwnerCosts
  owner_costs(std::initializer_list<std::shared_ptr<const Json>> values) const {
    OwnerCosts result;
    for (const auto &value : values) {
      if (!value ||
          std::any_of(result.begin(), result.end(), [&](const auto &entry) {
            return entry.first == value.get();
          }))
        continue;
      result.emplace_back(value.get(), owner_size(value));
    }
    return result;
  }
  void retain_owner(const std::shared_ptr<const Json> &owner,
                    const OwnerCosts &costs) {
    if (!owner)
      return;
    const auto *key = owner.get();
    auto &entry = owners[key];
    if (entry.references++ == 0) {
      entry.bytes =
          std::find_if(costs.begin(), costs.end(), [&](const auto &cost) {
            return cost.first == key;
          })->second;
      retained_bytes += entry.bytes;
    }
  }
  void release_owner(const std::shared_ptr<const Json> &owner) {
    if (!owner)
      return;
    const auto *key = owner.get();
    auto found = owners.find(key);
    if (found == owners.end())
      return;
    if (--found->second.references != 0)
      return;
    retained_bytes -= found->second.bytes;
    owners.erase(found);
  }
  void evict() {
    auto oldest_summary = std::min_element(
        summaries.begin(), summaries.end(),
        [](auto &a, auto &b) { return a.second.touched < b.second.touched; });
    auto oldest_cross =
        std::min_element(crosses.begin(), crosses.end(), [](auto &a, auto &b) {
          return a.second.touched < b.second.touched;
        });
    if (oldest_summary != summaries.end() &&
        (oldest_cross == crosses.end() ||
         oldest_summary->second.touched < oldest_cross->second.touched)) {
      summary_entries -= oldest_summary->second.entries;
      retained_bytes -= oldest_summary->second.bytes;
      release_owner(oldest_summary->second.owner);
      summaries.erase(oldest_summary);
      ++summary_evictions;
    } else {
      cross_entries -= oldest_cross->second.entries;
      retained_bytes -= oldest_cross->second.bytes;
      release_owner(oldest_cross->second.source);
      release_owner(oldest_cross->second.target);
      release_owner(oldest_cross->second.common);
      crosses.erase(oldest_cross);
      ++cross_evictions;
    }
  }
  Summary *summary(const std::shared_ptr<const Json> &value) {
    auto it = summaries.find(value.get());
    if (it == summaries.end()) {
      validate_summary(*value);
      auto entries = count(*value);
      // Sorted fixed-size hash/count vectors share the existing bounded bank.
      // Keep the conservative allowance for vectors, depth metadata and results.
      const auto costs = owner_costs({value});
      const auto bytes = costs.front().second + entries * 96 + 128 * 1024;
      if (bytes > max_bytes || costs.front().second > max_bytes - bytes)
        return nullptr;
      while (!summaries.empty() && (summaries.size() >= capacity ||
                                    summary_entries + entries > max_entries)) {
        auto oldest = std::min_element(
            summaries.begin(), summaries.end(), [](auto &a, auto &b) {
              return a.second.touched < b.second.touched;
            });
        summary_entries -= oldest->second.entries;
        retained_bytes -= oldest->second.bytes;
        release_owner(oldest->second.owner);
        summaries.erase(oldest);
        ++summary_evictions;
      }
      while (!summaries.empty() || !crosses.empty()) {
        if (retained_bytes + bytes + additional_bytes(costs) <= max_bytes)
          break;
        evict();
      }
      ++summary_decodes;
      it = summaries
               .emplace(value.get(),
                        Summary{value, {}, entries, ++clock, bytes, {}, {}})
               .first;
      retain_owner(value, costs);
      summary_entries += entries;
      retained_bytes += bytes;
      peak_bytes = std::max(peak_bytes, retained_bytes);
    } else
      it->second.touched = ++clock;
    return &it->second;
  }
  std::shared_ptr<const DecodedSketch>
  decoded(const std::shared_ptr<const Json> &owner, const std::string &depth) {
    auto *bank = summary(owner);
    if (bank) {
      const auto found = bank->depths.find(depth);
      if (found != bank->depths.end())
        return found->second;
    }
    auto value = std::make_shared<const DecodedSketch>(
        decode_sketch(owner->at("sketches").at(depth)));
    ++sketch_decodes;
    decoded_prefixes += value->counts.size();
    if (bank)
      bank->depths.emplace(depth, value);
    return value;
  }
};
ReuseCache::ReuseCache() : impl(std::make_unique<Impl>()) {}
ReuseCache::~ReuseCache() = default;
Json ReuseCache::diagnostics() const {
  return {{"hits", impl->result_hits},
          {"misses", impl->result_misses},
          {"summary_decodes", impl->summary_decodes},
          {"sketch_decodes", impl->sketch_decodes},
          {"decoded_prefixes", impl->decoded_prefixes},
          {"summary_evictions", impl->summary_evictions},
          {"cross_evictions", impl->cross_evictions},
          {"summary_entries", impl->summary_entries},
          {"cross_entries", impl->cross_entries},
          {"entry_limit", Impl::max_entries},
          {"retained_bytes_estimate", impl->retained_bytes},
          {"peak_bytes_estimate", impl->peak_bytes},
          {"byte_limit", Impl::max_bytes}};
}
Json ReuseCache::self(const std::shared_ptr<const Json> &s, const Json &alpha,
                      const Json &requests) {
  if (Impl::count(*s) > Impl::max_entries)
    return self_reuse(*s, alpha, requests);
  auto *bank = impl->summary(s);
  if (!bank)
    return self_reuse(*s, alpha, requests);
  return bank->self_results.get(
      Json::array({alpha, requests}),
      [&]() -> Json {
        auto r = survival_rate(alpha);
        double count = s->at("count").get<double>();
        if (r.is_null())
          return count ? Json(nullptr) : Json(0.);
        const double distinct = r.get<double>() * count;
        double denominator =
            requests.is_null() ? distinct : requests.get<double>();
        if (!denominator)
          return 0.;
        if (!distinct)
          return nullptr;
        const double scale = denominator / distinct;
        std::map<std::size_t, double> values;
        for (auto it = s->at("sketches").begin(); it != s->at("sketches").end();
             ++it) {
          auto sampled = impl->decoded(s, it.key());
          auto a = survival_rate(alpha, it.key());
          if (a.is_null() && sampled->sample_size)
            return nullptr;
          double rate = a.is_null() ? 0 : a.get<double>();
          Sum total;
          for (std::size_t i = 0; i < sampled->sample_size; ++i) {
            const auto &entry = sampled->counts[i];
            total.add(std::max(0., semcache::statistics::repeated_prefix_hits(
                                       rate, entry.second) +
                                       (scale - 1.) * rate * entry.second));
          }
          values[std::stoull(it.key())] = total.get() / sampled->tau;
        }
        return integral(values) / denominator;
      },
      impl->result_hits, impl->result_misses);
}
Json ReuseCache::cross(const std::shared_ptr<const Json> &source,
                       const std::shared_ptr<const Json> &target,
                       const Json &as, const Json &at,
                       const CrossDependence &dependence,
                       const Json &requests) {
  validate_summary(*source, false);
  validate_summary(*target);
  if (!compatible(*source, *target))
    return nullptr;
  auto r = survival_rate(at);
  double count = target->at("count").get<double>();
  if (r.is_null())
    return count ? Json(nullptr) : Json(0.);
  double denominator =
      requests.is_null() ? r.get<double>() * count : requests.get<double>();
  if (!denominator)
    return 0.;
  if (dependence.same_field) {
    std::size_t entries;
    try {
      entries = Impl::count(*target);
    } catch (const Json::exception &) {
      // Preserve lazy validation: an earlier unknown depth can return NULL
      // before a later malformed sketch is inspected by the public formula.
      return cross_reuse(*source, *target, as, at, dependence, requests);
    }
    if (entries > Impl::max_entries)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    auto *bank = impl->summary(target);
    if (!bank)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    return bank->same_field_results.get(
        Json::array({at, requests}),
        [&]() -> Json {
          std::map<std::size_t, double> values;
          for (auto it = target->at("sketches").begin();
               it != target->at("sketches").end(); ++it) {
            auto sampled = impl->decoded(target, it.key());
            const auto rate = survival_rate(at, it.key());
            if (rate.is_null() && sampled->sample_size)
              return nullptr;
            Sum total;
            for (std::size_t i = 0; i < sampled->sample_size; ++i) {
              const auto &entry = sampled->counts[i];
              total.add(present(rate.is_null() ? 0. : rate.get<double>(),
                                entry.second));
            }
            values[std::stoull(it.key())] = total.get() / sampled->tau;
          }
          return integral(values) / denominator;
        },
        impl->result_hits, impl->result_misses);
  }
  auto entries = Impl::count(*source) + Impl::count(*target);
  if (entries > Impl::max_entries)
    return cross_reuse(*source, *target, as, at, dependence, requests);
  auto key =
      std::make_tuple(source.get(), target.get(), dependence.common.get());
  auto found = impl->crosses.find(key);
  if (found == impl->crosses.end()) {
    // Only intersections are retained in Cross::depths. Charging both full
    // input sketches to every pair needlessly evicts the same working set.
    entries = 0;
    for (auto depth = source->at("sketches").begin();
         depth != source->at("sketches").end(); ++depth) {
      const auto matched = target->at("sketches").find(depth.key());
      if (matched != target->at("sketches").end())
        entries += std::min(depth->at("entries").size(),
                            matched->at("entries").size());
    }
    const auto costs = impl->owner_costs({source, target, dependence.common});
    const auto bytes = entries * 32 + 64 * 1024;
    std::size_t owner_total = 0;
    for (const auto &[_, size] : costs)
      owner_total += size;
    if (bytes > Impl::max_bytes || owner_total > Impl::max_bytes - bytes)
      return cross_reuse(*source, *target, as, at, dependence, requests);
    while (!impl->crosses.empty() &&
           (impl->crosses.size() >= Impl::capacity ||
            impl->cross_entries + entries > Impl::max_entries)) {
      auto oldest = std::min_element(
          impl->crosses.begin(), impl->crosses.end(),
          [](auto &a, auto &b) { return a.second.touched < b.second.touched; });
      impl->cross_entries -= oldest->second.entries;
      impl->retained_bytes -= oldest->second.bytes;
      impl->release_owner(oldest->second.source);
      impl->release_owner(oldest->second.target);
      impl->release_owner(oldest->second.common);
      impl->crosses.erase(oldest);
      ++impl->cross_evictions;
    }
    while (!impl->summaries.empty() || !impl->crosses.empty()) {
      if (impl->retained_bytes + bytes + impl->additional_bytes(costs) <=
          Impl::max_bytes)
        break;
      impl->evict();
    }
    found = impl->crosses
                .emplace(key, Impl::Cross{source,
                                          target,
                                          dependence.common,
                                          std::make_shared<Impl::CrossValues>(),
                                          entries,
                                          ++impl->clock,
                                          bytes})
                .first;
    impl->retain_owner(source, costs);
    impl->retain_owner(target, costs);
    impl->retain_owner(dependence.common, costs);
    impl->cross_entries += entries;
    impl->retained_bytes += bytes;
    impl->peak_bytes = std::max(impl->peak_bytes, impl->retained_bytes);
  } else
    found->second.touched = ++impl->clock;
  // Decoding a previously unseen operand can evict the pair from the bounded
  // cache. Retain its numeric results until this evaluation has finished.
  auto bank = found->second.values;
  return bank->results.get(
      Json::array({as, at, requests}),
      [&]() -> Json {
        std::map<std::size_t, double> values;
        for (auto it = source->at("sketches").begin();
             it != source->at("sketches").end(); ++it) {
          if (!target->at("sketches").contains(it.key()))
            continue;
          auto term = bank->depths.find(it.key());
          if (term == bank->depths.end()) {
            const auto left = impl->decoded(source, it.key()),
                       right = impl->decoded(target, it.key());
            auto sampled = cross_sample_counts(
                left->counts, right->counts,
                cross_capacity(it.value(), target->at("sketches").at(it.key())));
            std::vector<std::array<double, 3>> counts;
            counts.reserve(sampled.counts.size());
            const auto overlap = dependence.common
                ? semcache::statistics::prefix_common_counts(*dependence.common, it.key())
                : std::map<Hash, double>{};
            for (auto &entry : sampled.counts)
              counts.push_back(
                  {static_cast<double>(entry.second[0]),
                   static_cast<double>(entry.second[1]),
                   overlap.count(entry.first) ? overlap.at(entry.first) : 0.});
            term = bank->depths
                       .emplace(it.key(),
                                std::make_pair(sampled.tau, std::move(counts)))
                       .first;
          }
          auto a = survival_rate(as, it.key()), b = survival_rate(at, it.key());
          if ((a.is_null() || b.is_null()) && !term->second.second.empty())
            return nullptr;
          Sum total;
          if (!a.is_null() && !b.is_null())
            for (auto &counts : term->second.second)
              total.add(dependence.common
                            ? joint_presence(a.get<double>(), b.get<double>(),
                                             counts[0], counts[1], counts[2])
                            : present(a.get<double>(), counts[0]) *
                                  present(b.get<double>(), counts[1]));
          values[std::stoull(it.key())] = total.get() / term->second.first;
        }
        return integral(values) / denominator;
      },
      impl->result_hits, impl->result_misses);
}
Json ReuseCache::join(const std::shared_ptr<const Json> &l,
                      const std::shared_ptr<const Json> &r, const Json &length,
                      const Json &al, const Json &ar, const Json &left_requests,
                      const Json &right_requests) {
  auto a = survival_rate(al), b = survival_rate(ar);
  double lc = l->at("count").get<double>(), rc = r->at("count").get<double>();
  if ((!a.is_null() && a == 0) || (!b.is_null() && b == 0) || !lc || !rc)
    return 0.;
  if (a.is_null() || b.is_null())
    return nullptr;
  double right = right_requests.is_null() ? rc * b.get<double>()
                                          : right_requests.get<double>();
  auto x = self(l, al, left_requests), y = self(r, ar, right_requests);
  if (x.is_null() || y.is_null())
    return nullptr;
  double mean =
      (length.is_null() ? l->at("total_length") : length).get<double>() / lc;
  return x.get<double>() / right + std::max(0., right - 1) / right * mean +
         y.get<double>();
}
std::string prefix_namespace(const Json &op, const Json &action,
                             const std::string &system,
                             const std::string &spec) {
  const auto &layout = action.at("layout");
  Json fields = layout.value("field_order", Json::array());
  if (fields.empty()) {
    fields = layout.value("left_field_order", Json::array());
    for (auto &f : layout.value("right_field_order", Json::array()))
      fields.push_back(f);
  }
  Json inputs = op.value("input_fields", Json::array());
  if (inputs.empty()) {
    inputs = op.value("left_input_fields", Json::array());
    for (auto &f : op.value("right_input_fields", Json::array()))
      inputs.push_back(f);
  }
  std::map<std::size_t, std::size_t> inverse;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    auto pos = std::find(inputs.begin(), inputs.end(), fields[i]);
    if (pos == inputs.end())
      throw std::invalid_argument("Unknown layout field");
    inverse[pos - inputs.begin()] = i;
  }
  std::string task = op.at("task").get<std::string>(), translated;
  static const std::regex pattern("\\{([0-9]+)\\}");
  std::size_t previous = 0;
  for (std::sregex_iterator it(task.begin(), task.end(), pattern), end;
       it != end; ++it) {
    translated += task.substr(previous, it->position() - previous) + "{" +
                  std::to_string(inverse.at(std::stoull((*it)[1]))) + "}";
    previous = it->position() + it->length();
  }
  translated += task.substr(previous);
  std::string parts;
  for (auto &raw : layout.at("block_order")) {
    std::string block = raw;
    if (block == "DATA_BLOCK")
      break;
    if (!parts.empty())
      parts += "\n\n";
    parts += "<" + block + ">\n" +
             (block == "TASK_DESCRIPTION" ? translated : spec) + "\n</" +
             block + ">";
  }
  if (!parts.empty())
    parts += "\n\n";
  return canonical(
      Json::array({"prompt-v4", system, parts + "<DATA_BLOCK>\n{0}: "}));
}
Json prefix_length(const std::string &prefix) {
  const auto parts = Json::parse(prefix, nullptr, false);
  if (!parts.is_array() || parts.size() != 3 || !parts.at(1).is_string() ||
      !parts.at(2).is_string())
    return nullptr;
  const auto &system = parts.at(1).get_ref<const std::string &>();
  const auto &user = parts.at(2).get_ref<const std::string &>();
  return character_length(system) +
         character_length(user.substr(0, user.find("<DATA_BLOCK>")));
}
RequestKeyPlan::RequestKeyPlan(const Json &op, const std::string &model,
                               const Json &layout, bool two_inputs) {
  auto fixed =
      Json::array({op.at("kind"), op.at("task"),
                   op.value("output_schema", Json(nullptr)), model, layout});
  prefix_ = canonical(fixed);
  prefix_.back() = ',';
  Json names = op.value("input_fields", Json::array());
  if (names.empty()) {
    names = op.value("left_input_fields", Json::array());
    for (auto &n : op.value("right_input_fields", Json::array()))
      names.push_back(n);
  }
  std::map<std::string, std::string> descriptors;
  for (auto &d : op.value("input_field_schema", Json::array())) {
    std::string name = d.at("name");
    descriptors[name] = d.value("value_key", name);
  }
  auto right = op.value("right_input_fields", Json::array());
  for (auto &raw : names) {
    std::string name = raw;
    slots_.push_back({descriptors.count(name) ? descriptors.at(name) : name,
                      two_inputs && std::find(right.begin(), right.end(),
                                              raw) != right.end()});
  }
}
std::string RequestKeyPlan::operator()(const Json &left,
                                       const Json *right) const {
  return from_values(
      [&](const std::string &name, bool is_right) -> const Json & {
        if (is_right && !right)
          throw std::invalid_argument("Missing right semantic input");
        return (is_right ? *right : left).at("values").at(name);
      });
}
std::string request_key(const Json &op, const std::vector<Json> &unit,
                        const std::string &model, const Json &layout) {
  return RequestKeyPlan(op, model, layout, unit.size() == 2)(
      unit.at(0), unit.size() == 2 ? &unit[1] : nullptr);
}
Json propagate(const Json &op, const Json &prior,
               const std::vector<Json> &sizes) {
  Json out = prior;
  propagate_inplace(op, out, sizes);
  return out;
}
void propagate_inplace(const Json &op, Json &out,
                       const std::vector<Json> &sizes) {
  double incoming = 1;
  for (auto &n : sizes) {
    if (!number(n)) {
      for (auto &key : {"input_rows", "output_rows", "input_side_rows",
                        "relational_cost", "relational_seconds"})
        out.erase(key);
      return;
    }
    incoming *= n.get<double>();
  }
  if (sizes.empty())
    return;
  const bool group_prior = out.value("group_count_estimated", false);
  out.erase("group_count_estimated");
  Json rate = group_prior ? Json(nullptr)
                          : out.value("selectivity", Json(nullptr)),
       old = out.value("input_rows", Json(nullptr));
  std::string kind = op.value("kind", op.value("type", std::string()));
  std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
  const bool uncertain = out.value("uncertain", false) ||
                         out.value("survival_estimated", false) || group_prior;
  if (rate.is_null() && !uncertain && number(old) && old.get<double>() &&
      number(out.value("output_rows", Json(nullptr))))
    rate = out["output_rows"].get<double>() / old.get<double>();
  if (kind.find("PROJECTION") != std::string::npos || kind == "CROSS_PRODUCT" ||
      kind == "ORDER_BY" || kind == "CTE_SCAN")
    rate = 1.;
  if ((out.contains("selectivity_fusion") ||
       (out.value("survival_estimated", false) &&
        !out.value("survival_estimation_basis", std::string()).empty())) &&
      number(out.value("estimated_selectivity", Json(nullptr))))
    rate = out.at("estimated_selectivity");
  const auto duckdb = out.value("duckdb_selectivity",
                                op.value("duckdb_selectivity", Json(nullptr)));
  if (kind.rfind("SEMANTIC", 0) != 0 && rate.is_null() && number(duckdb) &&
      (kind.find("FILTER") != std::string::npos ||
       kind.find("JOIN") != std::string::npos)) {
    rate = duckdb;
    out["duckdb_selectivity"] = duckdb;
    out["estimated_selectivity"] = rate;
    out["survival_estimated"] = true;
    out["selectivity_fusion"] = {{"weights", {{"duckdb", 1.}}},
                                 {"repair", true}};
    out["survival_estimation_basis"] =
        "weak DuckDB selectivity prior; not observed";
  }
  out["input_rows"] = incoming;
  out["input_side_rows"] = sizes;
  if (old != Json(incoming))
    for (auto &key : {"output_rows", "relational_cost", "relational_seconds"})
      out.erase(key);
  if (op.value("join_type", std::string()) == "MARK" && sizes.size() == 2) {
    out["output_rows"] = sizes[0];
    out["selectivity"] =
        incoming ? Json(sizes[0].get<double>() / incoming) : Json(nullptr);
    out["cardinality_basis"] = "MARK join preserves each left input row";
  } else if (op.contains("sampling_passthrough_input")) {
    const auto side = op.at("sampling_passthrough_input").get<std::size_t>();
    if (side >= sizes.size())
      throw std::invalid_argument("Invalid materialized CTE output boundary");
    out["output_rows"] = sizes[side];
    out["selectivity"] =
        incoming ? Json(sizes[side].get<double>() / incoming) : Json(nullptr);
    out["cardinality_basis"] = "materialized CTE consumer output";
  } else if (op.value("scalar_aggregate", false)) {
    out["output_rows"] = 1.;
    out["selectivity"] = incoming ? Json(1 / incoming) : Json(nullptr);
  } else if (kind.find("LIMIT") != std::string::npos &&
             number(op.value("limit", Json(nullptr)))) {
    double n = std::min(std::max(0., incoming - op.value("offset", 0.)),
                        op["limit"].get<double>());
    out["output_rows"] = n;
    out["selectivity"] = incoming ? Json(n / incoming) : Json(nullptr);
  } else if (kind.find("AGGREGATE") != std::string::npos && rate.is_null() &&
             number(op.value("estimated_rows", Json(nullptr))) &&
             op.at("estimated_rows").get<double>() > 0) {
    // A logical group count is a coarse prior, not a conditional rate. In
    // particular an expected fractional input must not produce many groups.
    const double n = std::min(incoming, op.at("estimated_rows").get<double>());
    out["output_rows"] = n;
    out["selectivity"] = incoming ? Json(n / incoming) : Json(nullptr);
    out["group_count_estimated"] = true;
    out["cardinality_basis"] =
        "current upstream rows capped by DuckDB logical group-count prior";
  } else if (uncertain && rate.is_null()) {
    for (auto &key : {"output_rows", "relational_cost", "relational_seconds"})
      out.erase(key);
    out["selectivity"] = nullptr;
  } else if (!incoming)
    out["output_rows"] = 0.;
  else if (number(rate)) {
    if (!out.contains("selectivity_fusion"))
      out["selectivity"] = rate;
    out["output_rows"] = incoming * rate.get<double>();
  }
  if (kind.find("JOIN") != std::string::npos &&
      op.value("join_type", std::string()) == "LEFT" && sizes.size() == 2) {
    // Null extension preserves the left input even when the right is empty.
    Json output = out.value("output_rows", Json(nullptr));
    if (sizes[0] == 0 || sizes[1] == 0)
      output = sizes[0];
    else if (number(output))
      output = std::max(sizes[0].get<double>(), output.get<double>());
    if (number(output)) {
      out["output_rows"] = output;
      out["selectivity"] =
          incoming ? Json(output.get<double>() / incoming) : Json(nullptr);
    }
  }
  if (!out.contains("cardinality_basis"))
    out["cardinality_basis"] = "upstream estimates × local selectivity";
}
bool RequestLedger::charge(std::size_t requests, bool accuracy) {
  // Accuracy admission is bounded by per-operator sampling batches. Keep its
  // physical HTTP accounting independent from the ordinary selection budget.
  if (!accuracy && requests > remaining())
    return false;
  (accuracy ? accuracy_used_ : selection_used_) += requests;
  return true;
}
void RequestLedger::publish(const std::string &key, bool success,
                            const Json &value) {
  values_.insert_or_assign(key, RequestResult{success, value});
}
const RequestResult *RequestLedger::find(const std::string &key) const {
  auto it = values_.find(key);
  return it == values_.end() ? nullptr : &it->second;
}
}
