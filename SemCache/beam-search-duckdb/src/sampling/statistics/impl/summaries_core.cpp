#include "impl_internal.hpp"
#include "../joint_presence.hpp"
#include "semcache/prefix_reuse.hpp"
#include "semcache/prefix_sketch.h"
#include <charconv>
#include <cmath>
#include <mbedtls/sha256.h>

namespace semcache::sampling {
namespace stats_detail {

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
std::string encode(const Json &value, bool sorted) {
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
double present(double a, double count) {
  return a == 1 ? 1 : -std::expm1(count * std::log1p(-a));
}
double integral(std::map<std::size_t, double> values,
                std::optional<std::size_t> max_observed) {
  if (max_observed) {
    if (*max_observed == 0)
      return 0.;
    const auto limit = *max_observed + 1;
    if (!values.empty() && values.rbegin()->first > limit) {
      values.erase(values.upper_bound(limit), values.end());
      values[limit] = 0.;
    }
  }
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
std::optional<std::size_t> summary_observed_max(const Json &s) {
  if (s.contains("observed_max_length") &&
      s.at("observed_max_length").is_number_integer()) {
    const auto v = s.at("observed_max_length").get<std::int64_t>();
    return v <= 0 ? std::size_t(0) : static_cast<std::size_t>(v);
  }
  return std::nullopt;
}
std::optional<std::size_t> cross_observed_max(const Json &a, const Json &b) {
  const auto la = summary_observed_max(a), lb = summary_observed_max(b);
  if (la && lb)
    return std::min(*la, *lb);
  return la ? la : lb;
}
bool compatible(const Json &a, const Json &b) {
  if (a.at("metadata").at("serializer") != b.at("metadata").at("serializer"))
    return false;
  for (auto it = a.at("sketches").begin(); it != a.at("sketches").end(); ++it)
    if (b.at("sketches").contains(it.key()))
      return true;
  return false;
}
void validate_summary(const Json &s, bool field_serializer) {
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
CrossSample cross_sample(const Json &a, const Json &b) {
  const auto left = semcache::statistics::prefix_counts(a),
             right = semcache::statistics::prefix_counts(b);
  return cross_sample_counts(left, right, cross_capacity(a, b));
}
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
bool number(const Json &v) {
  return v.is_number() && std::isfinite(v.get<double>()) &&
         v.get<double>() >= 0;
}

} // namespace stats_detail

using namespace stats_detail;
namespace {
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

} // namespace

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
          0., scale < 1.
                  ? semcache::statistics::repeated_prefix_hits(scale * a,
                                                               entry.second)
                  : semcache::statistics::repeated_prefix_hits(a, entry.second) +
                        (scale - 1.) * a * entry.second));
    values[std::stoull(it.key())] = sum.get() / p.tau;
  }
  return integral(std::move(values), summary_observed_max(s)) / den;
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
  const double distinct = r.get<double>() * count;
  double den =
      requests.is_null() ? distinct : requests.get<double>();
  if (den <= 0.)
    return 0.;
  if (!requests.is_null() && distinct > 0. && den < distinct)
    den = distinct;
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
      values[std::stoull(it.key())] = std::min(den, total.get() / sampled.tau);
    }
    return integral(std::move(values), summary_observed_max(t)) / den;
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
    values[std::stoull(it.key())] = std::min(den, sum.get() / p.tau);
  }
  return integral(std::move(values), cross_observed_max(s, t)) / den;
}

} // namespace semcache::sampling
