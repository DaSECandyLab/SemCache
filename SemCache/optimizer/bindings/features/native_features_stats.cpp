#include "native_features_internal.hpp"

namespace semcache::python_features_detail {
const Json &get(const Json &object, const char *key) {
  static const Json missing = nullptr;
  const auto found = object.is_object() ? object.find(key) : object.end();
  return found == object.end() ? missing : *found;
}
const Json &get(const StatOverlay &stat, const char *key) {
  return stat.read(key);
}
bool known(const Json &x) {
  return x.is_number() && std::isfinite(x.get<double>());
}
void add_input_time(StatOverlay &stat, const Json &profile,
                    const Json &descriptor, const Json *variant) {
  const bool semantic = descriptor.at("semantic").get<bool>();
  const Json *bank = nullptr;
  const Json *value = nullptr;
  if (semantic) {
    if (variant) {
      bank = &get(get(get(profile, "request_profiles"),
                      variant->at("model").get<std::string>().c_str()),
                  variant->at("kind").get<std::string>().c_str());
      value = &get(*bank, "seconds_per_request");
    }
  } else {
    const auto &label =
        descriptor.at("timing_kind").get_ref<const std::string &>();
    bank = &get(get(get(profile, "operator_time_calibration"), "relational"),
                label.c_str());
    value = &get(*bank, "seconds_per_input");
    if (!known(*value) || value->get<double>() < 0.) {
      bank = &get(get(profile, "relational_profiles"), label.c_str());
      value = &get(*bank, "seconds_per_input");
    }
  }
  stat["seconds_per_input"] = value ? *value : Json();
  double confidence = 0.;
  if (bank) {
    const auto &count = bank->contains("samples") ? get(*bank, "samples")
                                                  : get(*bank, "repeats");
    if (known(count) && count.get<double>() > 0.)
      confidence = count.get<double>() / (count.get<double>() + 1.);
    const auto &cv = get(*bank, "latency_cv");
    if (known(cv) && cv.get<double>() > 0.)
      confidence /= 1. + cv.get<double>();
  }
  if (!get(stat, "reliability").is_object())
    stat["reliability"] = Json::object();
  stat["reliability"]["seconds_per_input"] = confidence;
}
bool truth(const Json &x) {
  if (x.is_null())
    return false;
  if (x.is_boolean())
    return x.get<bool>();
  if (x.is_number())
    return x.get<double>() != 0;
  return !x.empty();
}
double log_value(const Json &x) {
  return known(x) ? std::log1p(std::max(0., x.get<double>())) : 0.;
}
bool valid(const StatOverlay &stat, const char *key) {
  const auto &values = get(stat, "valid");
  return !values.is_object() || !values.contains(key) || truth(values.at(key));
}
double confidence(const StatOverlay &stat, const char *key, double value) {
  const auto &explicit_value = get(get(stat, "reliability"), key);
  double result = 0.;
  if (known(explicit_value))
    result = explicit_value.get<double>();
  else {
    const auto &samples = get(stat, "effective_samples");
    if (known(samples) && samples.get<double>() > 0)
      result = samples.get<double>() / (samples.get<double>() + 1.);
    const auto &bounds = get(get(stat, "intervals"), key);
    if (bounds.is_array() && !bounds.empty() &&
        std::all_of(bounds.begin(), bounds.end(), known)) {
      if (bounds.size() < 2)
        throw std::invalid_argument("Feature interval needs two bounds");
      result /=
          1. + std::max(0., bounds[1].get<double>() - bounds[0].get<double>()) /
                   std::max(1., std::abs(value));
    }
  }
  // Explicit per-metric evidence is independent of generic cardinality flags.
  if (!known(explicit_value) &&
      (truth(get(stat, "uncertain")) ||
       truth(get(stat, "low_survival_uncertain"))))
    result = 0.;
  return std::min(1., std::max(0., result));
}
std::array<double, 3> estimated(const StatOverlay &stat, const char *key,
                                bool ratio, bool probability, bool applicable) {
  const Json *value = &get(stat, key);
  if (std::strcmp(key, "selectivity") == 0 &&
      truth(get(stat, "shared_selectivity")) &&
      stat.contains("estimated_selectivity"))
    value = &stat.at("estimated_selectivity");
  if (!applicable || !known(*value) || value->get<double>() < 0 ||
      !valid(stat, key))
    return {0., 0., 0.};
  double number = value->get<double>();
  if (ratio && probability && number > 1.) {
    if (std::abs(number - 1.) > 1e-12 * std::max(number, 1.))
      throw std::invalid_argument(
          std::string("Probability feature outside [0,1]: ") + key);
    number = 1.;
  }
  return {ratio ? number : std::log1p(std::max(0., number)),
          confidence(stat, key, number), 1.};
}
std::size_t characters(const std::string &s) {
  return std::count_if(s.begin(), s.end(),
                       [](unsigned char c) { return (c & 0xc0) != 0x80; });
}
Json prefix_length(const Json &pre) {
  if (!pre.is_string())
    return nullptr;
  try {
    const auto parts = Json::parse(pre.get<std::string>());
    if (!parts.is_array() || parts.size() != 3 ||
        !std::all_of(parts.begin(), parts.end(),
                     [](const Json &p) { return p.is_string(); }))
      return nullptr;
    auto tail = parts[2].get<std::string>();
    const auto marker = tail.find("<DATA_BLOCK>");
    if (marker != std::string::npos)
      tail.resize(marker);
    return characters(parts[1].get<std::string>()) + characters(tail);
  } catch (const Json::exception &) {
    return nullptr;
  }
}
Json quality(const Json &profile, const Json &variant) {
  auto value = get(get(get(profile, "accuracy"),
                       variant.at("model").get<std::string>().c_str()),
                   variant.at("kind").get<std::string>().c_str());
  if (!value.is_object())
    return value;
  const auto &layouts = get(value, "layouts");
  if (layouts.is_object())
    return get(layouts, variant.at("layout_key").get<std::string>().c_str());
  return value.contains("value") ? value : Json(nullptr);
}
// Expansion summation and the final half-even adjustment match math.fsum.
double precise_sum(const std::vector<double> &values) {
  std::vector<double> partials;
  for (double x : values) {
    std::size_t count = 0;
    for (double y : partials) {
      if (std::abs(x) < std::abs(y))
        std::swap(x, y);
      const double hi = x + y, lo = y - (hi - x);
      if (!std::isfinite(hi))
        throw std::overflow_error("intermediate overflow in fsum");
      if (lo != 0.)
        partials[count++] = lo;
      x = hi;
    }
    partials.resize(count);
    if (x != 0.)
      partials.push_back(x);
  }
  double hi = 0., lo = 0.;
  std::size_t n = partials.size();
  if (n) {
    hi = partials[--n];
    while (n) {
      const double x = hi, y = partials[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo != 0.)
        break;
    }
    if (n && ((lo < 0. && partials[n - 1] < 0.) ||
              (lo > 0. && partials[n - 1] > 0.))) {
      const double y = lo * 2., x = hi + y;
      if (y == x - hi)
        hi = x;
    }
  }
  return hi;
}
HistoryIndex::HistoryIndex(const Json &history) {
  const char *distances[] = {"age_seconds", "requests_since",
                             "characters_since"};
  for (const auto &entry : history) {
    auto &group = models[entry.at("model_id").get<std::string>()];
    const auto &pre = get(entry, "pre");
    if (pre.is_null())
      group.unknown = true;
    else
      group
          .by_prefix[pre.is_string() ? pre.get_ref<const std::string &>()
                                     : pre.dump()]
          .push_back(&entry);
    for (std::size_t i = 0; i < 3; ++i) {
      const auto &v = get(entry, distances[i]);
      if (known(v) && v.get<double>() >= 0.)
        group.scales[i] = std::max(group.scales[i], v.get<double>());
    }
  }
}
std::array<double, history_width> history_features(const HistoryIndex &history,
                                                   const Json &variant,
                                                   const StatOverlay &stat) {
  std::array<double, history_width> out{};
  const auto &packed = get(stat, "l_existing_features");
  if (packed.is_array() && packed.size() == 12 &&
      std::all_of(packed.begin(), packed.end(), known) && packed[8] == 1.) {
    for (std::size_t i = 0; i < 6; ++i) {
      out[i] = packed[i].get<double>();
      out[12 + i] = packed[6 + i].get<double>();
    }
  } else {
    const auto &pre = get(stat, "pre");
    if (pre.is_null())
      return out;
    const auto found =
        history.models.find(variant.at("model").get<std::string>());
    const HistoryGroup *group =
        found == history.models.end() ? nullptr : &found->second;
    if (group && group->unknown)
      return out;
    static const std::vector<const Json *> empty;
    const std::vector<const Json *> *entries = &empty;
    if (group) {
      const auto key =
          pre.is_string() ? pre.get_ref<const std::string &>() : pre.dump();
      const auto compatible = group->by_prefix.find(key);
      if (compatible != group->by_prefix.end())
        entries = &compatible->second;
    }
    const auto &matches = get(stat, "l_existing");
    out[2] = std::log1p(static_cast<double>(entries->size()));
    out[14] = 1.;
    std::vector<double> values;
    const Json *strongest = nullptr;
    double maximum = 0.;
    bool complete = true;
    for (const auto *entry : *entries) {
      const auto &v = get(matches, entry->at("id").get<std::string>().c_str());
      if (!known(v) || v.get<double>() < 0.) {
        complete = false;
        break;
      }
      const double number = v.get<double>();
      values.push_back(number);
      if (!strongest || number > maximum ||
          (number == maximum &&
           entry->at("id").get_ref<const std::string &>() <
               strongest->at("id").get_ref<const std::string &>())) {
        strongest = entry;
        maximum = number;
      }
    }
    out[12] = out[13] = complete ? 1. : 0.;
    if (complete && !entries->empty()) {
      out[0] =
          std::log1p(precise_sum(values) / static_cast<double>(values.size()));
      out[1] = std::log1p(maximum);
      const char *distances[] = {"age_seconds", "requests_since",
                                 "characters_since"};
      for (std::size_t i = 0; i < 3; ++i) {
        const auto &v = get(*strongest, distances[i]);
        if (known(v) && v.get<double>() >= 0.) {
          out[3 + i] =
              group->scales[i] ? v.get<double>() / group->scales[i] : 0.;
          out[15 + i] = 1.;
        }
      }
    }
  }
  if (!valid(stat, "l_existing"))
    for (std::size_t i : {0, 1, 3, 4, 5})
      out[i] = out[12 + i] = 0.;
  const double c =
      out[14] && out[2] == 0. ? 1. : confidence(stat, "l_existing", out[0]);
  out[6] = c * out[12];
  out[7] = c * out[13];
  for (std::size_t i = 2; i < 6; ++i)
    out[6 + i] = out[12 + i];
  return out;
}
} // namespace semcache::python_features_detail
