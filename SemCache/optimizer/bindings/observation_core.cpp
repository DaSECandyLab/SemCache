#include "observation_core.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <numeric>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>
namespace py = pybind11;
namespace {
using Tail = std::pair<std::vector<double>, std::vector<double>>;
using Tails = std::vector<Tail>;
using Quotas = std::vector<std::pair<double, double>>;
// Match CPython 3.12 compensated float summation, including iteration order and rounding.
struct PythonSum {
  double hi = 0., lo = 0.;
  void add(double x) {
#if PY_VERSION_HEX >= 0x030C0000
    double t = hi + x;
    lo += std::abs(hi) >= std::abs(x) ? (hi - t) + x : (x - t) + hi;
    hi = t;
#else
    hi += x;
#endif
  }
  double value() const { return lo && std::isfinite(lo) ? hi + lo : hi; }
};

double square(double value) {
  double result = std::pow(value, 2);
  if (std::isfinite(value) && !std::isfinite(result)) {
    PyErr_SetString(PyExc_OverflowError,
                    "(34, 'Numerical result out of range')");
    throw py::error_already_set();
  }
  return result;
}
std::pair<double, double> interval(double passed, double observed,
                                   double population) {
  if (observed == population && observed)
    return {passed / observed, passed / observed};
  if (!observed)
    return {0., 1.};
  const double z2 = std::pow(1.96, 2), rate = passed / observed;
  double center = (rate + z2 / (2 * observed)) / (1 + z2 / observed);
  double radicand =
      z2 * (rate * (1 - rate) / observed + z2 / (4 * square(observed)));
  if (radicand < 0)
    throw py::value_error("math domain error");
  double radius = std::sqrt(radicand) / (1 + z2 / observed);
  return {std::max(0., center - radius), std::min(1., center + radius)};
}
double effective(const std::vector<double> &weights) {
  if (weights.empty())
    return 0.;
  PythonSum total_sum, square_sum;
  for (double w : weights) {
    total_sum.add(w);
    square_sum.add(w * w);
  }
  double total = total_sum.value(), squares = square_sum.value();
  if (squares == 0.) {
    PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
    throw py::error_already_set();
  }
  return square(total) / squares;
}
// The partials algorithm and final half-even correction used by math.fsum.
// Serialized sizes and positive weights are finite before reaching this kernel.
struct AccurateSum {
  std::vector<double> partials;
  void add(double x) {
    size_t i = 0;
    for (double y : partials) {
      if (std::abs(x) < std::abs(y))
        std::swap(x, y);
      double hi = x + y, lo = y - (hi - x);
      if (lo)
        partials[i++] = lo;
      x = hi;
    }
    partials.resize(i);
    if (x)
      partials.push_back(x);
  }
  double value() const {
    double hi = 0., lo = 0.;
    size_t n = partials.size();
    if (n) {
      hi = partials[--n];
      while (n) {
        double x = hi, y = partials[--n];
        hi = x + y;
        lo = y - (hi - x);
        if (lo)
          break;
      }
      if (n && ((lo < 0 && partials[n - 1] < 0) ||
                (lo > 0 && partials[n - 1] > 0))) {
        double y = lo * 2, x = hi + y;
        if (y == x - hi)
          hi = x;
      }
    }
    return hi;
  }
};
struct DepthGroup {
  double observed, passed, weight, population;
  std::pair<double, double> bounds;
};
py::dict field_rates(const Tails &cohorts, const Tails &populations,
                     const Quotas &quotas, const std::vector<long long> &depths,
                     bool unconditional, bool complete, double minimum,
                     double error) {
  if (cohorts.size() != populations.size() || cohorts.size() != quotas.size())
    throw py::value_error("Mismatched field cohorts");
  py::dict result;
  for (auto depth : depths) {
    std::vector<DepthGroup> groups;
    std::vector<double> masses;
    PythonSum population_sum, denominator_sum;
    double observed = 0., passed = 0.;
    bool all_census = true, represented = true;
    for (size_t i = 0; i < cohorts.size(); ++i) {
      const auto &c = cohorts[i];
      const auto &p = populations[i];
      size_t a = std::lower_bound(c.first.begin(), c.first.end(), depth) -
                 c.first.begin();
      size_t b = std::lower_bound(p.first.begin(), p.first.end(), depth) -
                 p.first.begin();
      double pop = p.second.at(b), n = c.first.size() - a, k = c.second.at(a);
      double weight =
          quotas[i].second ? quotas[i].first / quotas[i].second : 0.;
      bool census = quotas[i].second == quotas[i].first;
      groups.push_back({n, k, weight, pop, interval(k, n, census ? n : pop)});
      double mass = unconditional ? pop : n * weight;
      masses.push_back(mass);
      population_sum.add(pop);
      observed += n;
      passed += k;
      denominator_sum.add(mass);
      represented = represented && (!mass || n);
      all_census = all_census && census;
    }
    double population = population_sum.value(),
           denominator = denominator_sum.value();
    bool known = !population || (complete && observed && represented);
    PythonSum numerator_sum, lower_sum, upper_sum, sampled_sum, square_sum;
    py::list strata;
    for (size_t i = 0; i < groups.size(); ++i) {
      auto &g = groups[i];
      double mass = masses[i];
      if (g.observed) {
        numerator_sum.add(mass * g.passed / g.observed);
        sampled_sum.add(mass);
        square_sum.add(mass * mass / g.observed);
      }
      lower_sum.add(mass * g.bounds.first);
      upper_sum.add(mass * g.bounds.second);
      py::dict s;
      s["population"] = g.population;
      s["observed"] = py::int_(static_cast<long long>(g.observed));
      s["passed"] = py::int_(static_cast<long long>(g.passed));
      s["interval"] = py::make_tuple(g.bounds.first, g.bounds.second);
      s["interval"] = py::list(s["interval"]);
      strata.append(s);
    }
    double numerator = numerator_sum.value(), lower = lower_sum.value(),
           upper = upper_sum.value();
    double sampled = sampled_sum.value(), squares = square_sum.value();
    py::object rate =
        known && denominator
            ? py::cast(std::clamp(numerator / denominator, 0., 1.))
        : known && !population ? py::cast(0.)
                               : py::none();
    if (denominator && complete) {
      lower /= denominator;
      upper /= denominator;
    } else {
      lower = 0.;
      upper = 1.;
    }
    if (!population)
      lower = upper = 0.;
    double eff = squares ? square(sampled) / squares : 0.;
    bool census = complete && (observed == population || all_census);
    py::dict d;
    d["rate"] = rate;
    d["observed"] = py::int_(static_cast<long long>(observed));
    d["passed"] = py::int_(static_cast<long long>(passed));
    d["effective_samples"] = eff;
    d["population"] = population;
    d["interval"] = py::list(py::make_tuple(lower, upper));
    d["reliable"] = known && (!population || census ||
                              (eff >= minimum && (upper - lower) / 2 <= error));
    d["strata"] = strata;
    d["interval_basis"] = "stratified Wilson precision diagnostic";
    result[py::str(std::to_string(depth))] = d;
  }
  return result;
}
struct WeightContext {
  py::dict strata, row_strata;
  std::unordered_map<PyObject *, std::pair<py::object, double>> cache;
  std::deque<PyObject *> order;
  WeightContext(py::dict source, py::dict indices)
      : row_strata(std::move(indices)) {
    // Copy only scalar quotas: no row buffers or model execution callbacks.
    for (auto item : source) {
      py::list quotas;
      for (auto raw : py::reinterpret_borrow<py::iterable>(item.second)) {
        py::dict q = py::cast<py::dict>(raw);
        quotas.append(py::make_tuple(q["population"], q["selected"]));
      }
      strata[item.first] = quotas;
    }
  }
  double weight(py::object row) {
    auto found = cache.find(row.ptr());
    if (found != cache.end())
      return found->second.second;
    py::set lineage =
        py::reinterpret_steal<py::set>(PySet_New(row.attr("lineage").ptr()));
    if (!lineage)
      throw py::error_already_set();
    double value = 1.;
    for (auto raw : lineage) {
      py::tuple pair = py::cast<py::tuple>(raw);
      py::object index = row_strata[pair[0]][pair[1]];
      py::tuple quota = py::cast<py::tuple>(strata[pair[0]][index]);
      double selected = py::cast<double>(quota[1]);
      if (selected == 0.) {
        PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
        throw py::error_already_set();
      }
      value *= py::cast<double>(quota[0]) / selected;
    }
    cache.emplace(row.ptr(), std::make_pair(row, value));
    order.push_back(row.ptr());
    if (order.size() > 65536) {
      cache.erase(order.front());
      order.pop_front();
    }
    return value;
  }
  double mass(py::iterable rows) {
    PythonSum total;
    for (auto row : rows)
      total.add(weight(py::reinterpret_borrow<py::object>(row)));
    return total.value();
  }
  double size(py::iterable rows) {
    std::vector<double> weights;
    for (auto row : rows)
      weights.push_back(weight(py::reinterpret_borrow<py::object>(row)));
    return effective(weights);
  }
};
struct Observation {
  py::object token, references, rows, statistics, boundary;
  bool complete;
  Observation(py::object t, py::object refs, py::object r, py::object s, bool c,
              py::object b)
      : token(std::move(t)), references(std::move(refs)), rows(std::move(r)),
        statistics(std::move(s)), boundary(std::move(b)), complete(c) {}
};
// Each cell is (serialized value, Unicode length, UTF-8 bytes, bin, is_null).
struct SummaryWeights {
  std::vector<double> weights;
  double total, normal, eff;
  SummaryWeights(std::vector<double> values, double population)
      : weights(std::move(values)), total(population) {
    for (double w : weights)
      if (!std::isfinite(w) || w <= 0)
        throw py::value_error(
            "Field summaries require finite positive sampling weights");
    const double squared_total = square(total);
    PythonSum normal_sum, square_sum;
    for (double w : weights) {
      normal_sum.add(w);
      square_sum.add(w * w);
    }
    normal = normal_sum.value();
    const double squares = square_sum.value();
    if (squares == 0.) {
      PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
      throw py::error_already_set();
    }
    eff = squared_total / squares;
  }
};
// Serialized equality is shared by every field, including freshly measured
// compounds. Scalar lookup stays type-specific, so bool/int and signed float
// zero never alias before serialization. No value bank survives this call.
struct StringIds {
  py::dict indices;
  size_t size = 0;
  size_t intern(py::handle encoded) {
    PyObject *found = PyDict_GetItemWithError(indices.ptr(), encoded.ptr());
    if (found)
      return PyLong_AsSize_t(found);
    if (PyErr_Occurred())
      throw py::error_already_set();
    const size_t index = size++;
    indices[encoded] = py::int_(index);
    return index;
  }
};
struct CellMetric {
  size_t serialized, bin;
  double length, bytes;
  bool is_null;
};
struct MetricBank {
  StringIds encoded;
  py::dict texts, integers;
  std::unordered_map<uint64_t, size_t> floats;
  std::array<size_t, 3> special{{SIZE_MAX, SIZE_MAX, SIZE_MAX}};
  std::vector<CellMetric> metrics;
  py::function serialize;
  explicit MetricBank(py::function callback) : serialize(std::move(callback)) {}
  const CellMetric &measure(py::handle value) {
    PyObject *v = value.ptr();
    PyObject *cache = PyUnicode_CheckExact(v) ? texts.ptr()
                      : PyLong_CheckExact(v)  ? integers.ptr()
                                              : nullptr;
    size_t index = SIZE_MAX, special_index = SIZE_MAX;
    uint64_t float_bits = 0;
    if (cache) {
      PyObject *found = PyDict_GetItemWithError(cache, v);
      if (found)
        return metrics[PyLong_AsSize_t(found)];
      if (PyErr_Occurred())
        throw py::error_already_set();
    } else if (PyFloat_CheckExact(v)) {
      const double number = PyFloat_AS_DOUBLE(v);
      std::memcpy(&float_bits, &number, sizeof(float_bits));
      auto found = floats.find(float_bits);
      if (found != floats.end())
        return metrics[found->second];
    } else if (v == Py_None || PyBool_Check(v)) {
      special_index = v == Py_None ? 0 : v == Py_False ? 1 : 2;
      if (special[special_index] != SIZE_MAX)
        return metrics[special[special_index]];
    }
    py::object text = py::reinterpret_steal<py::object>(
        PyObject_CallOneArg(serialize.ptr(), v));
    if (!text)
      throw py::error_already_set();
    index = encoded.intern(text);
    if (index == metrics.size()) {
      if (!PyUnicode_Check(text.ptr()))
        throw py::type_error("Serialized values must be strings");
      const Py_ssize_t length = PyUnicode_GetLength(text.ptr());
      Py_ssize_t bytes = 0;
      if (!PyUnicode_AsUTF8AndSize(text.ptr(), &bytes))
        throw py::error_already_set();
      constexpr std::array<Py_ssize_t, 7> bounds{
          {8, 32, 128, 512, 2048, 8192, 32768}};
      const size_t bin =
          std::lower_bound(bounds.begin(), bounds.end(), length) -
          bounds.begin();
      metrics.push_back({index, bin, static_cast<double>(length),
                         static_cast<double>(bytes), v == Py_None});
    }
    if (cache) {
      if (PyDict_SetItem(cache, v, py::int_(index).ptr()) < 0)
        throw py::error_already_set();
    } else if (PyFloat_CheckExact(v))
      floats.emplace(float_bits, index);
    else if (special_index != SIZE_MAX)
      special[special_index] = index;
    return metrics[index];
  }
};
struct SummaryField {
  std::string name;
  bool include_width;
  size_t present = 0, nonnull = 0, top = 0;
  PythonSum lengths, bytes;
  std::array<AccurateSum, 8> bins;
  AccurateSum nulls;
  std::unordered_map<size_t, size_t> counts;
  explicit SummaryField(std::string key)
      : name(std::move(key)),
        include_width(name.compare(0, 17, "__semantic_input:") != 0) {}
  void add(const CellMetric &cell, double weight) {
    ++present;
    lengths.add(cell.length * weight);
    bytes.add(cell.bytes * weight);
    bins.at(cell.bin).add(weight);
    if (cell.is_null)
      nulls.add(weight);
    else {
      ++nonnull;
      top = std::max(top, ++counts[cell.serialized]);
    }
  }
};
struct SummaryAccumulator {
  const SummaryWeights &prepared;
  size_t rows;
  py::dict indices;
  std::vector<SummaryField> fields;
  PythonSum width;
  SummaryAccumulator(const SummaryWeights &weights, size_t count)
      : prepared(weights), rows(count) {
    if (rows != weights.weights.size())
      throw py::value_error("Mismatched sample weights");
  }
  bool add(py::handle name, const CellMetric &cell, double weight) {
    PyObject *found = PyDict_GetItemWithError(indices.ptr(), name.ptr());
    size_t index;
    if (found)
      index = PyLong_AsSize_t(found);
    else {
      if (PyErr_Occurred())
        throw py::error_already_set();
      index = fields.size();
      fields.emplace_back(py::cast<std::string>(name));
      indices[name] = py::int_(index);
    }
    auto &field = fields[index];
    field.add(cell, weight);
    return field.include_width;
  }
  py::tuple finish() {
    if (!rows)
      return py::make_tuple(py::none(), py::dict());
    std::sort(fields.begin(), fields.end(),
              [](const auto &a, const auto &b) { return a.name < b.name; });
    py::dict result;
    for (auto &field : fields) {
      if (field.present != rows)
        continue;
      py::list histogram;
      for (const auto &bin : field.bins)
        histogram.append(bin.value() / prepared.total);
      py::dict d;
      d["null_fraction"] = field.nulls.value() / prepared.total;
      d["average_length"] = field.lengths.value() / prepared.total;
      d["average_bytes"] = field.bytes.value() / prepared.total;
      d["sample_distinct_fraction"] =
          field.nonnull ? py::cast(static_cast<double>(field.counts.size()) /
                                   field.nonnull)
                        : py::none();
      d["sample_top_frequency"] =
          field.nonnull
              ? py::cast(static_cast<double>(field.top) / field.nonnull)
              : py::none();
      d["sample_effective_samples"] = prepared.eff;
      d["length_histogram"] = histogram;
      d["basis"] = "weighted admitted sample; serialized characters";
      result[py::str(field.name)] = d;
    }
    return py::make_tuple(width.value() / prepared.normal, result);
  }
};
// Measured-cell and raw-row inputs share reduction and publication rules.
py::tuple summarize(py::list rows, const SummaryWeights &prepared) {
  if (rows.empty())
    return py::make_tuple(py::none(), py::dict());
  SummaryAccumulator summary(prepared, rows.size());
  StringIds encoded;
  for (size_t i = 0; i < rows.size(); ++i) {
    py::dict row = py::cast<py::dict>(rows[i]);
    double bytes = 0., w = prepared.weights[i];
    for (auto raw : row) {
      py::tuple cell = py::cast<py::tuple>(raw.second);
      const bool is_null = py::cast<bool>(cell[4]);
      const size_t serialized =
          is_null ? 0 : encoded.intern(py::str(py::cast<std::string>(cell[0])));
      CellMetric metric{serialized, py::cast<size_t>(cell[3]),
                        py::cast<double>(cell[1]), py::cast<double>(cell[2]),
                        is_null};
      if (summary.add(raw.first, metric, w))
        bytes += metric.bytes;
    }
    summary.width.add(w * bytes);
  }
  return summary.finish();
}
py::tuple summarize_values(py::sequence rows, const SummaryWeights &prepared,
                           py::function serialize,
                           py::object published_dictionary) {
  if (rows.empty())
    return py::make_tuple(py::none(), py::dict());
  SummaryAccumulator summary(prepared, rows.size());
  MetricBank bank(std::move(serialize));
  size_t i = 0;
  for (auto raw : rows) {
    py::object values = py::reinterpret_borrow<py::object>(raw).attr("values");
    double bytes = 0., w = prepared.weights[i++];
    auto add = [&](py::handle key, const CellMetric &metric) {
      if (summary.add(py::handle(key), metric, w))
        bytes += metric.bytes;
    };
    if (PyDict_CheckExact(values.ptr()) ||
        reinterpret_cast<PyObject *>(Py_TYPE(values.ptr())) ==
            published_dictionary.ptr()) {
      Py_ssize_t position = 0;
      PyObject *key, *value;
      while (PyDict_Next(values.ptr(), &position, &key, &value)) {
        auto metric = bank.measure(py::handle(value));
        // Null membership belongs to the cell, not its serialized ID.
        metric.is_null = value == Py_None;
        add(py::handle(key), metric);
      }
    } else {
      // Custom mappings retain items() order; duplicate keys keep the last measured metric.
      py::dict indices;
      std::vector<CellMetric> cells;
      for (auto raw_cell :
           py::reinterpret_borrow<py::iterable>(values.attr("items")())) {
        py::tuple pair =
            py::tuple(py::reinterpret_borrow<py::object>(raw_cell));
        if (pair.size() != 2)
          throw py::value_error("Mapping items must contain two values");
        auto metric = bank.measure(pair[1]);
        metric.is_null = pair[1].is_none();
        indices[pair[0]] = py::int_(cells.size());
        cells.push_back(metric);
      }
      for (auto cell : indices)
        add(cell.first, cells[py::cast<size_t>(cell.second)]);
    }
    summary.width.add(w * bytes);
  }
  return summary.finish();
}

}
void bind_observations(py::module_ &m) {
  m.def("observation_effective", &effective);
  m.def("observation_interval", &interval);
  m.def("observation_field_rates", &field_rates);
  py::class_<SummaryWeights>(m, "ObservationSummaryWeights")
      .def(py::init<std::vector<double>, double>());
  m.def("observation_summarize", &summarize);
  m.def("observation_summarize_values", &summarize_values, py::arg("rows"),
        py::arg("weights"), py::arg("serialize"),
        py::arg("published_dictionary") = py::none());
  py::class_<WeightContext, std::shared_ptr<WeightContext>>(
      m, "ObservationWeights")
      .def(py::init<py::dict, py::dict>())
      .def("weight", &WeightContext::weight)
      .def("mass", &WeightContext::mass)
      .def("effective", &WeightContext::size);
  py::class_<Observation, std::shared_ptr<Observation>>(
      m, "NativeObservation", py::custom_type_setup([](PyHeapTypeObject *heap) {
        auto *type = &heap->ht_type;
        type->tp_flags |= Py_TPFLAGS_HAVE_GC;
        type->tp_traverse = [](PyObject *self, visitproc visit, void *arg) {
          Py_VISIT(Py_TYPE(self));
          if (py::detail::is_holder_constructed(self)) {
            auto &value = py::cast<Observation &>(py::handle(self));
            Py_VISIT(value.token.ptr());
            Py_VISIT(value.references.ptr());
            Py_VISIT(value.rows.ptr());
            Py_VISIT(value.statistics.ptr());
            Py_VISIT(value.boundary.ptr());
          }
          return 0;
        };
        type->tp_clear = [](PyObject *self) {
          if (py::detail::is_holder_constructed(self)) {
            auto &value = py::cast<Observation &>(py::handle(self));
            value.token = py::object();
            value.references = py::object();
            value.rows = py::object();
            value.statistics = py::object();
            value.boundary = py::object();
          }
          return 0;
        };
      }))
      .def(py::init<py::object, py::object, py::object, py::object, bool,
                    py::object>())
      .def_readonly("token", &Observation::token)
      .def_readonly("references", &Observation::references)
      .def_readonly("rows", &Observation::rows)
      .def_readonly("statistics", &Observation::statistics)
      .def_readonly("complete", &Observation::complete)
      .def_readonly("boundary", &Observation::boundary);
}
