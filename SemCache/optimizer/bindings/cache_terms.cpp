#include "bridge.hpp"
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <utility>
#include <vector>
#include "semcache/prefix_reuse.hpp"
#include "semcache/prefix_cooccurrence.hpp"

namespace py = pybind11;
namespace {
// CPython 3.12's compensated float sum; never compile with fast-math.
struct OrderedSum {
  double hi = 0.0, lo = 0.0;
  void add(double value) {
    const double total = hi + value;
    lo += std::abs(hi) >= std::abs(value) ? (hi - total) + value
                                          : (value - total) + hi;
    hi = total;
  }
  double value() const { return lo != 0.0 && std::isfinite(lo) ? hi + lo : hi; }
};
void valid_rate(double alpha) {
  if (!(alpha >= 0.0 && alpha <= 1.0))
    throw std::invalid_argument("Survival probability must be in [0,1]");
}
void valid_frequency(double count) {
  if (!(count > 0.0) || !std::isfinite(count))
    throw std::invalid_argument("Frequency must be finite and positive");
}
void valid_tau(double tau) {
  if (!(tau > 0.0 && tau <= 1.0))
    throw std::invalid_argument("Sample threshold must be in (0,1]");
}
double presence(double alpha, double count) {
  return alpha == 1.0 ? 1.0 : -std::expm1(count * std::log1p(-alpha));
}
class SelfTerms {
public:
  SelfTerms(std::vector<double> counts, double tau)
      : counts_(std::move(counts)), tau_(tau) {
    valid_tau(tau_);
    for (double count : counts_)
      valid_frequency(count);
  }
  double evaluate(double alpha) const {
    valid_rate(alpha);
    OrderedSum total;
    for (double count : counts_)
      total.add(semcache::statistics::repeated_prefix_hits(alpha, count));
    return total.value() / tau_;
  }

private:
  std::vector<double> counts_;
  double tau_;
};
class CrossTerms {
public:
  CrossTerms(std::vector<std::pair<double, double>> counts, double tau)
      : counts_(std::move(counts)), tau_(tau) {
    valid_tau(tau_);
    for (const auto &counts : counts_) {
      valid_frequency(counts.first);
      valid_frequency(counts.second);
    }
  }
  double evaluate(double source, double target) const {
    valid_rate(source);
    valid_rate(target);
    OrderedSum total;
    for (const auto &counts : counts_) {
      const double left = presence(source, counts.first);
      total.add(left * presence(target, counts.second));
    }
    return total.value() / tau_;
  }

private:
  std::vector<std::pair<double, double>> counts_;
  double tau_;
};
}

void bind_cache_terms(py::module_ &module) {
  module.def("offline_prefix_common", [](py::handle bank, py::handle left,
                                         py::handle right, const std::string &f,
                                         const std::string &g) {
    using namespace semcache::sampling;
    return to_python(semcache::statistics::offline_prefix_common(
        from_python(bank), from_python(left), from_python(right), f, g));
  });
  py::class_<SelfTerms>(module, "SelfTerms")
      .def(py::init<std::vector<double>, double>())
      .def("evaluate", &SelfTerms::evaluate,
           py::call_guard<py::gil_scoped_release>());
  py::class_<CrossTerms>(module, "CrossTerms")
      .def(py::init<std::vector<std::pair<double, double>>, double>())
      .def("evaluate", &CrossTerms::evaluate,
           py::call_guard<py::gil_scoped_release>());
}
