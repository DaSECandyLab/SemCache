#include <algorithm>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
struct FeatureBuffer {
  std::vector<double> values;
};
struct FeatureBuffers {
  std::shared_ptr<FeatureBuffer> numeric = std::make_shared<FeatureBuffer>();
  std::shared_ptr<FeatureBuffer> history = std::make_shared<FeatureBuffer>();
  std::vector<std::size_t> offsets{0};

  static void append_sequence(py::handle value, std::size_t expected,
                              std::vector<double> &destination,
                              const char *label) {
    auto sequence = py::reinterpret_borrow<py::sequence>(value);
    const auto size = py::len(sequence);
    if (static_cast<std::size_t>(size) != expected)
      throw std::invalid_argument(std::string(label) +
                                  " feature width mismatch");
    const auto required = destination.size() + expected;
    if (destination.capacity() < required)
      destination.reserve(std::max(required, destination.capacity() * 2));
    for (auto item : sequence)
      destination.push_back(py::cast<double>(item));
  }

  FeatureBuffers(py::iterable graphs, std::size_t width,
                 std::size_t history_width) {
    if (!width || !history_width)
      throw std::invalid_argument("Feature widths must be positive");
    for (auto graph : graphs) {
      auto nodes = py::cast<py::dict>(py::cast<py::dict>(graph)["nodes"]);
      for (auto entry : nodes) {
        auto row = py::cast<py::dict>(entry.second);
        append_sequence(row["numeric"], width, numeric->values, "Numeric");
        if (row["l_existing"].is_none()) {
          history->values.insert(history->values.end(), history_width, 0.0);
        } else {
          append_sequence(row["l_existing"], history_width, history->values,
                          "History");
        }
      }
      offsets.push_back(offsets.back() + nodes.size());
    }
  }
};
}
void bind_feature_buffers(py::module_ &module) {
  py::class_<FeatureBuffer, std::shared_ptr<FeatureBuffer>>(
      module, "FeatureBuffer", py::buffer_protocol())
      .def_buffer([](FeatureBuffer &buffer) {
        return py::buffer_info(buffer.values.data(), sizeof(double),
                               py::format_descriptor<double>::format(), 1,
                               {buffer.values.size()}, {sizeof(double)});
      });
  py::class_<FeatureBuffers>(module, "FeatureBuffers")
      .def(py::init<py::iterable, std::size_t, std::size_t>())
      .def_property_readonly(
          "numeric",
          [](const FeatureBuffers &buffers) { return buffers.numeric; })
      .def_property_readonly(
          "history",
          [](const FeatureBuffers &buffers) { return buffers.history; })
      .def_readonly("offsets", &FeatureBuffers::offsets);
}
