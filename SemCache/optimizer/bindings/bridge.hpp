#pragma once
#include "sampling/common/exact_integer.hpp"
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>

namespace semcache::sampling {
namespace py = pybind11;
using Json = nlohmann::ordered_json;
// Direct same-process object conversion at import, batched prediction and final
// export boundaries. This is not a serialized message/RPC protocol.
inline Json from_python(py::handle value) {
  if (value.is_none())
    return nullptr;
  if (PyBool_Check(value.ptr()))
    return value.ptr() == Py_True;
  if (PyLong_Check(value.ptr())) {
    int overflow = 0;
    auto signed_value = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (!overflow && !PyErr_Occurred())
      return signed_value;
    PyErr_Clear();
    if (overflow > 0) {
      auto unsigned_value = PyLong_AsUnsignedLongLong(value.ptr());
      if (!PyErr_Occurred())
        return unsigned_value;
      PyErr_Clear();
    }
    return make_integer(py::str(value).cast<std::string>());
  }
  if (PyFloat_Check(value.ptr())) {
    const auto number = py::cast<double>(value);
    if (!std::isfinite(number))
      throw py::value_error("Nonfinite native input");
    return number;
  }
  if (PyUnicode_Check(value.ptr()))
    return py::cast<std::string>(value);
  if (PyDict_Check(value.ptr())) {
    Json result = Json::object();
    for (auto item : py::reinterpret_borrow<py::dict>(value))
      result[py::str(item.first).cast<std::string>()] =
          from_python(item.second);
    return result;
  }
  if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
    Json result = Json::array();
    for (auto item : py::reinterpret_borrow<py::iterable>(value))
      result.push_back(from_python(item));
    return result;
  }
  throw py::type_error(
      "Native input requires typed scalar/dict/list/tuple values");
}
inline py::object to_python(const Json &value) {
  if (value.is_null())
    return py::none();
  if (value.is_boolean())
    return py::bool_(value.get<bool>());
  if (value.is_number_unsigned())
    return py::int_(value.get<std::uint64_t>());
  if (value.is_number_integer())
    return py::int_(value.get<std::int64_t>());
  if (value.is_number_float())
    return py::float_(value.get<double>());
  if (value.is_string())
    return py::str(value.get<std::string>());
  if (exact_integer(value)) {
    auto text = integer_text(value);
    auto number = PyLong_FromString(text.data(), nullptr, 10);
    if (!number)
      throw py::error_already_set();
    return py::reinterpret_steal<py::object>(number);
  }
  if (value.is_object()) {
    py::dict result;
    for (auto it = value.begin(); it != value.end(); ++it)
      result[py::str(it.key())] = to_python(it.value());
    return result;
  }
  py::list result;
  for (const auto &item : value)
    result.append(to_python(item));
  return result;
}
}
