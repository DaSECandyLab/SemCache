#include "state_bindings.hpp"
#include "bridge.hpp"
#include <cmath>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace semcache::python_adapter {
using Json = nlohmann::json;

Json import_resources(const py::dict &models, const py::dict &constraints) {
  py::dict encoded, limits;
  py::list order;
  Json hashes = Json::object();
  auto number = [](py::handle value) -> py::object {
    if (PyLong_CheckExact(value.ptr()))
      return py::reinterpret_borrow<py::object>(value);
    if (PyFloat_CheckExact(value.ptr()) &&
        std::isfinite(py::cast<double>(value)))
      return py::reinterpret_borrow<py::object>(value);
    return py::none();
  };
  for (auto entry : models) {
    const auto name = py::cast<std::string>(entry.first);
    const auto row = py::cast<py::dict>(entry.second);
    py::dict metadata;
    metadata["loaded"] =
        row.contains("loaded") && PyObject_IsTrue(row["loaded"].ptr());
    for (const auto key : {"gpu_weight_bytes", "gpu_kv_reserve_bytes",
                           "cpu_weight_bytes", "cpu_kv_reserve_bytes"})
      metadata[key] = row.contains(key) ? number(row[key]) : py::none();
    encoded[entry.first] = metadata;
    order.append(entry.first);
    hashes[name] = static_cast<std::uint64_t>(py::hash(entry.first));
  }
  for (const auto key : {"gpu_memory_bytes", "cpu_memory_bytes"})
    if (constraints.contains(key))
      limits[key] = number(constraints[key]);
  return Json{{"models", sampling::from_python(encoded)},
              {"constraints", sampling::from_python(limits)},
              {"model_order", sampling::from_python(order)},
              {"model_hashes", hashes}};
}
Json import_problem(const py::dict &config) {
  py::dict metadata = config.attr("copy")().cast<py::dict>();
  metadata["models"] = py::dict();
  metadata["constraints"] = py::dict();
  auto result = Json(sampling::from_python(metadata));
  result.update(import_resources(config["models"].cast<py::dict>(),
                                 config["constraints"].cast<py::dict>()));
  return result;
}
py::dict action_dict(const StateProblem &p, const Action &action) {
  return py::module_::import("json")
      .attr("loads")(p.action_json(action))
      .cast<py::dict>();
}
Action parse_action(const StateProblem &problem, const StatePtr &state,
                    const py::dict &value) {
  problem.complete(state);
  if (!value.contains("operator_id") ||
      !PyLong_CheckExact(value["operator_id"].ptr()))
    throw std::invalid_argument("Illegal state transition");
  Id oid = py::cast<Id>(value["operator_id"]);
  py::dict base;
  for (auto entry : value) {
    auto key = py::cast<std::string>(entry.first);
    if (key != "inputs" && key != "pair_for")
      base[entry.first] = entry.second;
  }
  auto text =
      py::module_::import("json")
          .attr("dumps")(base, py::arg("sort_keys") = true,
                         py::arg("ensure_ascii") = false,
                         py::arg("allow_nan") = false,
                         py::arg("separators") = py::make_tuple(",", ":"))
          .cast<std::string>();
  Action action = problem.implementation(oid, text);
  const auto choices = problem.input_choices(state, oid);
  if (!value.contains("inputs") || value["inputs"].is_none()) {
    if (choices.size() != 1)
      throw std::invalid_argument(
          "Illegal state transition: ambiguous or unavailable inputs");
    action.inputs = choices.front();
  } else {
    // The Python contract compares tuple(inputs) with integer choices,
    // including equality-compatible bool/float IDs at this public boundary.
    py::tuple supplied_inputs;
    try {
      supplied_inputs = py::tuple(value["inputs"]);
    } catch (const py::error_already_set &) {
      throw std::invalid_argument("Illegal state transition");
    }
    bool matched = false;
    for (const auto &choice : choices) {
      if (supplied_inputs.equal(py::tuple(py::cast(choice)))) {
        action.inputs = choice;
        matched = true;
        break;
      }
    }
    if (!matched)
      throw std::invalid_argument("Illegal state transition");
  }
  if (std::find(choices.begin(), choices.end(), action.inputs) ==
          choices.end() ||
      !problem.feasible(state, action))
    throw std::invalid_argument("Illegal state transition");
  action.pair_for = problem.pair_target(state, oid, action.inputs);
  py::object expected =
      action.pair_for ? py::cast(*action.pair_for) : py::none();
  py::object supplied =
      value.contains("pair_for")
          ? py::reinterpret_borrow<py::object>(value["pair_for"])
          : py::none();
  if (!supplied.equal(expected))
    throw std::invalid_argument("Illegal candidate-pair target");
  py::dict normalized(base);
  normalized["inputs"] =
      value.contains("inputs") && !value["inputs"].is_none()
          ? py::reinterpret_borrow<py::object>(value["inputs"])
          : py::cast(action.inputs);
  if (action.pair_for)
    normalized["pair_for"] = py::cast(*action.pair_for);
  action.ingress_json =
      py::module_::import("json")
          .attr("dumps")(normalized, py::arg("sort_keys") = true,
                         py::arg("ensure_ascii") = false,
                         py::arg("allow_nan") = false,
                         py::arg("separators") = py::make_tuple(",", ":"))
          .cast<std::string>();
  return action;
}

}

using semcache::python_adapter::action_dict;
using semcache::python_adapter::import_problem;
using semcache::python_adapter::import_resources;
using semcache::python_adapter::parse_action;

void bind_state(py::module_ &m) {
  using namespace semcache;
  py::register_exception<ActionBudgetExceeded>(m, "ActionBudgetExceeded",
                                               PyExc_ValueError);
  py::class_<PlanNode, NodePtr>(m, "PlanNodeHandle")
      .def_readonly("handle_id", &PlanNode::handle_id)
      .def_readonly("node_id", &PlanNode::node_id)
      .def_readonly("children", &PlanNode::children)
      .def_readonly("height", &PlanNode::height)
      .def_readonly("operator_id", &PlanNode::operator_id)
      .def_readonly("field_ids", &PlanNode::fields)
      .def_readonly("relation_keys", &PlanNode::relations)
      .def_readonly("pending_join", &PlanNode::pending_join)
      .def_readonly("pair_field_ids", &PlanNode::pair_columns)
      .def("__deepcopy__", [](const NodePtr &n, py::dict) { return n; });
  py::class_<State, StatePtr>(m, "StateHandle")
      .def_readonly("handle_id", &State::handle_id)
      .def_readonly("done", &State::done)
      .def_readonly("depth", &State::depth)
      .def_readonly("action_json", &State::action_json)
      .def_property_readonly("roots", &State::root_ids)
      .def_property_readonly("remaining", &State::remaining)
      .def_property_readonly("history", &State::history)
      .def_property_readonly("node_count", &State::node_count)
      .def("node_at", &State::node_at)
      .def("nodes", &State::nodes)
      .def("structure_key", &State::structure_key)
      .def("__deepcopy__", [](const StatePtr &s, py::dict) { return s; });
  py::class_<StateProblem, std::shared_ptr<StateProblem>>(m, "StateProblem")
      .def(py::init([](const py::dict &config) {
        return std::make_shared<StateProblem>(import_problem(config));
      }))
      .def("initial", &StateProblem::initial)
      .def("update_resources",
           [](StateProblem &p, const py::dict &models,
              const py::dict &constraints) {
             auto resources = import_resources(models, constraints);
             p.update_resources(
                 resources.at("models"), resources.at("constraints"),
                 resources.at("model_order"), resources.at("model_hashes"));
           })
      .def("complete", &StateProblem::complete)
      .def("input_choices", &StateProblem::input_choices)
      .def("state_key", &StateProblem::state_key)
      .def("logical_structure", [](const StateProblem &p, const StatePtr &state) {
        const auto shape = p.logical_structure(state);
        py::dict result;
        result["key"] = shape.key;
        result["features"] = shape.features;
        return result;
      })
      .def(
          "actions",
          [](const StateProblem &p, const StatePtr &s,
             bool prioritize_relational, bool fixed_layout,
             const std::optional<ActionPriorities> &highest_priorities,
             const std::optional<std::string> &fixed_model) {
            auto actions =
                p.actions(s, prioritize_relational, fixed_layout,
                          highest_priorities ? &*highest_priorities : nullptr,
                          fixed_model);
            py::list result;
            for (const auto &a : actions)
              result.append(action_dict(p, a));
            return result;
          },
          py::arg("state"), py::arg("prioritize_relational") = false,
          py::arg("fixed_layout") = false,
          py::arg("highest_priorities") = py::none(),
          py::arg("fixed_model_id") = py::none())
      .def("transition",
           [](const StateProblem &p, const StatePtr &s, const py::dict &a) {
             return p.transition(s, parse_action(p, s, a));
           })
      .def("feasible_action",
           [](const StateProblem &p, const StatePtr &s, const py::dict &a) {
             auto base = sampling::from_python(a);
             base.erase("inputs");
             base.erase("pair_for");
             const auto action = p.implementation(
                 base.at("operator_id").get<Id>(), nlohmann::json(base).dump());
             return p.feasible(s, action);
           });
}
