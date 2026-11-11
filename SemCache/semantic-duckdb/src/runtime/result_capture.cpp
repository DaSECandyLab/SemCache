// Stream the actual input of the final aggregate without rerunning semantic work.
#include "runtime/result_capture.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "llm/json_utils.hpp"
#include <fstream>
#include <mutex>
#include <map>

namespace duckdb {
namespace semantic_extension {
namespace {
using Json = semantic::llm_detail::Json;

using NamedBindings = std::map<std::pair<idx_t, idx_t>, string>;
void InputNames(LogicalOperator &op, NamedBindings &names) {
	for (auto &child : op.children)
		InputNames(*child, names);
	if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &projection = op.Cast<LogicalProjection>();
		for (idx_t i = 0; i < projection.expressions.size(); ++i)
			names[{projection.table_index, i}] = projection.expressions[i]->GetName();
	} else if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		for (const auto &binding : get.GetColumnBindings())
			if (binding.table_index == get.table_index && binding.column_index < get.GetColumnIds().size())
				names[{binding.table_index, binding.column_index}] =
				    get.GetColumnName(get.GetColumnIds()[binding.column_index]);
	} else if (op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR && op.GetName() == "SEMANTIC_PROJECTION") {
		auto &projection = static_cast<SemanticLogicalOperator &>(op);
		names[{projection.table_index, 0}] = "__semantic_output_" + std::to_string(projection.OperatorId());
	}
}

class PhysicalResultCapture : public PhysicalOperator {
public:
	PhysicalResultCapture(PhysicalPlan &plan, PhysicalOperator &child, string path, vector<string> names)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, child.GetTypes(), child.estimated_cardinality),
	      file(path, std::ios::trunc) {
		children.push_back(child);
		if (!file)
			throw InvalidInputException("Cannot open aggregate input capture");
		Json columns = Json::array();
		for (idx_t i = 0; i < names.size(); ++i)
			columns.push_back({names[i], types[i].ToString()});
		file << Json({{"columns", columns}}).dump() << '\n';
		file.flush();
	}
	mutable std::ofstream file;
	mutable std::mutex mutex;
	string GetName() const override {
		return "RESULT_INPUT_CAPTURE";
	}
	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		Json rows = Json::array();
		for (idx_t row = 0; row < input.size(); ++row) {
			Json values = Json::array();
			for (idx_t column = 0; column < input.ColumnCount(); ++column) {
				auto value = input.GetValue(column, row);
				values.push_back(value.IsNull() ? Json(nullptr) : Json(value.ToString()));
			}
			rows.push_back(std::move(values));
		}
		{
			std::lock_guard<std::mutex> guard(mutex);
			file << rows.dump() << '\n';
			file.flush();
			if (!file)
				throw InvalidInputException("Cannot write aggregate input capture");
		}
		output.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
};

class LogicalResultCapture : public LogicalExtensionOperator {
public:
	LogicalResultCapture(unique_ptr<LogicalOperator> child, string path, vector<string> names)
	    : path(std::move(path)), names(std::move(names)) {
		children.push_back(std::move(child));
	}
	string path;
	vector<string> names;
	string GetName() const override {
		return "RESULT_INPUT_CAPTURE";
	}
	bool SupportSerialization() const override {
		return false;
	}
	vector<ColumnBinding> GetColumnBindings() override {
		return children[0]->GetColumnBindings();
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		auto &child = planner.CreatePlan(*children[0]);
		return planner.Make<PhysicalResultCapture>(child, path, names);
	}

protected:
	void ResolveTypes() override {
		types = children[0]->types;
	}
};
}

void CaptureAggregateInput(unique_ptr<LogicalOperator> &plan, const string &path) {
	if (path.empty())
		return;
	auto *current = plan.get();
	while (current->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		if (current->children.size() != 1)
			throw InvalidInputException("Aggregate capture requires one final aggregate");
		current = current->children[0].get();
	}
	auto &child = current->children[0];
	// Legal filter/join reordering can place a non-projection directly below the
	// aggregate. Preserve its actual ordered bindings instead of rejecting it or
	// moving capture across a predicate, which would record the wrong rows.
	NamedBindings named;
	InputNames(*child, named);
	vector<string> names;
	for (const auto &binding : child->GetColumnBindings()) {
		auto found = named.find({binding.table_index, binding.column_index});
		if (found == named.end())
			throw InvalidInputException("Aggregate capture cannot resolve input column %s", binding.ToString());
		names.push_back(found->second);
	}
	if (names.size() != child->types.size())
		throw InvalidInputException("Aggregate capture input schema differs");
	child = make_uniq<LogicalResultCapture>(std::move(child), path, std::move(names));
	plan->ResolveOperatorTypes();
}
}
}
