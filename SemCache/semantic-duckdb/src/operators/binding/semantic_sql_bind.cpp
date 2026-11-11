#include "operators/binding/semantic_bind.hpp"
#include "runtime/query_runtime.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "prompt/prompt_renderer.hpp"

namespace duckdb {
namespace semantic_extension {
namespace {
semantic::Field FieldOf(const string &name, const LogicalType &type, size_t depth = 0) {
	if (depth > 32)
		throw BinderException("Semantic schema nesting exceeds 32");
	if (type.id() == LogicalTypeId::LIST)
		return {name, semantic::FieldType::LIST, {FieldOf("item", ListType::GetChildType(type), depth + 1)}};
	if (type.id() == LogicalTypeId::STRUCT) {
		std::vector<semantic::Field> fields;
		for (const auto &child : StructType::GetChildTypes(type))
			fields.push_back(FieldOf(child.first, child.second, depth + 1));
		return {name, semantic::FieldType::STRUCT, std::move(fields)};
	}
	for (auto candidate : {semantic::FieldType::BOOLEAN, semantic::FieldType::INTEGER, semantic::FieldType::BIGINT,
	                       semantic::FieldType::DOUBLE, semantic::FieldType::VARCHAR}) {
		if (type == ToLogicalType(candidate))
			return {name, candidate};
	}
	throw BinderException("Unsupported semantic SQL input/output type %s; use an explicit cast", type.ToString());
}

unique_ptr<Expression> Pack(ClientContext &context, vector<unique_ptr<Expression>> arguments) {
	// Package multiple bound expressions into one named STRUCT so the physical
	// operator receives a single value while preserving field identity and type.
	FunctionBinder binder(context);
	ErrorData error;
	auto result = binder.BindScalarFunction(DEFAULT_SCHEMA, "struct_pack", std::move(arguments), error);
	if (error.HasError())
		throw BinderException(error.Message());
	return result;
}
}

unique_ptr<FunctionData> BindSemanticSQL(ClientContext &context, ScalarFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	ValidateExecutionSettings(context);
	if (!ClientConfig::GetConfig(context).enable_optimizer ||
	    DBConfig::GetConfig(context).options.disabled_optimizers.count(OptimizerType::EXTENSION))
		throw BinderException("Semantic SQL requires DuckDB optimizer and extension hooks enabled");
	if (GetSetting(context, "semantic_optimizer_socket").empty() &&
	    GetSetting(context, "semantic_sampling_action").empty())
		throw BinderException("Set semantic_optimizer_socket or semantic_sampling_action before binding semantic SQL");
	if (arguments.size() < 2 || !arguments[0]->IsFoldable())
		throw BinderException("Semantic SQL requires a constant prompt template and input expressions");
	auto task = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	if (task.IsNull())
		throw BinderException("Semantic prompt template cannot be NULL");
	auto result = make_uniq<SemanticBindData>();
	const auto sampling = GetSetting(context, "semantic_sampling_action");
	if (!sampling.empty()) {
		auto setting = semantic::llm_detail::ParseJSON(sampling);
		result->sampling = true;
		result->result_cache_probe = setting.value("result_cache_probe", false);
		if (setting.contains("llm_request_limit")) {
			const auto &limit = setting.at("llm_request_limit");
			if (!limit.is_number_integer() || (!limit.is_number_unsigned() && limit.get<int64_t>() < 0))
				throw BinderException("Invalid native sampling llm_request_limit");
			result->llm_request_limit = limit.get<uint64_t>();
		}
		for (const auto *name : {"selectivity_batch_size", "selectivity_batch_max_bytes", "accuracy_batch_size",
		                         "accuracy_batch_max_bytes"})
			if (setting.contains(name))
				throw BinderException("Sampling batch options are no longer supported; remove %s. "
				                      "Sampling uses one record/pair per prompt",
				                      name);
	}
	result->models = GetSetting(context, "semantic_models");
	auto registry = semantic::ModelRegistry::Parse(result->models);
	result->execution_context.global_system_prefix = semantic::GlobalSystemPrefix();
	auto &node = result->node;
	node.kind = function.name == "semantic_filter"    ? semantic::NodeKind::SEMANTIC_FILTER
	            : function.name == "semantic_project" ? semantic::NodeKind::SEMANTIC_PROJECTION
	                                                  : semantic::NodeKind::SEMANTIC_JOIN;
	auto &config = node.semantic;
	config.operator_id = 0; // Stable IDs are assigned to canonical logical nodes, not bind attempts.
	config.task_description = task.GetValue<string>();
	// Temporary values are only used for validation. Post-optimizer PlanSpec is mandatory.
	config.model_id = *registry.KnownModelIds().begin();
	config.layout.block_order = {semantic::PromptBlock::OPERATOR_SPECIFICATION, semantic::PromptBlock::TASK_DESCRIPTION,
	                             semantic::PromptBlock::DATA_BLOCK};
	if (node.kind == semantic::NodeKind::SEMANTIC_PROJECTION) {
		if (arguments.size() < 3 || arguments.back()->return_type.id() != LogicalTypeId::STRUCT ||
		    !arguments.back()->IsFoldable() || !ExpressionExecutor::EvaluateScalar(context, *arguments.back()).IsNull())
			throw BinderException(
			    "semantic_project requires a final NULL::STRUCT(name TYPE) single-field output contract");
		function.return_type = arguments.back()->return_type;
		config.output_schema.kind = semantic::OutputSchema::Kind::OBJECT;
		auto &fields = StructType::GetChildTypes(function.return_type);
		if (fields.size() != 1)
			throw BinderException("semantic_project requires exactly one output field");
		config.output_schema.fields.push_back(FieldOf(fields[0].first, fields[0].second));
	}
	const auto input_end = arguments.size() - (node.kind == semantic::NodeKind::SEMANTIC_PROJECTION ? 1 : 0);
	for (idx_t i = 1; i < input_end; ++i) {
		if (arguments[i]->IsVolatile() || arguments[i]->HasSubquery())
			throw BinderException("Semantic input expressions must be nonvolatile and contain no subqueries");
		FieldOf("input", arguments[i]->return_type);
		config.input_fields.push_back("arg" + std::to_string(i - 1));
	}
	config.layout.field_order = config.input_fields;
	// Join side ownership can only be checked when the two bound relational children are known.
	semantic::ValidateSemanticConfig(
	    node.kind == semantic::NodeKind::SEMANTIC_JOIN ? semantic::NodeKind::SEMANTIC_FILTER : node.kind, config);
	function.arguments.clear();
	for (auto &argument : arguments)
		function.arguments.push_back(argument->return_type);
	function.varargs = LogicalType::INVALID;
	context.registered_state->GetOrCreate<QueryState>("semantic.runtime");
	return std::move(result);
}

void PrepareSemanticSQL(ClientContext &context, LogicalOperator &child, BoundFunctionExpression &marker) {
	// Resolve column bindings only after the marker's relational child is known;
	// this prevents a field from silently binding to the wrong join side.
	auto &binding = marker.bind_info->Cast<SemanticBindData>();
	if (binding.node.kind == semantic::NodeKind::SEMANTIC_PROJECTION)
		marker.children.pop_back(); // Retain contract through binder serialization, remove only at lowering.
	auto &config = binding.node.semantic;
	vector<unique_ptr<Expression>> left, right;
	const bool join = binding.node.kind == semantic::NodeKind::SEMANTIC_JOIN;
	vector<ColumnBinding> left_bindings, right_bindings;
	if (join) {
		if ((child.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT &&
		     child.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
		     child.type != LogicalOperatorType::LOGICAL_ANY_JOIN) || child.children.size() != 2)
			throw BinderException("Semantic SQL join requires a direct inner binary join boundary");
		left_bindings = child.children[0]->GetColumnBindings();
		right_bindings = child.children[1]->GetColumnBindings();
	}
	for (idx_t i = 1; i < marker.children.size(); ++i) {
		auto &expression = marker.children[i];
		expression->alias = config.input_fields[i - 1];
		bool uses_left = false, uses_right = false;
		if (join) {
			ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
			    *expression, [&](const BoundColumnRefExpression &ref) {
				    if (ref.depth)
					    throw BinderException("Correlated semantic joins are not supported");
				    if (std::find(left_bindings.begin(), left_bindings.end(), ref.binding) != left_bindings.end())
					    uses_left = true;
				    else if (std::find(right_bindings.begin(), right_bindings.end(), ref.binding) !=
				             right_bindings.end())
					    uses_right = true;
				    else
					    throw BinderException("Semantic join input column is unavailable");
			    });
			if (uses_left == uses_right || (uses_left && !right.empty()))
				throw BinderException(
				    "Semantic join inputs must reference exactly one side each, left inputs before right inputs");
		}
		(uses_right ? right : left).push_back(std::move(expression));
	}
	marker.children.resize(1);
	if (join) {
		if (left.empty() || right.empty())
			throw BinderException("Semantic join must reference both left and right attributes");
		config.left_input_fields.assign(config.input_fields.begin(), config.input_fields.begin() + left.size());
		config.right_input_fields.assign(config.input_fields.begin() + left.size(), config.input_fields.end());
		config.input_fields.clear();
		config.layout.field_order.clear();
		config.layout.left_field_order = config.left_input_fields;
		config.layout.right_field_order = config.right_input_fields;
	}
	marker.children.push_back(Pack(context, std::move(left)));
	if (join)
		marker.children.push_back(Pack(context, std::move(right)));
	semantic::ValidateSemanticConfig(binding.node.kind, config);
}
}
}
