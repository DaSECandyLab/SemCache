// Central registration point for every SQL-visible function and both optimizer
// phases. Marker scalar functions must be lowered and must never execute directly.
#include "operators/semantic_operators.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "operators/binding/semantic_bind.hpp"
#include "operators/plan/semantic_rewriter.hpp"
#include "optimizer/semantic_optimizer.hpp"
#include "runtime/query_runtime.hpp"

namespace duckdb {

static void SampleJSON(DataChunk &args, ExpressionState &, Vector &result) {
	for (idx_t i = 0; i < args.size(); ++i)
		FlatVector::GetData<string_t>(result)[i] =
		    StringVector::AddString(result, semantic_extension::SampleRowJSON(args.GetValue(0, i)));
}

void RegisterSemanticOperators(ExtensionLoader &loader) {
	ScalarFunction sample_json("semantic_sample_json", {LogicalType::ANY}, LogicalType::VARCHAR, SampleJSON);
	sample_json.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(sample_json);
	// Settings are connection/query inputs; none are hidden process globals.
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption(
	    "semantic_parallelism",
	    "Execution resource hint (1..1024); normal HTTP execution uses bounded concurrency per physical model",
	    LogicalType::BIGINT, Value::BIGINT(1));
	config.AddExtensionOption("semantic_unrestricted",
	                          "Allow independent bounded Beam sampling requests to overlap on one physical model",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));
	config.AddExtensionOption("semantic_max_concurrent_requests_per_model",
	                          "Maximum in-flight normal execution requests for one physical model", LogicalType::BIGINT,
	                          Value::BIGINT(1));
	config.AddExtensionOption("semantic_enforce_model_operator_order",
	                          "Prioritize earlier plan operators among queued normal requests for one physical model",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));
	config.AddExtensionOption("semantic_models", "Model registry JSON", LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("semantic_request_reuse", "Reuse identical zero-temperature requests within one query",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption("semantic_response_cache",
	                          "Bounded exact LLM response cache JSON (empty inherits process configuration)",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("semantic_runtime", "Optional batch admission and runtime reconfiguration JSON",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("semantic_telemetry", "Optional append-only JSONL telemetry file", LogicalType::VARCHAR,
	                          Value(""));
	config.AddExtensionOption("semantic_capture_aggregate_input", "Optional same-execution aggregate input JSONL file",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("semantic_optimizer_socket", "Local Python optimizer Unix socket", LogicalType::VARCHAR,
	                          Value(""));
	config.AddExtensionOption("semantic_sampling_action", "Explicit model/layout for one bounded native sampling stage",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("semantic_optimizer_timeout_ms", "Total optimizer RPC deadline in milliseconds",
	                          LogicalType::BIGINT, Value::BIGINT(10000));
	for (auto &name : {"semantic_filter", "semantic_project", "semantic_join"}) {
		ScalarFunction sql_function(name, {LogicalType::VARCHAR}, LogicalType::BOOLEAN,
		                            semantic_extension::ExecuteUnloweredMarker, semantic_extension::BindSemanticSQL);
		sql_function.varargs = LogicalType::ANY;
		sql_function.stability = FunctionStability::VOLATILE;
		sql_function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		loader.RegisterFunction(sql_function);
	}

	// pre_optimize extracts stable semantic boundaries; optimize receives the
	// external decision after DuckDB has optimized relational children.
	OptimizerExtension extension;
	extension.pre_optimize_function = semantic_extension::ExtractSemanticQuery;
	extension.optimize_function = semantic_extension::ApplySemanticDecision;
	OptimizerExtension::Register(config, std::move(extension));
}

}
