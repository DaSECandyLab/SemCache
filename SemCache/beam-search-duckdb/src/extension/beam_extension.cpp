#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "profile_search.hpp"

namespace duckdb {
static void BeamSearchInfo(DataChunk &args, ExpressionState &, Vector &result) {
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(
      result,
      "{\"extension\":\"beam_search\",\"version\":1,\"core\":\"resident\","
      "\"requires_python\":false,\"learned_value_interface\":\"native_"
      "callback\"}");
}
static void BeamSearchPlan(DataChunk &args, ExpressionState &, Vector &result) {
  UnaryExecutor::Execute<string_t, string_t>(
      args.data[0], result, args.size(), [&](string_t input) {
        try {
          return StringVector::AddString(
              result, semcache::search_profile_configuration(
                          semcache::sampling::Json::parse(input.GetString()))
                          .dump());
        } catch (const std::exception &error) {
          throw InvalidInputException("beam_search configuration/search: %s",
                                      error.what());
        }
      });
}
void RegisterBeamSearch(ExtensionLoader &loader) {
  loader.RegisterFunction(ScalarFunction("beam_search_info", {},
                                         LogicalType::VARCHAR, BeamSearchInfo));
  auto plan = ScalarFunction("beam_search_plan", {LogicalType::VARCHAR},
                             LogicalType::VARCHAR, BeamSearchPlan);
  plan.stability = FunctionStability::VOLATILE;
  loader.RegisterFunction(plan);
}
}

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(beam_search, loader) {
  duckdb::RegisterBeamSearch(loader);
}
}
