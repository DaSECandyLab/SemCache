#pragma once

// Two-phase DuckDB optimizer extension state: QueryIR is captured before native
// optimization and the external PlanSpec is safely applied afterwards.

#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "llm/json_utils.hpp"

namespace duckdb {
namespace semantic_extension {

struct SemanticPlanningState : ClientContextState {
	semantic::llm_detail::Json query;
	unique_ptr<LogicalOperator> model_plan;
	void QueryBegin(ClientContext &) override {
		query = nullptr;
		model_plan.reset();
	}
};

void ExtractSemanticQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
void ApplySemanticDecision(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);

}
}
