#pragma once

#include "operators/execution/semantic_operator_factory.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "operators/execution/semantic_execution.hpp"
#include "operators/execution/semantic_input.hpp"
#include "operators/execution/semantic_source.hpp"
#include "operators/execution/semantic_window.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "operators/plan/semantic_plan_tree.hpp"
#include <deque>


namespace duckdb {
namespace semantic_extension {
namespace join_detail {

struct JoinInputs {
	explicit JoinInputs(bool prefetch) : left(prefetch), right(prefetch) {
	}
	SemanticInput left, right;
	void Stop() {
		left.Stop();
		right.Stop();
	}
};
class PhysicalSemanticJoinInput : public PhysicalOperator {
public:
	PhysicalSemanticJoinInput(PhysicalPlan &plan, PhysicalOperator &child, shared_ptr<JoinInputs> inputs, bool left)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, child.GetTypes(), child.estimated_cardinality),
	      inputs(std::move(inputs)), left(left) {
		children.push_back(child);
	}
	shared_ptr<JoinInputs> inputs;
	bool left;
	string GetName() const override {
		return left ? "SEMANTIC_JOIN_LEFT_INPUT" : "SEMANTIC_JOIN_RIGHT_INPUT";
	}
	bool IsSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return true;
	}
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		return (left ? inputs->left : inputs->right).Publish(context.client, chunk, input.interrupt_state);
	}
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &, OperatorSinkFinalizeInput &) const override {
		(left ? inputs->left : inputs->right).Complete();
		return SinkFinalizeType::READY;
	}
};
struct JoinExecutionRequest {
	PendingSemanticRequest pending;
	Value value;
	vector<Value> output_values;
	idx_t input_index;
	bool ready = false;
	size_t retained_bytes = 0;
};
struct StagedJoinPair {
	vector<Value> output_values;
	semantic::Row left_values;
	semantic::Row right_values;
	idx_t pair = 0;
	idx_t left = 0;
	idx_t right = 0;
};
struct JoinState : SemanticSourceState {
	static constexpr idx_t SEMANTIC_JOIN_CHUNK_SIZE = 64;
	PendingSemanticRequest pending;
	std::deque<JoinExecutionRequest> execution_pending;
	std::deque<StagedJoinPair> staged_chunk;
	std::vector<semantic::Row> right_rows;
	SemanticPendingWindow execution_window;
	bool execution_input_complete = false;
	ColumnDataCollection right;
	ColumnDataAppendState append;
	ColumnDataScanState scan;
	ExpressionExecutor left_executor, right_executor, candidate_executor;
	DataChunk left_chunk, right_chunk, left_arguments, right_arguments, candidates, candidate_results;
	semantic::Row left_values;
	idx_t left_row = 0, right_row = 0, left_offset = 0, right_offset = 0;
	bool admitted = false, scanning = false, right_complete = false, budget_stopped = false;
	bool concurrent_execution = false, chunk_execution = false, chunk_active = false;
	JoinState(ClientContext &context, const vector<unique_ptr<Expression>> &expressions,
	          const vector<unique_ptr<Expression>> &predicates, const vector<LogicalType> &left_types,
	          const vector<LogicalType> &right_types, const vector<LogicalType> &types);
};

class PhysicalSemanticNestedLoopJoin : public PhysicalOperator {
public:
	PhysicalSemanticNestedLoopJoin(PhysicalPlan &plan, vector<LogicalType> types, idx_t cardinality,
	                               unique_ptr<FunctionData> binding, vector<unique_ptr<Expression>> expressions,
	                               vector<unique_ptr<Expression>> predicates, shared_ptr<JoinInputs> inputs);
	unique_ptr<FunctionData> binding;
	vector<unique_ptr<Expression>> expressions, predicates;
	shared_ptr<JoinInputs> inputs;
	shared_ptr<Pipeline> registration_anchor;

	string GetName() const override {
		return "SEMANTIC_JOIN";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override;
	bool IsSource() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override;
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType PrepareExecutionCandidate(JoinState &local, SemanticInput *&blocked_input) const;
	bool AwaitExecution(JoinState &local, SemanticInput *upstream, const InterruptState &interrupt,
	                    SourceResultType &result) const;
	SourceResultType GetExecutionData(ExecutionContext &context, DataChunk &output, OperatorSourceInput &input,
	                                  JoinState &local) const;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &output,
	                                 OperatorSourceInput &input) const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;
};

unique_ptr<LogicalOperator> CreateLogicalLeftMajorCrossProduct(unique_ptr<LogicalOperator> left,
                                                                 unique_ptr<LogicalOperator> right,
                                                                 idx_t estimated_cardinality);

} // namespace join_detail
} // namespace semantic_extension
} // namespace duckdb
