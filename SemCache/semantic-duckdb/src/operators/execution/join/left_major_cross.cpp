#include "operators/execution/join/join_internal.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {
namespace semantic_extension {
namespace join_detail {

namespace {

class LeftMajorCrossGlobalState : public GlobalSinkState {
public:
	LeftMajorCrossGlobalState(ClientContext &context, const PhysicalOperator &op)
	    : rhs_materialized(context, op.children[1].get().GetTypes()) {
		rhs_materialized.InitializeAppend(append_state);
	}
	ColumnDataCollection rhs_materialized;
	ColumnDataAppendState append_state;
	mutex rhs_lock;
};

class LeftMajorCrossOperatorState : public CachingOperatorState {
public:
	explicit LeftMajorCrossOperatorState(ColumnDataCollection &rhs)
	    : rhs(rhs), left_pos(0), scanning_row(false) {
		rhs.InitializeScanChunk(scan_chunk);
	}
	ColumnDataCollection &rhs;
	ColumnDataScanState scan_state;
	DataChunk scan_chunk;
	idx_t left_pos;
	bool scanning_row;
};

class PhysicalLeftMajorCrossProduct : public CachingPhysicalOperator {
public:
	PhysicalLeftMajorCrossProduct(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperator &left,
	                              PhysicalOperator &right, idx_t estimated_cardinality)
	    : CachingPhysicalOperator(physical_plan, PhysicalOperatorType::CROSS_PRODUCT, std::move(types),
	                              estimated_cardinality) {
		children.push_back(left);
		children.push_back(right);
	}
	string GetName() const override {
		return "LEFT_MAJOR_CROSS_PRODUCT";
	}
	OrderPreservationType OperatorOrder() const override {
		return OrderPreservationType::INSERTION_ORDER;
	}
	bool ParallelOperator() const override {
		return false;
	}
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
	bool SinkOrderDependent() const override {
		return true;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<LeftMajorCrossGlobalState>(context, *this);
	}
	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &sink = input.global_state.Cast<LeftMajorCrossGlobalState>();
		lock_guard<mutex> guard(sink.rhs_lock);
		sink.rhs_materialized.Append(sink.append_state, chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &) const override {
		auto &sink = sink_state->Cast<LeftMajorCrossGlobalState>();
		return make_uniq<LeftMajorCrossOperatorState>(sink.rhs_materialized);
	}
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override {
		PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
	}
	vector<const_reference<PhysicalOperator>> GetSources() const override {
		return children[0].get().GetSources();
	}

protected:
	OperatorResultType ExecuteInternal(ExecutionContext &, DataChunk &input, DataChunk &chunk, GlobalOperatorState &,
	                                   OperatorState &state_p) const override {
		auto &state = state_p.Cast<LeftMajorCrossOperatorState>();
		if (state.rhs.Count() == 0)
			return OperatorResultType::FINISHED;
		if (input.size() == 0)
			return OperatorResultType::NEED_MORE_INPUT;
		for (;;) {
			if (!state.scanning_row) {
				state.rhs.InitializeScan(state.scan_state);
				state.scanning_row = true;
			}
			state.scan_chunk.Reset();
			state.rhs.Scan(state.scan_state, state.scan_chunk);
			if (state.scan_chunk.size() == 0) {
				state.scanning_row = false;
				++state.left_pos;
				if (state.left_pos >= input.size()) {
					state.left_pos = 0;
					return OperatorResultType::NEED_MORE_INPUT;
				}
				continue;
			}
			chunk.SetCardinality(state.scan_chunk.size());
			for (idx_t c = 0; c < input.ColumnCount(); ++c)
				ConstantVector::Reference(chunk.data[c], input.data[c], state.left_pos, state.scan_chunk.size());
			for (idx_t c = 0; c < state.scan_chunk.ColumnCount(); ++c)
				chunk.data[input.ColumnCount() + c].Reference(state.scan_chunk.data[c]);
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
	}
};

class LogicalLeftMajorCrossProduct : public LogicalExtensionOperator {
public:
	LogicalLeftMajorCrossProduct(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right,
	                             idx_t cardinality) {
		children.push_back(std::move(left));
		children.push_back(std::move(right));
		estimated_cardinality = cardinality;
	}
	string GetName() const override {
		return "LEFT_MAJOR_CROSS_PRODUCT";
	}
	vector<ColumnBinding> GetColumnBindings() override {
		auto result = children[0]->GetColumnBindings();
		auto right = children[1]->GetColumnBindings();
		result.insert(result.end(), right.begin(), right.end());
		return result;
	}
	bool SupportSerialization() const override {
		return false;
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		auto &left = planner.CreatePlan(*children[0]);
		auto &right = planner.CreatePlan(*children[1]);
		return planner.Make<PhysicalLeftMajorCrossProduct>(types, left, right, estimated_cardinality);
	}

protected:
	void ResolveTypes() override {
		types = children[0]->types;
		types.insert(types.end(), children[1]->types.begin(), children[1]->types.end());
	}
};

} // namespace

unique_ptr<LogicalOperator> CreateLogicalLeftMajorCrossProduct(unique_ptr<LogicalOperator> left,
                                                                 unique_ptr<LogicalOperator> right,
                                                                 idx_t estimated_cardinality) {
	return make_uniq<LogicalLeftMajorCrossProduct>(std::move(left), std::move(right), estimated_cardinality);
}

} // namespace join_detail
} // namespace semantic_extension
} // namespace duckdb
