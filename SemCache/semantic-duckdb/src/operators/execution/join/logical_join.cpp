#include "operators/execution/join/join_internal.hpp"

namespace duckdb {
namespace semantic_extension {
namespace join_detail {

SourceResultType PhysicalSemanticNestedLoopJoin::GetDataInternal(ExecutionContext &context, DataChunk &output,
                                                                 OperatorSourceInput &input) const {
	auto &local = input.global_state.Cast<JoinState>();
	if (local.concurrent_execution) {
		try {
			return GetExecutionData(context, output, input, local);
		} catch (...) {
			inputs->Stop();
			throw;
		}
	}
	if (local.budget_stopped) {
		output.SetCardinality(0);
		return SourceResultType::FINISHED;
	}
	auto &config = binding->Cast<SemanticBindData>();
	// Candidate scanning and expression evaluation also belong to this join.
	// Fetch/Await return BLOCKED rather than waiting inside this invocation.
	// Ensure retains lazy operator_start semantics for an empty candidate set.
	auto &timing_runtime = context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")
	                           ->Ensure(context.client, config);
	SemanticOperatorActiveScope active(timing_runtime, config.node.semantic.operator_id);
	if (!local.admitted && !local.right_complete)
		inputs->right.Request();
	optional_ptr<QueryRuntime> runtime;
	idx_t emitted = 0, attempted = 0;
	for (;;) {
		if (!local.admitted) {
			if (emitted)
				break;
			local.left_chunk.Reset();
			auto status = inputs->left.Fetch(local.left_chunk, input.interrupt_state);
			if (status != SourceResultType::HAVE_MORE_OUTPUT) {
				if (status == SourceResultType::FINISHED) {
					inputs->Stop();
					context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")
					    ->Finish(context.client, config);
				}
				return status;
			}
			local.left_arguments.Reset();
			local.left_executor.Execute(local.left_chunk, local.left_arguments);
			local.admitted = true;
		}
		if (!local.scanning) {
			local.left_values = ReadRow(local.left_arguments.GetValue(0, local.left_row));
			local.right_chunk.Reset();
			local.right_row = local.right_offset = 0;
			if (local.right_complete)
				local.right.InitializeScan(local.scan);
			local.scanning = true;
		}
		if (local.right_row == local.right_chunk.size()) {
			if (emitted)
				break; // Publish output before requesting more upstream work.
			local.right_offset += local.right_chunk.size();
			local.right_chunk.Reset();
			local.right_row = 0;
			if (!local.right_complete) {
				auto status = inputs->right.Fetch(local.right_chunk, input.interrupt_state);
				if (status == SourceResultType::BLOCKED)
					return status;
				if (status == SourceResultType::FINISHED) {
					local.right_complete = true;
					if (!local.right.Count()) {
						inputs->Stop();
						context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")
						    ->Finish(context.client, config);
						return SourceResultType::FINISHED;
					}
				} else {
					local.right.Append(local.append, local.right_chunk);
				}
			} else {
				local.right.Scan(local.scan, local.right_chunk);
			}
			if (!local.right_chunk.size()) {
				++local.left_row;
				local.scanning = false;
				if (local.left_row == local.left_chunk.size()) {
					local.left_offset += local.left_chunk.size();
					local.left_row = 0;
					local.admitted = false;
				}
				continue;
			}
			local.right_arguments.Reset();
			local.right_executor.Execute(local.right_chunk, local.right_arguments);
			local.right_rows.clear();
			local.right_rows.reserve(local.right_chunk.size());
			for (idx_t r = 0; r < local.right_chunk.size(); ++r)
				local.right_rows.push_back(ReadRow(local.right_arguments.GetValue(0, r)));
			if (!predicates.empty()) {
				local.candidates.Reset();
				local.candidate_results.Reset();
				for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
					local.candidates.data[col].Reference(local.left_chunk.GetValue(col, local.left_row));
				for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
					local.candidates.data[local.left_chunk.ColumnCount() + col].Reference(
					    local.right_chunk.data[col]);
				local.candidates.SetCardinality(local.right_chunk.size());
				local.candidate_executor.Execute(local.candidates, local.candidate_results);
			}
		}

		bool eligible = true;
		for (idx_t col = 0; col < local.candidate_results.ColumnCount(); ++col) {
			auto value = local.candidate_results.GetValue(col, local.right_row);
			eligible &= !value.IsNull() && value.GetValue<bool>();
		}
		++attempted;
		if (eligible) {
			if (emitted && !local.pending.complete)
				break;
			if (!runtime)
				runtime = &context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")
				               ->Get(context.client, config);
			std::lock_guard<std::recursive_mutex> guard(runtime->mutex);
			auto &counts = runtime->counts[config.node.semantic.operator_id];
			auto left = local.left_offset + local.left_row;
			auto right = local.right_offset + local.right_row;
			// Only left row zero can run before the right cardinality is known.
			// Its flattened pair ID equals the right index independently of N_R.
			auto pair = left ? left * local.right.Count() + right : right;
			Value value;
			try {
				value = InferSemanticValue(context, config, *runtime, local.left_values,
				                           &local.right_rows[local.right_row], counts.first,
				                           local.right_complete ? local.right.Count() : 0, pair, left, right,
				                           nullptr, &local.pending, local.CompletionCallback());
			} catch (const SemanticRequestBlocked &) {
				output.SetCardinality(emitted);
				if (emitted)
					return SourceResultType::HAVE_MORE_OUTPUT;
				SourceResultType result;
				if (local.Await(local.pending.response, input.interrupt_state, result))
					return result;
				continue;
			} catch (const SamplingBudgetExhausted &) {
				local.budget_stopped = true;
				inputs->Stop();
				if (!emitted)
					context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")
					    ->Finish(context.client, config);
				output.SetCardinality(emitted);
				return emitted ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
			}
			++counts.first;
			if (value.GetValue<bool>()) {
				for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
					output.SetValue(col, emitted, local.left_chunk.GetValue(col, local.left_row));
				for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
					output.SetValue(local.left_chunk.ColumnCount() + col, emitted,
					                local.right_chunk.GetValue(col, local.right_row));
				++emitted;
				++counts.second;
			}
		}
		++local.right_row;
		if (attempted >= STANDARD_VECTOR_SIZE) {
			if (emitted)
				break;
			// An empty quantum is CPU work, not an asynchronous wait. Calling
			// InterruptState::Callback here would reschedule our own task before
			// it can return BLOCKED, spinning in Executor::RescheduleTask.
			if (context.client.IsInterrupted())
				throw InterruptException();
			attempted = 0;
		}
	}
	output.SetCardinality(emitted);
	D_ASSERT(emitted);
	return SourceResultType::HAVE_MORE_OUTPUT;
}

void PhysicalSemanticNestedLoopJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta) {
	inputs->left.Reset();
	inputs->right.Reset();
	op_state.reset();
	sink_state.reset();
	meta.GetState().SetPipelineSource(current, *this);
	// Register both producers in the native executor graph without a build
	// completion dependency on the real consumer. The API adds a dependency
	// to its parent, so retain an unscheduled registration anchor instead.
	registration_anchor = make_shared_ptr<Pipeline>(meta.GetExecutor());
	for (auto &child : children) {
		auto &producer = meta.CreateChildMetaPipeline(*registration_anchor, child.get());
		producer.Build(child.get().children[0].get());
	}
}

vector<const_reference<PhysicalOperator>> PhysicalSemanticNestedLoopJoin::GetSources() const {
	return {*this};
}

} // namespace join_detail

namespace {
using namespace join_detail;
// A zero-copy pipeline finalizer releases blocked producers on LIMIT/early
// termination. It never buffers output and leaves model calls in the source.
class PhysicalSemanticJoinCompletion : public PhysicalOperator {
public:
	PhysicalSemanticJoinCompletion(PhysicalPlan &plan, PhysicalOperator &source, unique_ptr<FunctionData> binding,
	                               shared_ptr<JoinInputs> inputs)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, source.GetTypes(), source.estimated_cardinality),
	      binding(std::move(binding)), inputs(std::move(inputs)) {
		children.push_back(source);
	}
	unique_ptr<FunctionData> binding;
	shared_ptr<JoinInputs> inputs;
	string GetName() const override {
		return "SEMANTIC_JOIN_COMPLETION";
	}
	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		output.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	bool RequiresOperatorFinalize() const override {
		return true;
	}
	OperatorFinalResultType OperatorFinalize(Pipeline &, Event &, ClientContext &context,
	                                         OperatorFinalizeInput &) const override {
		inputs->Stop();
		context.registered_state->GetOrCreate<QueryState>("semantic.runtime")
		    ->Finish(context, binding->Cast<SemanticBindData>());
		return OperatorFinalResultType::FINISHED;
	}
};
class LogicalSemanticJoin : public SemanticLogicalOperator {
public:
	LogicalSemanticJoin(unique_ptr<LogicalOperator> child, BoundFunctionExpression &marker, idx_t table_index)
	    : SemanticLogicalOperator(std::move(child), marker, table_index) {
		// A direct cross product preserves two independent child pipelines; the
		// semantic predicate is the condition that turns it into an inner join.
		if (children[0]->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
			if (children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
			    children[0]->type != LogicalOperatorType::LOGICAL_ANY_JOIN)
				throw BinderException("semantic_join requires a direct binary inner join boundary");
			// Keep the relational join intact and evaluate only its matched pairs.
			return;
		}
		auto cross = std::move(children[0]);
		children.clear();
		children.push_back(std::move(cross->children[0]));
		children.push_back(std::move(cross->children[1]));
	}
	string GetName() const override {
		return "SEMANTIC_JOIN";
	}
	vector<ColumnBinding> GetColumnBindings() override {
		auto result = children[0]->GetColumnBindings();
		if (children.size() == 1)
			return result;
		auto right = children[1]->GetColumnBindings();
		result.insert(result.end(), right.begin(), right.end());
		return result;
	}
	bool UsesCandidatePipeline(LogicalOperator &source) {
		if (source.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT)
			return true;
		// Semantic filters must run on the candidate rows before this join. They
		// cannot be absorbed as ordinary predicates by the nested-loop executor.
		for (auto current = children[0].get(); current != &source; current = current->children[0].get())
			if (current->GetName() == "SEMANTIC_FILTER" || current->GetName() == "MODEL_BINDING_ORDER")
				return true;
		return false;
	}
	void ResolveColumnBindings(ColumnBindingResolver &resolver, vector<ColumnBinding> &bindings) override {
		if (children.size() == 1)
			resolver.VisitOperator(*children[0]);
		auto &source = children.size() == 1 ? ModelPairSource(*children[0]) : *this;
		if (children.size() == 1 && UsesCandidatePipeline(source)) {
			for (auto &expr : expressions)
				resolver.VisitExpression(&expr);
			bindings = GetColumnBindings();
			return;
		}
		// Each expression executes on its own side's chunk: resolve side-local indices.
		for (idx_t i = 0; i < 2; ++i) {
			resolver.VisitOperator(*source.children[i]);
			resolver.VisitExpression(&expressions[i]);
		}
		bindings = GetColumnBindings();
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		vector<unique_ptr<Expression>> predicates;
		auto &source = children.size() == 1 ? ModelPairSource(*children[0]) : *this;
		if (children.size() == 1 && UsesCandidatePipeline(source)) {
			if (source.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
				unique_ptr<LogicalOperator> *owner = &children[0];
				while (owner->get() != &source)
					owner = &(*owner)->children[0];
				auto replacement = CreateLogicalLeftMajorCrossProduct(
				    std::move(source.children[0]), std::move(source.children[1]), source.estimated_cardinality);
				replacement->ResolveOperatorTypes();
				*owner = std::move(replacement);
			}
			auto &child_plan = planner.CreatePlan(*children[0]);
			auto &child = OrderSamplingInput(planner, *children[0], child_plan, binding->Cast<SemanticBindData>());
			return CreatePhysicalSemanticPairFilter(planner, types, child, std::move(binding), std::move(expressions));
		}
		if (children.size() == 1) {
			auto current = children[0].get();
			while (current != &source) {
				for (auto &expr : current->expressions)
					predicates.push_back(std::move(expr));
				current = current->children[0].get();
			}
		}
		auto &left = planner.CreatePlan(*source.children[0]);
		auto &right = planner.CreatePlan(*source.children[1]);
		auto inputs = make_shared_ptr<JoinInputs>(!binding->Cast<SemanticBindData>().sampling);
		auto completion_binding = binding->Copy();
		auto &result = planner.Make<PhysicalSemanticNestedLoopJoin>(
		    types, estimated_cardinality, std::move(binding), std::move(expressions), std::move(predicates), inputs);
		result.children.push_back(planner.Make<PhysicalSemanticJoinInput>(left, inputs, true));
		result.children.push_back(planner.Make<PhysicalSemanticJoinInput>(right, inputs, false));
		return planner.Make<PhysicalSemanticJoinCompletion>(result, std::move(completion_binding), inputs);
	}

protected:
	void ResolveTypes() override {
		types = children[0]->types;
		if (children.size() == 1)
			return;
		types.insert(types.end(), children[1]->types.begin(), children[1]->types.end());
	}
};

} // namespace

unique_ptr<LogicalOperator> CreateLogicalSemanticJoin(unique_ptr<LogicalOperator> child,
                                                      BoundFunctionExpression &marker, idx_t table_index) {
	return make_uniq<LogicalSemanticJoin>(std::move(child), marker, table_index);
}
}
}
