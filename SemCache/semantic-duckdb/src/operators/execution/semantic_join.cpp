// Binary semantic inner join: request one left chunk, stream the right side for
// its first row, then replay the cached right input for later left rows.
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
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include <deque>

namespace duckdb {
namespace semantic_extension {
namespace {
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
struct JoinState : SemanticSourceState {
	PendingSemanticRequest pending;
	std::deque<JoinExecutionRequest> execution_pending;
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
	bool concurrent_execution = false;
	JoinState(ClientContext &context, const vector<unique_ptr<Expression>> &expressions,
	          const vector<unique_ptr<Expression>> &predicates, const vector<LogicalType> &left_types,
	          const vector<LogicalType> &right_types, const vector<LogicalType> &types)
	    : right(context, right_types), left_executor(context), right_executor(context), candidate_executor(context) {
		right.InitializeAppend(append);
		left_chunk.Initialize(context, left_types);
		right.InitializeScanChunk(right_chunk);
		left_executor.AddExpression(*expressions[0]);
		right_executor.AddExpression(*expressions[1]);
		left_arguments.Initialize(context, {expressions[0]->return_type});
		right_arguments.Initialize(context, {expressions[1]->return_type});
		if (!predicates.empty()) {
			vector<LogicalType> outputs;
			for (auto &expr : predicates) {
				candidate_executor.AddExpression(*expr);
				outputs.push_back(expr->return_type);
			}
			candidates.Initialize(context, types);
			candidate_results.Initialize(context, outputs);
		}
	}
};
class PhysicalSemanticNestedLoopJoin : public PhysicalOperator {
public:
	PhysicalSemanticNestedLoopJoin(PhysicalPlan &plan, vector<LogicalType> types, idx_t cardinality,
	                               unique_ptr<FunctionData> binding, vector<unique_ptr<Expression>> expressions,
	                               vector<unique_ptr<Expression>> predicates, shared_ptr<JoinInputs> inputs)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, std::move(types), cardinality),
	      binding(std::move(binding)), expressions(std::move(expressions)), predicates(std::move(predicates)),
	      inputs(std::move(inputs)) {
	}
	unique_ptr<FunctionData> binding;
	vector<unique_ptr<Expression>> expressions, predicates;
	shared_ptr<JoinInputs> inputs;
	shared_ptr<Pipeline> registration_anchor;
	string GetName() const override {
		return "SEMANTIC_JOIN";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		result["operator_id"] = std::to_string(binding->Cast<SemanticBindData>().node.semantic.operator_id);
		result["Pair Identity"] = "LEFT_MAJOR";
		result["Right Input"] = "STREAM_FIRST_LEFT_THEN_REPLAY";
		result["Join Type"] = "INNER";
		result["Candidate Predicates"] = std::to_string(predicates.size());
		return result;
	}
	bool IsSource() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return binding->Cast<SemanticBindData>().sampling ? OrderPreservationType::FIXED_ORDER
		                                                  : OrderPreservationType::NO_ORDER;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		auto &config = binding->Cast<SemanticBindData>();
		auto &runtime = context.registered_state->GetOrCreate<QueryState>("semantic.runtime")->Ensure(context, config);
		auto result = make_uniq<JoinState>(context, expressions, predicates, children[0].get().GetTypes(),
		                                   children[1].get().GetTypes(), types);
		result->concurrent_execution = !config.sampling && !runtime.adaptive;
		return std::move(result);
	}
	// Advance only the input cursor. Pending requests own their output rows, so
	// crossing a right chunk or a left row never waits for the preceding window.
	SourceResultType PrepareExecutionCandidate(JoinState &local, SemanticInput *&blocked_input) const {
		for (;;) {
			if (!local.admitted) {
				local.left_chunk.Reset();
				auto status = inputs->left.Fetch(local.left_chunk, local.CompletionCallback());
				if (status != SourceResultType::HAVE_MORE_OUTPUT) {
					if (status == SourceResultType::BLOCKED)
						blocked_input = &inputs->left;
					if (status == SourceResultType::FINISHED)
						inputs->Stop();
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
			if (local.right_row < local.right_chunk.size())
				return SourceResultType::HAVE_MORE_OUTPUT;
			local.right_offset += local.right_chunk.size();
			local.right_chunk.Reset();
			local.right_row = 0;
			if (!local.right_complete) {
				auto status = inputs->right.Fetch(local.right_chunk, local.CompletionCallback());
				if (status == SourceResultType::BLOCKED) {
					blocked_input = &inputs->right;
					return status;
				}
				if (status == SourceResultType::FINISHED) {
					local.right_complete = true;
					if (!local.right.Count()) {
						inputs->Stop();
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
			if (!predicates.empty()) {
				local.candidates.Reset();
				local.candidate_results.Reset();
				for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
					local.candidates.data[col].Reference(local.left_chunk.GetValue(col, local.left_row));
				for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
					local.candidates.data[local.left_chunk.ColumnCount() + col].Reference(local.right_chunk.data[col]);
				local.candidates.SetCardinality(local.right_chunk.size());
				local.candidate_executor.Execute(local.candidates, local.candidate_results);
			}
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
	}
	bool AwaitExecution(JoinState &local, SemanticInput *upstream, const InterruptState &interrupt,
	                    SourceResultType &result) const {
		return local.AwaitAny(
		    local.execution_pending, [&] { return upstream && upstream->Ready(); }, interrupt, result);
	}
	SourceResultType GetExecutionData(ExecutionContext &context, DataChunk &output, OperatorSourceInput &input,
	                                  JoinState &local) const {
		output.Reset();
		auto &config = binding->Cast<SemanticBindData>();
		auto &runtime =
		    context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime")->Get(context.client, config);
		std::lock_guard<std::recursive_mutex> guard(runtime.mutex);
		SemanticOperatorActiveScope active(runtime, config.node.semantic.operator_id);
		auto &counts = runtime.counts[config.node.semantic.operator_id];
		if (!local.admitted && !local.right_complete && !local.execution_input_complete)
			inputs->right.Request();
		idx_t emitted = 0, attempted = 0;
		SemanticInput *blocked_input = nullptr;
		for (;;) {
			// Any completion frees one slot. Refill it before returning available
			// results, even while an earlier row or input chunk remains in flight.
			while (!local.execution_input_complete && !blocked_input &&
			       local.execution_window.CanAdmit(local.execution_pending.size(), runtime.max_inflight_per_model)) {
				auto status = PrepareExecutionCandidate(local, blocked_input);
				if (status != SourceResultType::HAVE_MORE_OUTPUT) {
					local.execution_input_complete = status == SourceResultType::FINISHED;
					break;
				}
				const auto candidate_row = local.right_row++;
				bool eligible = true;
				for (idx_t col = 0; col < local.candidate_results.ColumnCount(); ++col) {
					auto value = local.candidate_results.GetValue(col, candidate_row);
					eligible &= !value.IsNull() && value.GetValue<bool>();
				}
				if (++attempted >= STANDARD_VECTOR_SIZE) {
					if (context.client.IsInterrupted())
						throw InterruptException();
					attempted = 0;
				}
				if (!eligible) {
					if (!attempted && !local.execution_pending.empty())
						break;
					continue;
				}
				const auto left = local.left_offset + local.left_row;
				const auto right = local.right_offset + candidate_row;
				const auto pair = left ? left * local.right.Count() + right : right;
				local.execution_pending.emplace_back();
				auto &request = local.execution_pending.back();
				request.input_index = counts.first + local.execution_pending.size() - 1;
				for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
					request.output_values.push_back(local.left_chunk.GetValue(col, local.left_row));
				for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
					request.output_values.push_back(local.right_chunk.GetValue(col, candidate_row));
				auto right_value = ReadRow(local.right_arguments.GetValue(0, candidate_row));
				try {
					request.value = InferSemanticValue(
					    context, config, runtime, local.left_values, &right_value, request.input_index,
					    local.right_complete ? local.right.Count() : 0, pair, left, right, nullptr, &request.pending);
					request.ready = true;
				} catch (const SemanticRequestBlocked &) {
				} catch (const SamplingBudgetExhausted &) {
					// The failed admission has no response; drain the already admitted
					// prefix, then finish without touching another upstream chunk.
					local.execution_pending.pop_back();
					local.budget_stopped = local.execution_input_complete = true;
					inputs->Stop();
					break;
				}
				local.execution_window.Add(request);
				if (!attempted)
					break;
			}
			if (emitted >= MinValue<idx_t>(STANDARD_VECTOR_SIZE, runtime.max_inflight_per_model) ||
			    local.execution_pending.empty()) {
				output.SetCardinality(emitted);
				if (emitted)
					return SourceResultType::HAVE_MORE_OUTPUT;
				if (local.execution_input_complete)
					return SourceResultType::FINISHED;
				SourceResultType wait_result;
				if (AwaitExecution(local, blocked_input, input.interrupt_state, wait_result))
					return wait_result;
				blocked_input = nullptr;
				continue;
			}
			auto completed = FirstReadySemanticRow(local.execution_pending);
			if (completed == local.execution_pending.end()) {
				if (emitted) {
					output.SetCardinality(emitted);
					return SourceResultType::HAVE_MORE_OUTPUT;
				}
				SourceResultType wait_result;
				if (AwaitExecution(local, blocked_input, input.interrupt_state, wait_result))
					return wait_result;
				blocked_input = nullptr;
				continue;
			}
			auto &request = *completed;
			if (!request.ready) {
				// Completion owns its immutable prompt/event data. It does not use
				// the input cursor, which may already refer to another left row.
				try {
					request.value =
					    InferSemanticValue(context, config, runtime, local.left_values, nullptr, request.input_index, 0,
					                       UINT64_MAX, UINT64_MAX, UINT64_MAX, nullptr, &request.pending);
				} catch (const SamplingBudgetExhausted &) {
					// Transport can exhaust the allowance after CPU admission. The
					// failing row and its tail cannot extend the validated prefix.
					local.budget_stopped = local.execution_input_complete = true;
					local.execution_pending.clear();
					local.execution_window.Clear();
					inputs->Stop();
					output.SetCardinality(emitted);
					return emitted ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
				}
			}
			++counts.first;
			if (request.value.GetValue<bool>()) {
				for (idx_t col = 0; col < request.output_values.size(); ++col)
					output.SetValue(col, emitted, request.output_values[col]);
				++emitted;
				++counts.second;
			}
			local.execution_window.Remove(request);
			local.execution_pending.erase(completed);
		}
	}
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &output,
	                                 OperatorSourceInput &input) const override {
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
					if (status == SourceResultType::FINISHED)
						inputs->Stop();
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
				auto right_value = ReadRow(local.right_arguments.GetValue(0, local.right_row));
				Value value;
				try {
					value = InferSemanticValue(context, config, *runtime, local.left_values, &right_value, counts.first,
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
	void BuildPipelines(Pipeline &current, MetaPipeline &meta) override {
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
	vector<const_reference<PhysicalOperator>> GetSources() const override {
		return {*this};
	}
};
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
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		vector<unique_ptr<Expression>> predicates;
		auto &source = children.size() == 1 ? ModelPairSource(*children[0]) : *this;
		const auto options = GetSetting(context, "semantic_runtime");
		if (children.size() == 2 && !options.empty() &&
		    semantic::llm_detail::ParseJSON(options).value("version", 1) == 2) {
			// In chunk-adaptive mode, DuckDB enumerates candidate-pair chunks;
			// the semantic predicate remains one logical join with a concurrent
			// pair-filter implementation. Relational ON joins remain intact.
			auto &left = planner.CreatePlan(*source.children[0]);
			auto &right = planner.CreatePlan(*source.children[1]);
			std::function<void(Expression &)> shift = [&](Expression &expr) {
				if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF)
					expr.Cast<BoundReferenceExpression>().index += left.GetTypes().size();
				ExpressionIterator::EnumerateChildren(expr, shift);
			};
			shift(*expressions[1]);
			auto &pairs = planner.Make<PhysicalCrossProduct>(types, left, right, estimated_cardinality);
			return CreatePhysicalSemanticPairFilter(planner, types, pairs, std::move(binding), std::move(expressions));
		}
		if (children.size() == 1 && UsesCandidatePipeline(source)) {
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
}
unique_ptr<LogicalOperator> CreateLogicalSemanticJoin(unique_ptr<LogicalOperator> child,
                                                      BoundFunctionExpression &marker, idx_t table_index) {
	return make_uniq<LogicalSemanticJoin>(std::move(child), marker, table_index);
}
}
}
