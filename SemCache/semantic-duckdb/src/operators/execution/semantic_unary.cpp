// Shared unary dataflow: filter selects original rows, projection appends a
// typed STRUCT. Both return each DataChunk directly to the downstream pipeline.
#include "operators/execution/semantic_operator_factory.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "operators/execution/semantic_execution.hpp"
#include "operators/execution/semantic_input.hpp"
#include "operators/execution/semantic_source.hpp"
#include "operators/execution/semantic_window.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "operators/plan/semantic_logical_operator.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include <deque>
#include <functional>

namespace duckdb {
namespace semantic_extension {
namespace {

class PhysicalSemanticUnaryInput : public PhysicalOperator {
public:
	PhysicalSemanticUnaryInput(PhysicalPlan &plan, PhysicalOperator &child, shared_ptr<SemanticInput> input)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, child.GetTypes(), child.estimated_cardinality),
	      input(std::move(input)) {
		children.push_back(child);
	}
	shared_ptr<SemanticInput> input;
	string GetName() const override {
		return "SEMANTIC_UNARY_INPUT";
	}
	bool IsSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return true;
	}
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &state) const override {
		return input->Publish(context.client, chunk, state.interrupt_state);
	}
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &, OperatorSinkFinalizeInput &) const override {
		input->Complete();
		return SinkFinalizeType::READY;
	}
};
struct UnaryPendingRow {
	PendingSemanticRequest pending;
	vector<Value> output_values;
	Value value;
	uint64_t input_index;
	bool ready = false;
	size_t retained_bytes = 0;
};
struct UnarySourceState : SemanticSourceState {
	DataChunk input, output;
	idx_t emitted = 0;
	unique_ptr<SemanticOperatorState> execution;
	vector<PendingSemanticRequest> pending_requests;
	vector<Value> pending_values;
	vector<bool> pending_ready;
	idx_t pending_begin = 0, pending_end = 0;
	idx_t row = 0;
	bool admitted = false, stopped = false;
	bool concurrent_execution = false, chunk_execution = false, input_complete = false;
	std::deque<UnaryPendingRow> execution_pending;
	SemanticPendingWindow execution_window;
};
class PhysicalSemanticUnary : public PhysicalOperator {
public:
	PhysicalSemanticUnary(PhysicalPlan &plan, vector<LogicalType> types, idx_t cardinality,
	                      unique_ptr<FunctionData> binding, vector<unique_ptr<Expression>> expressions, bool project,
	                      bool pair = false)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, std::move(types), cardinality),
	      binding(std::move(binding)), expressions(std::move(expressions)), project(project), pair(pair),
	      input(make_shared_ptr<SemanticInput>(!this->binding->Cast<SemanticBindData>().sampling)) {
	}

	unique_ptr<FunctionData> binding;
	vector<unique_ptr<Expression>> expressions;
	const bool project;
	const bool pair;
	shared_ptr<SemanticInput> input;
	shared_ptr<Pipeline> registration_anchor;

	string GetName() const override {
		return pair ? "SEMANTIC_JOIN" : project ? "SEMANTIC_PROJECTION" : "SEMANTIC_FILTER";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		result["Operator"] = GetName();
		result["operator_id"] = std::to_string(binding->Cast<SemanticBindData>().node.semantic.operator_id);
		if (pair)
			result["Pair Identity"] = "INPUT_ORDINAL";
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
		auto result = make_uniq<UnarySourceState>();
		result->input.Initialize(context, children[0].get().GetTypes());
		result->output.Initialize(context, types);
		result->chunk_execution = runtime.adaptive && runtime.adaptive->ChunkMode();
		result->concurrent_execution = !config.sampling && (!runtime.adaptive || result->chunk_execution);
		return std::move(result);
	}
	bool AwaitExecution(UnarySourceState &state, bool upstream_blocked, const InterruptState &interrupt,
	                    SourceResultType &result) const {
		return state.AwaitAny(
		    state.execution_pending, [&] { return upstream_blocked && input->Ready(); }, interrupt, result);
	}
	SourceResultType GetExecutionData(ExecutionContext &context, DataChunk &output, OperatorSourceInput &source,
	                                  UnarySourceState &state) const {
		auto &config = binding->Cast<SemanticBindData>();
		auto holder = context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime");
		auto &runtime = holder->Get(context.client, config);
		std::lock_guard<std::recursive_mutex> guard(runtime.mutex);
		SemanticOperatorActiveScope active(runtime, config.node.semantic.operator_id);
		auto &counts = runtime.counts[config.node.semantic.operator_id];
		auto &local = *state.execution;
		output.Reset();
		auto &emitted = state.emitted;
		if (!emitted)
			state.output.Reset();
		// Tiny ready-first outputs otherwise become tiny admission chunks in the
		// next adaptive operator. Accumulate a few request windows while keeping
		// admission rolling, and flush early at an input boundary/upstream wait.
		const auto request_width = MinValue<idx_t>(STANDARD_VECTOR_SIZE, runtime.max_inflight_per_model);
		const auto output_width =
		    MinValue<idx_t>(STANDARD_VECTOR_SIZE, request_width * (state.chunk_execution ? 4 : 1));
		auto publish = [&] {
			state.output.SetCardinality(emitted);
			output.Reference(state.output);
			emitted = 0;
			return SourceResultType::HAVE_MORE_OUTPUT;
		};
		for (;;) {
			bool upstream_blocked = false;
			// Refill every released request slot. Adaptive execution drains only
			// the current input chunk before admitting a new configuration; ready
			// rows can already flow downstream while that chunk is still active.
			while (!state.input_complete && !upstream_blocked &&
			       state.execution_window.CanAdmit(state.execution_pending.size(), runtime.max_inflight_per_model)) {
				if (!state.admitted || state.row == state.input.size()) {
					if (state.admitted && state.chunk_execution) {
						if (!state.execution_pending.empty())
							break;
						runtime.EndChunk(context.client, config.node.semantic.operator_id);
					}
					state.admitted = false;
					state.input.Reset();
					auto status = input->Fetch(state.input, state.CompletionCallback());
					if (status != SourceResultType::HAVE_MORE_OUTPUT) {
						upstream_blocked = status == SourceResultType::BLOCKED;
						state.input_complete = !upstream_blocked;
						break;
					}
					local.arguments.Reset();
					local.executor.Execute(state.input, local.arguments);
					state.row = 0;
					state.admitted = true;
					if (state.chunk_execution)
						runtime.BeginChunk(context.client, config.node.semantic.operator_id, counts.first,
						                   state.input.size());
				}
				state.execution_pending.emplace_back();
				auto &request = state.execution_pending.back();
				request.input_index = counts.first + state.execution_pending.size() - 1;
				for (idx_t col = 0; col < state.input.ColumnCount(); ++col)
					request.output_values.push_back(state.input.GetValue(col, state.row));
				try {
					auto right = pair ? ReadRow(local.arguments.GetValue(1, state.row)) : semantic::Row();
					request.value =
					    InferSemanticValue(context, config, runtime, ReadRow(local.arguments.GetValue(0, state.row)),
					                       pair ? &right : nullptr, request.input_index, 0, UINT64_MAX, UINT64_MAX,
					                       UINT64_MAX, nullptr, &request.pending);
					request.ready = true;
				} catch (const SemanticRequestBlocked &) {
				} catch (const SamplingBudgetExhausted &) {
					state.execution_pending.pop_back();
					state.stopped = state.input_complete = true;
					input->Stop();
					break;
				}
				state.execution_window.Add(request);
				++state.row;
			}
			if (emitted >= output_width || state.execution_pending.empty()) {
				if (emitted)
					return publish();
				if (state.input_complete) {
					holder->Finish(context.client, config);
					return SourceResultType::FINISHED;
				}
				SourceResultType wait_result;
				if (AwaitExecution(state, upstream_blocked, source.interrupt_state, wait_result))
					return wait_result;
				continue;
			}
			auto completed = FirstReadySemanticRow(state.execution_pending);
			if (completed == state.execution_pending.end()) {
				if (emitted && (!state.chunk_execution || upstream_blocked || state.row == state.input.size()))
					return publish();
				SourceResultType wait_result;
				if (AwaitExecution(state, upstream_blocked, source.interrupt_state, wait_result))
					return wait_result;
				continue;
			}
			auto &request = *completed;
			if (!request.ready) {
				// The completion closure owns its prompt and telemetry. A newer
				// input chunk may already have replaced the argument vectors.
				try {
					request.value = InferSemanticValue(context, config, runtime, {}, nullptr, request.input_index, 0,
					                                   UINT64_MAX, UINT64_MAX, UINT64_MAX, nullptr, &request.pending);
				} catch (const SamplingBudgetExhausted &) {
					state.execution_pending.clear();
					state.execution_window.Clear();
					state.stopped = state.input_complete = true;
					input->Stop();
					continue;
				}
			}
			++counts.first;
			if (project || request.value.GetValue<bool>()) {
				for (idx_t col = 0; col < request.output_values.size(); ++col)
					state.output.SetValue(col, emitted, request.output_values[col]);
				if (project)
					state.output.SetValue(request.output_values.size(), emitted, request.value);
				++emitted;
				++counts.second;
			}
			state.execution_window.Remove(request);
			state.execution_pending.erase(completed);
		}
	}
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &output,
	                                 OperatorSourceInput &source) const override {
		auto &state = source.global_state.Cast<UnarySourceState>();
		auto &config = binding->Cast<SemanticBindData>();
		auto holder = context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime");
		if (!state.execution)
			state.execution = make_uniq<SemanticOperatorState>(context, expressions);
		if (state.concurrent_execution) {
			try {
				return GetExecutionData(context, output, source, state);
			} catch (...) {
				input->Stop();
				throw;
			}
		}
		auto &local = *state.execution;
		auto &emitted = state.emitted;
		// Empty output chunks are common for selective filters. Keep fetching in a
		// loop instead of recursively re-entering while holding runtime.mutex.
		for (;;) {
			if (state.stopped)
				return SourceResultType::FINISHED;
			if (!state.admitted) {
				state.input.Reset();
				auto status = input->Fetch(state.input, source.interrupt_state);
				if (status != SourceResultType::HAVE_MORE_OUTPUT) {
					if (status == SourceResultType::FINISHED)
						holder->Finish(context.client, config);
					return status;
				}
				// Input expression evaluation is local operator work outside the
				// shared runtime lock; exclude the preceding nonblocking Fetch.
				auto &preparation_runtime = holder->Ensure(context.client, config);
				SemanticOperatorActiveScope preparation(preparation_runtime, config.node.semantic.operator_id);
				state.output.Reset();
				emitted = 0;
				local.arguments.Reset();
				local.executor.Execute(state.input, local.arguments);
				state.row = 0;
				state.admitted = true;
			}
			auto &runtime = holder->Get(context.client, config);
			std::unique_lock<std::recursive_mutex> guard(runtime.mutex);
			SemanticOperatorActiveScope active(runtime, config.node.semantic.operator_id);
			auto &counts = runtime.counts[config.node.semantic.operator_id];
			if (state.row == 0 && state.pending_requests.empty() && runtime.adaptive && runtime.adaptive->ChunkMode())
				runtime.BeginChunk(context.client, config.node.semantic.operator_id, counts.first, state.input.size());
			while (state.row < state.input.size() || !state.pending_requests.empty()) {
				if (runtime.budget_exhausted && state.pending_requests.empty()) {
					state.stopped = true;
					input->Stop();
					break;
				}
				// Sampling drains independent single-row requests in input order and
				// retains the completed prefix when its physical-call budget expires.
				if (state.pending_requests.empty() && !runtime.adaptive && config.sampling) {
					vector<Value> values;
					try {
						values = InferSemanticSamples(context, config, runtime, local.arguments, state.row,
						                              state.input.size(), pair, counts.first);
					} catch (const SamplingBudgetExhausted &) {
						state.stopped = true;
						input->Stop();
						break;
					}
					for (auto &value : values) {
						++counts.first;
						if (project || value.GetValue<bool>()) {
							for (idx_t col = 0; col < state.input.ColumnCount(); ++col)
								state.output.SetValue(col, emitted, state.input.GetValue(col, state.row));
							if (project)
								state.output.SetValue(state.input.ColumnCount(), emitted, value);
							++emitted;
							++counts.second;
						}
						++state.row;
					}
					continue;
				}
				if (state.pending_requests.empty()) {
					state.pending_begin = state.row;
					// Runtime policy updates admit the next row only after observing
					// the previous one. Static plans may fill the request window.
					const auto width =
					    runtime.adaptive && !runtime.adaptive->ChunkMode() ? 1 : runtime.max_inflight_per_model;
					state.pending_end = std::min<idx_t>(state.input.size(), state.row + width);
					state.pending_requests.resize(state.pending_end - state.pending_begin);
					state.pending_values.resize(state.pending_requests.size());
					state.pending_ready.assign(state.pending_requests.size(), false);
					for (idx_t offset = 0; offset < state.pending_requests.size(); ++offset) {
						const auto row = state.pending_begin + offset;
						try {
							auto right = pair ? ReadRow(local.arguments.GetValue(1, row)) : semantic::Row();
							auto value = InferSemanticValue(
							    context, config, runtime, ReadRow(local.arguments.GetValue(0, row)),
							    pair ? &right : nullptr, counts.first + offset, 0, UINT64_MAX, UINT64_MAX, UINT64_MAX,
							    nullptr, &state.pending_requests[offset], state.CompletionCallback());
							state.pending_values[offset] = std::move(value);
							state.pending_ready[offset] = true;
						} catch (const SemanticRequestBlocked &) {
						} catch (const SamplingBudgetExhausted &) {
							state.stopped = true;
							input->Stop();
							break;
						}
					}
				}
				if (state.stopped)
					break;
				SourceResultType wait_result = SourceResultType::BLOCKED;
				bool blocked = false;
				for (idx_t offset = 0; offset < state.pending_requests.size(); ++offset) {
					if (state.pending_ready[offset])
						continue;
					const auto row = state.pending_begin + offset;
					if (state.Await(state.pending_requests[offset].response, source.interrupt_state, wait_result)) {
						blocked = true;
						break;
					}
					try {
						auto right = pair ? ReadRow(local.arguments.GetValue(1, row)) : semantic::Row();
						state.pending_values[offset] = InferSemanticValue(
						    context, config, runtime, ReadRow(local.arguments.GetValue(0, row)),
						    pair ? &right : nullptr, counts.first + offset, 0, UINT64_MAX, UINT64_MAX, UINT64_MAX,
						    nullptr, &state.pending_requests[offset], state.CompletionCallback());
						state.pending_ready[offset] = true;
					} catch (const SemanticRequestBlocked &) {
						blocked = true;
						break;
					}
				}
				if (blocked) {
					output.SetCardinality(0);
					return wait_result;
				}
				for (idx_t offset = 0; offset < state.pending_values.size(); ++offset) {
					auto &value = state.pending_values[offset];
					const auto row = state.pending_begin + offset;
					++counts.first;
					if (project || value.GetValue<bool>()) {
						for (idx_t col = 0; col < state.input.ColumnCount(); ++col)
							state.output.SetValue(col, emitted, state.input.GetValue(col, row));
						if (project)
							state.output.SetValue(state.input.ColumnCount(), emitted, value);
						++emitted;
						++counts.second;
					}
				}
				state.row = state.pending_end;
				state.pending_requests.clear();
				state.pending_values.clear();
				state.pending_ready.clear();
			}
			if (runtime.adaptive && runtime.adaptive->ChunkMode())
				runtime.EndChunk(context.client, config.node.semantic.operator_id);
			state.admitted = false;
			state.output.SetCardinality(emitted);
			if (emitted) {
				output.Reference(state.output);
				return SourceResultType::HAVE_MORE_OUTPUT;
			}
			if (state.stopped)
				return SourceResultType::FINISHED;
			// Release runtime.mutex before fetching the next input chunk.
			active.Stop();
			guard.unlock();
		}
	}
	void BuildPipelines(Pipeline &current, MetaPipeline &meta) override {
		input->Reset();
		op_state.reset();
		sink_state.reset();
		meta.GetState().SetPipelineSource(current, *this);
		registration_anchor = make_shared_ptr<Pipeline>(meta.GetExecutor());
		auto &producer = meta.CreateChildMetaPipeline(*registration_anchor, children[0].get());
		producer.Build(children[0].get().children[0].get());
	}
	vector<const_reference<PhysicalOperator>> GetSources() const override {
		return {*this};
	}
};

class PhysicalSemanticUnaryCompletion : public PhysicalOperator {
public:
	PhysicalSemanticUnaryCompletion(PhysicalPlan &plan, PhysicalOperator &child, unique_ptr<FunctionData> binding)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, child.GetTypes(), child.estimated_cardinality),
	      binding(std::move(binding)), input(child.Cast<PhysicalSemanticUnary>().input) {
		children.push_back(child);
	}
	unique_ptr<FunctionData> binding;
	shared_ptr<SemanticInput> input;
	string GetName() const override {
		return "SEMANTIC_UNARY_COMPLETION";
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
		input->Stop();
		context.registered_state->GetOrCreate<QueryState>("semantic.runtime")
		    ->Finish(context, binding->Cast<SemanticBindData>());
		return OperatorFinalResultType::FINISHED;
	}
};
class LogicalSemanticUnary : public SemanticLogicalOperator {
public:
	LogicalSemanticUnary(unique_ptr<LogicalOperator> child, BoundFunctionExpression &marker, idx_t table_index)
	    : SemanticLogicalOperator(std::move(child), marker, table_index),
	      project(binding->Cast<SemanticBindData>().node.kind == semantic::NodeKind::SEMANTIC_PROJECTION) {
	}

	const bool project;

	string GetName() const override {
		return project ? "SEMANTIC_PROJECTION" : "SEMANTIC_FILTER";
	}

	vector<ColumnBinding> GetColumnBindings() override {
		auto result = children[0]->GetColumnBindings();
		if (project)
			result.emplace_back(table_index, 0);
		return result;
	}

	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		auto &child_plan = planner.CreatePlan(*children[0]);
		auto &child = OrderSamplingInput(planner, *children[0], child_plan, binding->Cast<SemanticBindData>());
		auto completion_binding = binding->Copy();
		auto &result = planner.Make<PhysicalSemanticUnary>(types, child.estimated_cardinality, std::move(binding),
		                                                   std::move(expressions), project);
		result.children.push_back(
		    planner.Make<PhysicalSemanticUnaryInput>(child, result.Cast<PhysicalSemanticUnary>().input));
		return planner.Make<PhysicalSemanticUnaryCompletion>(result, std::move(completion_binding));
	}

protected:
	void ResolveTypes() override {
		types = children[0]->types;
		if (project)
			types.push_back(generated_type);
	}
};

}

PhysicalOperator &OrderSamplingInput(PhysicalPlanGenerator &planner, LogicalOperator &logical_child,
                                     PhysicalOperator &physical_child, const SemanticBindData &binding) {
	if (binding.llm_request_limit == UINT64_MAX)
		return physical_child;
	// Private bounded-stage IDs are fixed-width ordinals. Find their surviving
	// scan binding after native projection pruning, then sort before inference.
	auto bindings = logical_child.GetColumnBindings();
	idx_t ordinal = DConstants::INVALID_INDEX;
	std::function<void(LogicalOperator &)> find = [&](LogicalOperator &op) {
		if (ordinal != DConstants::INVALID_INDEX)
			return;
		if (op.type == LogicalOperatorType::LOGICAL_GET) {
			auto &scan = op.Cast<LogicalGet>();
			const auto &columns = scan.GetColumnIds();
			for (idx_t i = 0; i < columns.size(); ++i) {
				if (columns[i].IsVirtualColumn() || scan.GetColumnName(columns[i]) != "__sample_id")
					continue;
				for (idx_t index = 0; index < bindings.size(); ++index)
					if (bindings[index] == ColumnBinding(scan.table_index, i)) {
						ordinal = index;
						return;
					}
			}
		}
		for (auto &child : op.children)
			find(*child);
	};
	find(logical_child);
	if (ordinal == DConstants::INVALID_INDEX || physical_child.types[ordinal] != LogicalType::VARCHAR)
		throw BinderException("Bounded native sampling must preserve its input ordinal identity");
	vector<BoundOrderByNode> orders;
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
	                    make_uniq<BoundReferenceExpression>(LogicalType::VARCHAR, ordinal));
	vector<idx_t> projections;
	for (idx_t index = 0; index < physical_child.types.size(); ++index)
		projections.push_back(index);
	auto &sorted = planner.Make<PhysicalOrder>(physical_child.types, std::move(orders), std::move(projections),
	                                           physical_child.estimated_cardinality);
	sorted.children.push_back(physical_child);
	return sorted;
}

PhysicalOperator &CreatePhysicalSemanticPairFilter(PhysicalPlanGenerator &planner, vector<LogicalType> types,
                                                   PhysicalOperator &child, unique_ptr<FunctionData> binding,
                                                   vector<unique_ptr<Expression>> expressions) {
	auto completion_binding = binding->Copy();
	auto &result = planner.Make<PhysicalSemanticUnary>(std::move(types), child.estimated_cardinality,
	                                                   std::move(binding), std::move(expressions), false, true);
	result.children.push_back(
	    planner.Make<PhysicalSemanticUnaryInput>(child, result.Cast<PhysicalSemanticUnary>().input));
	return planner.Make<PhysicalSemanticUnaryCompletion>(result, std::move(completion_binding));
}

unique_ptr<LogicalOperator> CreateLogicalSemanticFilter(unique_ptr<LogicalOperator> child,
                                                        BoundFunctionExpression &marker, idx_t table_index) {
	return make_uniq<LogicalSemanticUnary>(std::move(child), marker, table_index);
}

unique_ptr<LogicalOperator> CreateLogicalSemanticProjection(unique_ptr<LogicalOperator> child,
                                                            BoundFunctionExpression &marker, idx_t table_index) {
	return make_uniq<LogicalSemanticUnary>(std::move(child), marker, table_index);
}

}
}
