#include "operators/execution/join/join_internal.hpp"

namespace duckdb {
namespace semantic_extension {
namespace join_detail {

JoinState::JoinState(ClientContext &context, const vector<unique_ptr<Expression>> &expressions,
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

PhysicalSemanticNestedLoopJoin::PhysicalSemanticNestedLoopJoin(
    PhysicalPlan &plan, vector<LogicalType> types, idx_t cardinality, unique_ptr<FunctionData> binding,
    vector<unique_ptr<Expression>> expressions, vector<unique_ptr<Expression>> predicates, shared_ptr<JoinInputs> inputs)
    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, std::move(types), cardinality),
      binding(std::move(binding)), expressions(std::move(expressions)), predicates(std::move(predicates)),
      inputs(std::move(inputs)) {
}

InsertionOrderPreservingMap<string> PhysicalSemanticNestedLoopJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["operator_id"] = std::to_string(binding->Cast<SemanticBindData>().node.semantic.operator_id);
	result["Pair Identity"] = "LEFT_MAJOR";
	result["Right Input"] = "STREAM_FIRST_LEFT_THEN_REPLAY";
	result["Join Type"] = "INNER";
	result["Candidate Predicates"] = std::to_string(predicates.size());
	return result;
}

OrderPreservationType PhysicalSemanticNestedLoopJoin::SourceOrder() const {
	return binding->Cast<SemanticBindData>().sampling ? OrderPreservationType::FIXED_ORDER
	                                                  : OrderPreservationType::NO_ORDER;
}

unique_ptr<GlobalSourceState> PhysicalSemanticNestedLoopJoin::GetGlobalSourceState(ClientContext &context) const {
	auto &config = binding->Cast<SemanticBindData>();
	auto &runtime = context.registered_state->GetOrCreate<QueryState>("semantic.runtime")->Ensure(context, config);
	auto result = make_uniq<JoinState>(context, expressions, predicates, children[0].get().GetTypes(),
	                                   children[1].get().GetTypes(), types);
	result->chunk_execution = runtime.adaptive && runtime.adaptive->ChunkMode();
	result->concurrent_execution = !config.sampling && (!runtime.adaptive || result->chunk_execution);
	return std::move(result);
}

SourceResultType PhysicalSemanticNestedLoopJoin::PrepareExecutionCandidate(JoinState &local,
                                                                           SemanticInput *&blocked_input) const {
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
				local.candidates.data[local.left_chunk.ColumnCount() + col].Reference(local.right_chunk.data[col]);
			local.candidates.SetCardinality(local.right_chunk.size());
			local.candidate_executor.Execute(local.candidates, local.candidate_results);
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
}

bool PhysicalSemanticNestedLoopJoin::AwaitExecution(JoinState &local, SemanticInput *upstream,
                                                    const InterruptState &interrupt, SourceResultType &result) const {
	return local.AwaitAny(
	    local.execution_pending, [&] { return upstream && upstream->Ready(); }, interrupt, result);
}

SourceResultType PhysicalSemanticNestedLoopJoin::GetExecutionData(ExecutionContext &context, DataChunk &output,
                                                                  OperatorSourceInput &input, JoinState &local) const {
	output.Reset();
	auto &config = binding->Cast<SemanticBindData>();
	auto holder = context.client.registered_state->GetOrCreate<QueryState>("semantic.runtime");
	if (local.execution_input_complete && local.staged_chunk.empty() && local.execution_pending.empty()) {
		holder->Finish(context.client, config);
		return SourceResultType::FINISHED;
	}
	auto &runtime = holder->Get(context.client, config);
	std::lock_guard<std::recursive_mutex> guard(runtime.mutex);
	SemanticOperatorActiveScope active(runtime, config.node.semantic.operator_id);
	auto &counts = runtime.counts[config.node.semantic.operator_id];
	if (!local.admitted && !local.right_complete && !local.execution_input_complete)
		inputs->right.Request();
	idx_t emitted = 0, attempted = 0;
	const idx_t output_width =
	    MinValue<idx_t>(STANDARD_VECTOR_SIZE, runtime.max_inflight_per_model * (local.chunk_execution ? 4 : 1));
	SemanticInput *blocked_input = nullptr;
	for (;;) {
		if (local.chunk_execution && local.chunk_active && local.staged_chunk.empty()) {
			runtime.Reoptimize(context.client);
			if (local.execution_pending.empty()) {
				runtime.EndChunk(context.client, config.node.semantic.operator_id);
				local.chunk_active = false;
			}
		}
		if (local.chunk_execution && local.staged_chunk.empty() && !local.execution_input_complete &&
		    !local.budget_stopped && !blocked_input &&
		    (!local.chunk_active || !runtime.HasPendingPlanUpdate(config.node.semantic.operator_id))) {
			while (local.staged_chunk.size() < JoinState::SEMANTIC_JOIN_CHUNK_SIZE) {
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
				if (!eligible)
					continue;
				StagedJoinPair staged;
				staged.left = local.left_offset + local.left_row;
				staged.right = local.right_offset + candidate_row;
				staged.pair = staged.left ? staged.left * local.right.Count() + staged.right : staged.right;
				staged.output_values.reserve(local.left_chunk.ColumnCount() + local.right_chunk.ColumnCount());
				for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
					staged.output_values.push_back(local.left_chunk.GetValue(col, local.left_row));
				for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
					staged.output_values.push_back(local.right_chunk.GetValue(col, candidate_row));
				staged.left_values = local.left_values;
				staged.right_values = local.right_rows[candidate_row];
				local.staged_chunk.push_back(std::move(staged));
			}
			if (!local.staged_chunk.empty()) {
				blocked_input = nullptr;
				if (!local.chunk_active) {
					runtime.BeginChunk(context.client, config.node.semantic.operator_id, counts.first,
					                   local.staged_chunk.size());
					local.chunk_active = true;
				} else {
					runtime.ExtendChunk(context.client, config.node.semantic.operator_id, local.staged_chunk.size());
				}
			}
		}
		// Any completion frees one slot. Refill it before returning available
		// results, even while an earlier row or input chunk remains in flight.
		while ((local.chunk_execution ? !local.staged_chunk.empty()
		                              : (!local.execution_input_complete && !blocked_input)) &&
		       local.execution_window.CanAdmit(local.execution_pending.size(), runtime.max_inflight_per_model)) {
			if (local.chunk_execution) {
				auto staged = std::move(local.staged_chunk.front());
				local.staged_chunk.pop_front();
				local.execution_pending.emplace_back();
				auto &request = local.execution_pending.back();
				request.input_index = counts.first + local.execution_pending.size() - 1;
				request.output_values = std::move(staged.output_values);
				try {
					request.value = InferSemanticValue(
					    context, config, runtime, staged.left_values, &staged.right_values, request.input_index,
					    local.right_complete ? local.right.Count() : 0, staged.pair, staged.left, staged.right,
					    nullptr, &request.pending);
					request.ready = true;
				} catch (const SemanticRequestBlocked &) {
				} catch (const SamplingBudgetExhausted &) {
					local.execution_pending.pop_back();
					local.staged_chunk.clear();
					local.budget_stopped = local.execution_input_complete = true;
					inputs->Stop();
					break;
				}
				local.execution_window.Add(request);
				continue;
			}
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
			request.output_values.reserve(local.left_chunk.ColumnCount() + local.right_chunk.ColumnCount());
			for (idx_t col = 0; col < local.left_chunk.ColumnCount(); ++col)
				request.output_values.push_back(local.left_chunk.GetValue(col, local.left_row));
			for (idx_t col = 0; col < local.right_chunk.ColumnCount(); ++col)
				request.output_values.push_back(local.right_chunk.GetValue(col, candidate_row));
			try {
				request.value = InferSemanticValue(
				    context, config, runtime, local.left_values, &local.right_rows[candidate_row],
				    request.input_index, local.right_complete ? local.right.Count() : 0, pair, left, right, nullptr,
				    &request.pending);
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
		if (emitted >= output_width || (local.execution_pending.empty() && local.staged_chunk.empty())) {
			output.SetCardinality(emitted);
			if (emitted)
				return SourceResultType::HAVE_MORE_OUTPUT;
			if (local.execution_input_complete) {
				if (local.chunk_execution && local.chunk_active) {
					runtime.EndChunk(context.client, config.node.semantic.operator_id);
					local.chunk_active = false;
				}
				holder->Finish(context.client, config);
				return SourceResultType::FINISHED;
			}
			SourceResultType wait_result;
			if (AwaitExecution(local, blocked_input, input.interrupt_state, wait_result))
				return wait_result;
			blocked_input = nullptr;
			continue;
		}
		if (local.execution_pending.empty())
			continue;
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
				    InferSemanticValue(context, config, runtime, {}, nullptr, request.input_index, 0,
				                       UINT64_MAX, UINT64_MAX, UINT64_MAX, nullptr, &request.pending);
			} catch (const SamplingBudgetExhausted &) {
				// Transport can exhaust the allowance after CPU admission. The
				// failing row and its tail cannot extend the validated prefix.
				local.budget_stopped = local.execution_input_complete = true;
				local.staged_chunk.clear();
				local.execution_pending.clear();
				local.execution_window.Clear();
				inputs->Stop();
				holder->Finish(context.client, config);
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
		if (local.chunk_execution && local.chunk_active && local.staged_chunk.empty() &&
		    local.execution_pending.empty()) {
			runtime.EndChunk(context.client, config.node.semantic.operator_id);
			local.chunk_active = false;
		}
	}
}

} // namespace join_detail
} // namespace semantic_extension
} // namespace duckdb
