// The only operator path that renders prompts and invokes the query scheduler;
// keeping it shared guarantees identical audit semantics for all operator kinds.
#include "operators/execution/semantic_execution.hpp"
#include "duckdb/common/crypto/md5.hpp"
#include "prompt/prompt_renderer.hpp"

namespace duckdb {
namespace semantic_extension {

SemanticOperatorState::SemanticOperatorState(ExecutionContext &context,
                                             const vector<unique_ptr<Expression>> &expressions)
    : executor(context.client), selection(STANDARD_VECTOR_SIZE) {
	vector<LogicalType> types;
	for (auto &expression : expressions) {
		executor.AddExpression(*expression);
		types.push_back(expression->return_type);
	}
	arguments.Initialize(context.client, types);
}

void RecordCacheCorrection(Json &event, const semantic::LLMResponse &response) {
	if (!response.cache_result_confirmed && !response.cache_correction_calls &&
	    response.cache_correction_metadata_json.empty())
		return;
	event["cache_result_confirmed"] = response.cache_result_confirmed;
	event["cache_correction_calls"] = response.cache_correction_calls;
	event["cache_correction_fallback"] = response.cache_correction_fallback;
	if (!response.cache_correction_metadata_json.empty())
		event["cache_correction"] = semantic::llm_detail::ParseJSON(response.cache_correction_metadata_json);
}

void RecordSemanticRequestFailure(QueryRuntime &runtime, Json &event, const std::exception &error,
                                  const semantic::LLMResponse *response) {
	if (runtime.file.is_open()) {
		if (response) {
			RecordCacheCorrection(event, *response);
			event["submit_ns"] = response->submit_ns;
			event["response_ns"] = response->response_ns;
			event["raw_result"] = response->raw_text;
			if (!response->output_normalization.empty()) {
				event["provider_raw_result"] = response->original_raw_text;
				event["result_normalization"] = response->output_normalization;
			}
			event["latency_ms"] = double(response->response_ns - response->submit_ns) / 1000000.0;
			event["input_tokens"] = response->input_tokens;
			event["output_tokens"] = response->output_tokens;
		}
		event["parse_status"] = "failed";
		event["error"] = error.what();
		runtime.Log(event);
	}
}

[[noreturn]] void FailSemanticRequest(const SemanticBindData &binding, QueryRuntime &runtime, Json &event,
                                      uint64_t sequence, const std::exception &error,
                                      const semantic::LLMResponse *response) {
	RecordSemanticRequestFailure(runtime, event, error, response);
	throw SemanticRequestFailure("semantic query=%s operator=%lld request_seq=%llu: %s",
	                             binding.execution_context.query_id, binding.node.semantic.operator_id, sequence,
	                             error.what());
}

Value InferSemanticValue(ExecutionContext &context, const SemanticBindData &binding, QueryRuntime &runtime,
                         const semantic::Row &left, optional_ptr<const semantic::Row> right, uint64_t input_index,
                         uint64_t right_count, uint64_t pair_index, uint64_t left_index, uint64_t right_index,
                         semantic::RenderedPrompt *prepared_prompt, PendingSemanticRequest *pending,
                         std::function<void()> ready) {
	if (pending && pending->complete) {
		if (pending->response.valid() &&
		    pending->response.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			throw SemanticRequestBlocked();
		auto complete = std::move(pending->complete);
		pending->response = {};
		pending->notify_when_ready = {};
		return complete();
	}
	if (context.client.IsInterrupted())
		throw InterruptException();
	runtime.Reoptimize(context.client);
	auto operator_id = binding.node.semantic.operator_id;
	// Retain the immutable admission snapshot through transport and parsing. A
	// rule may update this RUNNING operator after the call, before its batch ends.
	auto batch = runtime.adaptive ? runtime.adaptive->Admit(operator_id, input_index) : nullptr;
	const auto &node = batch ? batch->node : binding.node;
	auto sequence = runtime.sequence + 1;
	const bool telemetry = runtime.file.is_open();
	Json event;
	if (telemetry) {
		event = {{"event", "semantic_row"},
		         {"query_id", binding.execution_context.query_id},
		         {"operator_id", operator_id},
		         {"request_seq", sequence},
		         {"model_id", node.semantic.model_id},
		         {"input_index", input_index},
		         {"operator_kind", semantic::OperatorName(binding.node.kind)}};
		event[binding.node.kind == semantic::NodeKind::SEMANTIC_JOIN ? "pair_id" : "row_id"] = input_index;
		if (left_index != UINT64_MAX) {
			event["pair_id"] = pair_index;
			event["left_index"] = left_index;
			event["right_index"] = right_index;
			event["right_count"] = right_count ? Json(right_count) : Json(nullptr);
		} else if (right_count) {
			auto pair = pair_index == UINT64_MAX ? input_index : pair_index;
			event["pair_id"] = pair;
			event["left_index"] = pair / right_count;
			event["right_index"] = pair % right_count;
			event["right_count"] = right_count;
		} else if (right) {
			// A relational join supplies already-filtered row pairs. Side row
			// identities are unavailable; retain each requested pair separately.
			event["pair_input"] = true;
			event["left_index"] = input_index;
			event["right_index"] = input_index;
		}
		event["block_order"] = Json::array();
		for (auto block : node.semantic.layout.block_order)
			event["block_order"].push_back(semantic::BlockName(block));
		event["field_order"] = node.semantic.layout.field_order;
		event["left_field_order"] = node.semantic.layout.left_field_order;
		event["right_field_order"] = node.semantic.layout.right_field_order;
		if (batch) {
			event["batch_seq"] = batch->sequence;
			event["batch_index"] = input_index - batch->first_input;
			event["config_generation"] = batch->generation;
		}
	}

	try {
		auto &prepared = runtime.prompts[operator_id];
		const auto generation = batch ? batch->generation : 0;
		if (!prepared.second || prepared.first != generation) {
			prepared.second = make_uniq<semantic::PromptRenderer>(binding.execution_context.global_system_prefix, node);
			prepared.first = generation;
		}
		const bool cache_enabled = runtime.client.ResultCacheEnabled();
		auto prompt = prepared_prompt ? std::move(*prepared_prompt)
		                              : (right ? prepared.second->RenderPair(left, *right, false, cache_enabled)
		                                       : prepared.second->RenderRow(left, false, cache_enabled));
		semantic::LLMRequest request;
		request.query_id = binding.execution_context.query_id;
		request.operator_id = operator_id;
		request.model_id = node.semantic.model_id;
		request.output_schema = node.semantic.output_schema;
		// Bind only the fixed contract here; no per-row JSON cache key is built.
		request.system_text = std::move(prompt.system_text);
		request.user_text = std::move(prompt.user_text);
		request.semantic_cache_key = std::move(prompt.semantic_cache_key);
		if (binding.result_cache_probe) {
			// Use the ordinary renderer and transport cache contract. Confirmed
			// hits only; this branch never admits a request to the scheduler.
			const auto cache_calls = runtime.client.ConfigureCacheProbe(request);
			MD5Context hash;
			hash.Add(request.result_cache_key.empty() ? request.system_text + request.user_text
			                                          : request.result_cache_key);
			runtime.Log({{"event", "semantic_result_cache_probe"},
			             {"prompt_key", hash.FinishHex()},
			             {"hit", cache_calls == 0},
			             {"cache_calls", cache_calls},
			             {"request_reuse", runtime.reuse_results || cache_enabled},
			             {"operator_id", operator_id}});
			if (binding.node.kind != semantic::NodeKind::SEMANTIC_PROJECTION)
				return Value::BOOLEAN(false);
			child_list_t<LogicalType> fields;
			for (const auto &field : request.output_schema.fields)
				fields.emplace_back(field.name, ToLogicalType(field));
			return Value(LogicalType::STRUCT(std::move(fields)));
		}
		auto &results = runtime.ResultBucket(request, generation);
		const bool reuse = runtime.reuse_results && results.Enabled() && request.temperature == 0;
		auto cached = reuse ? results.find(request.user_text) : results.end();
		if (cached != results.end()) {
			auto value = cached->second.value;
			const auto source_sequence = cached->second.source_request_seq;
			event["display_to_input"] = prompt.display_to_input;
			auto client_context = &context.client;
			auto finish_cached = [&runtime, &binding, value, operator_id, telemetry, event, source_sequence,
			                      client_context]() mutable -> Value {
				const bool emitted =
				    binding.node.kind == semantic::NodeKind::SEMANTIC_PROJECTION || value.GetValue<bool>();
				++runtime.result_reuses[operator_id];
				if (telemetry) {
					event.erase("request_seq");
					event["event"] = "semantic_result_reuse";
					event["source_request_seq"] = source_sequence;
					event["after_request_seq"] = runtime.sequence;
					event["parse_status"] = "validated";
					event["emitted"] = emitted;
					event["latency_ms"] = 0;
					runtime.Log(event);
				}
				if (runtime.adaptive)
					runtime.adaptive->Observe(operator_id, emitted, 0, false);
				runtime.Reoptimize(*client_context);
				return value;
			};
			if (pending) {
				pending->response = {};
				pending->notify_when_ready = {};
				pending->complete = std::move(finish_cached);
				throw SemanticRequestBlocked();
			}
			return finish_cached();
		}
		// A different operator can reach the same immutable contract while its
		// transport is pending. Share that response instead of spending a second call.
		if (reuse) {
			auto &pending_results = runtime.in_flight[&results];
			auto found = pending_results.find(request.user_text);
			if (found != pending_results.end()) {
				auto shared = found->second;
				if (pending)
					pending->shared_flight = shared;
				if (pending && !ready)
					pending->notify_when_ready = [shared](std::function<void()> callback) {
						shared->Listen(std::move(callback));
					};
				if (ready)
					shared->Listen(std::move(ready));
				auto future = shared->response;
				auto source_sequence = shared->sequence;
				event["display_to_input"] = prompt.display_to_input;
				auto finish_reuse = [&runtime, &binding, operator_id, telemetry, event, shared,
				                     source_sequence]() mutable -> Value {
					try {
						auto value = shared->Publish(true);
						++runtime.result_reuses[operator_id];
						const bool emitted =
						    binding.node.kind == semantic::NodeKind::SEMANTIC_PROJECTION || value.GetValue<bool>();
						if (telemetry) {
							event.erase("request_seq");
							event["event"] = "semantic_result_reuse";
							event["source_request_seq"] = source_sequence;
							event["after_request_seq"] = runtime.sequence;
							event["parse_status"] = "validated";
							event["emitted"] = emitted;
							event["latency_ms"] = 0;
							runtime.Log(event);
						}
						if (runtime.adaptive)
							runtime.adaptive->Observe(operator_id, emitted, 0, false);
						return value;
					} catch (const SamplingBudgetExhausted &) {
						throw;
					} catch (const SemanticRequestFailure &) {
						throw;
					} catch (const semantic::LLMError &error) {
						FailSemanticRequest(binding, runtime, event, source_sequence, error,
						                    error.confirmed_response.get());
					} catch (const std::exception &error) {
						FailSemanticRequest(binding, runtime, event, source_sequence, error);
					}
				};
				if (pending) {
					pending->response = future;
					pending->complete = std::move(finish_reuse);
					throw SemanticRequestBlocked();
				}
				return finish_reuse();
			}
		}
		runtime.ReserveRequestAllowance();
		sequence = ++runtime.sequence;
		if (telemetry && prompt.canonical_text_for_hashing.empty())
			prompt.canonical_text_for_hashing =
			    "semantic-prompt-v4\nsystem:" + std::to_string(request.system_text.size()) + "\n" +
			    request.system_text + "\nuser:" + std::to_string(request.user_text.size()) + "\n" + request.user_text;
		if (telemetry) {
			event["system_text"] = request.system_text;
			event["user_text"] = request.user_text;
			// Statistics consume original bound values rather than reparsing tags
			// that can also appear inside a user-authored task description.
			event["operator_specification"] = semantic::OperatorSpecification(binding.node.kind);
			event["input_values"] = Json::array();
			const auto &fields = right ? node.semantic.left_input_fields : node.semantic.input_fields;
			for (const auto &field : fields)
				event["input_values"].push_back(left.at(field).Serialize());
			if (right) {
				for (const auto &field : node.semantic.right_input_fields)
					event["input_values"].push_back(right->at(field).Serialize());
			}
			event["prompt_serialization_version"] = 4;
			event["display_to_input"] = prompt.display_to_input;
			MD5Context hash;
			hash.Add(prompt.canonical_text_for_hashing);
			event["prompt_hash"] = hash.FinishHex();
			event["prompt_hash_algorithm"] = "md5";
			event["prompt_bytes"] = prompt.canonical_text_for_hashing.size();
		}

		auto user_key = reuse ? request.user_text : string();
		auto results_ptr = &results;
		auto shared = std::make_shared<InFlightSemanticResult>();
		shared->sequence = sequence;
		if (pending) {
			pending->shared_flight = shared;
			// Prompt, exact-key and audit copies retained by the one owner.
			pending->retained_bytes = request.user_text.size() * 4 + request.system_text.size() * 2;
		}
		if (pending && !ready)
			pending->notify_when_ready = [shared](std::function<void()> callback) {
				shared->Listen(std::move(callback));
			};
		if (ready)
			shared->Listen(std::move(ready));
		auto response_future = runtime.scheduler.Submit(std::move(request), [shared]() { shared->Wake(); });
		shared->response = response_future;
		if (reuse)
			runtime.in_flight[results_ptr].emplace(user_key, shared);
		auto client_context = &context.client;
		auto finish = [&runtime, &binding, client_context, operator_id, sequence, telemetry, reuse, results_ptr, event,
		               user_key, response_future, flight = shared.get()]() mutable -> Value {
			try {
				auto response = response_future.get();
				double latency = double(response.response_ns - response.submit_ns) / 1000000.0;
				if (telemetry) {
					RecordCacheCorrection(event, response);
					auto &total = runtime.latency_totals[operator_id];
					++total.first;
					total.second += latency;
					// Keep telemetry memory bounded even for a very large Cartesian traversal.
					if (runtime.latencies[operator_id].size() < 4096)
						runtime.latencies[operator_id].push_back(latency);
					event["submit_ns"] = response.submit_ns;
					event["response_ns"] = response.response_ns;
					event["server_start_ns"] = response.server_start_ns;
					event["raw_result"] = response.raw_text;
					if (!response.output_normalization.empty()) {
						event["provider_raw_result"] = response.original_raw_text;
						event["result_normalization"] = response.output_normalization;
					}
					event["latency_ms"] = latency;
					event["server_metadata"] = response.server_metadata_json.empty()
					                               ? Json(nullptr)
					                               : semantic::llm_detail::ParseJSON(response.server_metadata_json);
					event["input_tokens"] = response.input_tokens;
					if (response.result_cache_hit)
						event["prompt_cache_input_tokens"] = response.input_tokens >= 0 ? Json(response.input_tokens) : Json(nullptr);
					event["output_tokens"] = response.output_tokens;
					event["cached_input_tokens"] =
					    response.cached_input_tokens >= 0 ? Json(response.cached_input_tokens) : Json(nullptr);
				}
				auto value = ParseOutput(response.raw_text, binding);
				if (reuse) {
					results_ptr->emplace(user_key, SemanticResultCache::Result {value, sequence});
					runtime.in_flight[results_ptr].erase(user_key);
				}
				const bool emitted =
				    binding.node.kind == semantic::NodeKind::SEMANTIC_PROJECTION || value.GetValue<bool>();
				if (telemetry) {
					if (flight->dependency_publication)
						event["result_publication"] = "shared_dependency";
					event["parse_status"] = "validated";
					event["emitted"] = emitted;
					runtime.Log(event);
				}
				if (runtime.adaptive)
					runtime.adaptive->Observe(operator_id, emitted, latency);
				runtime.Reoptimize(*client_context);
				return value;
			} catch (const SamplingBudgetExhausted &) {
				throw;
			} catch (const SemanticRequestFailure &) {
				throw;
			} catch (const semantic::LLMError &error) {
				FailSemanticRequest(binding, runtime, event, sequence, error, error.confirmed_response.get());
			} catch (const std::exception &error) {
				FailSemanticRequest(binding, runtime, event, sequence, error);
			}
		};
		shared->publish = std::move(finish);
		if (pending) {
			pending->response = response_future;
			pending->complete = [shared]() {
				return shared->Publish();
			};
			throw SemanticRequestBlocked();
		}
		return shared->Publish();
	} catch (const SamplingBudgetExhausted &) {
		throw;
	} catch (const SemanticRequestFailure &) {
		throw;
	} catch (const semantic::LLMError &error) {
		FailSemanticRequest(binding, runtime, event, sequence, error, error.confirmed_response.get());
	} catch (const std::exception &error) {
		FailSemanticRequest(binding, runtime, event, sequence, error);
	}
}

}
}
