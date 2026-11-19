#include "runtime/query_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace duckdb {
namespace semantic_extension {

void QueryState::QueryBegin(ClientContext &) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	runtime.reset();
}

void QueryState::QueryEnd(ClientContext &, optional_ptr<ErrorData> error) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	if (runtime) {
		runtime->scheduler.Shutdown();
		runtime->in_flight.clear();
	}
	if (runtime && runtime->file.is_open()) {
		// Shutdown cancels queued work and joins every admitted transport. The
		// remaining identities were paid, but no operator consumed their result.
		// Audit that work without parsing it or inventing downstream rows.
		for (const auto &entry : runtime->unconsumed_requests) {
			try {
				runtime->Log(
				    {{"event", "semantic_request_discarded"},
				     {"query_id", query_id},
				     {"operator_id", entry.second.operator_id},
				     {"request_seq", entry.first},
				     {"model_id", entry.second.model_id},
				     {"reason", error ? "query_failed_before_consumption" : "query_terminated_before_consumption"}});
			} catch (...) {
				if (!error)
					throw;
			}
			++runtime->discarded_requests[entry.second.operator_id];
		}
		runtime->unconsumed_requests.clear();
		runtime->cached_request_sequences.clear();
		// Include operators whose paid work was entirely abandoned before the
		// first consumed row; their processed cardinalities remain zero.
		for (const auto &entry : runtime->request_counts)
			runtime->counts.emplace(entry.first, std::make_pair(uint64_t(0), uint64_t(0)));
	}
	// LIMIT and optimized-away/empty branches can bypass operator finalization.
	// On successful termination, close their remaining batches without inventing
	// calls. Failed queries retain their partial PENDING/RUNNING observations.
	if (runtime && runtime->adaptive && !error) {
		for (const auto &action : runtime->plan_spec["actions"]) {
			auto id = action["operator_id"].get<int64_t>();
			if (runtime->finished.insert(id).second)
				runtime->adaptive->Finish(id);
		}
	}
	// Emit one terminal report even on failure so partial model work is visible
	// and cannot be mistaken for a successfully completed query.
	if (runtime && runtime->file.is_open()) {
		// Report evaluated semantic inputs, even if a partially filled output
		// chunk was abandoned after an empty join branch or LIMIT. Keep the
		// execution counters unchanged until execution has actually ended.
		for (auto &entry : runtime->counts)
			entry.second = runtime->published_counts[entry.first];
		for (const auto &entry : runtime->published_counts)
			runtime->counts[entry.first] = entry.second;
		if (runtime->sampling) {
			try {
				uint64_t completed = 0;
				for (const auto &entry : runtime->counts)
					completed += entry.second.first;
				runtime->Log({{"event", "sampling_stage_progress"},
				              {"query_id", query_id},
				              {"completed_inputs", completed},
				              {"requests", runtime->llm_submissions},
				              {"budget_exhausted", runtime->budget_exhausted.load()}});
			} catch (...) {
			}
		}
		for (auto &entry : runtime->counts) {
			// Cleanup telemetry must never hide the query's original failure.
			try {
				auto &samples = runtime->latencies[entry.first];
				auto total = runtime->latency_totals[entry.first];
				std::sort(samples.begin(), samples.end());
				auto percentile = [&](double fraction) -> Json {
					if (samples.empty() || total.first > samples.size())
						return nullptr;
					return samples[size_t(std::ceil(fraction * samples.size())) - 1];
				};
				runtime->Log(
				    {{"llm_calls", runtime->request_counts[entry.first]},
				     {"discarded_requests", runtime->discarded_requests[entry.first]},
				     {"result_reuses", runtime->result_reuses[entry.first] + runtime->response_cache_hits[entry.first]},
				     {"selectivity",
				      entry.second.first ? Json(double(entry.second.second) / entry.second.first) : Json(nullptr)},
				     {"latency_avg_ms", total.first ? Json(total.second / total.first) : Json(nullptr)},
				     {"latency_percentiles_complete", total.first == samples.size()},
				     {"latency_p50_ms", percentile(0.50)},
				     {"latency_p95_ms", percentile(0.95)},
				     {"event", "operator_summary"},
				     {"query_id", query_id},
				     {"operator_id", entry.first},
				     {"input_rows", entry.second.first},
				     {"output_rows", entry.second.second},
				     {"processing_seconds", runtime->OperatorProcessingSeconds(entry.first)},
				     {"processing_time_basis", "union_of_operator_active_and_http_intervals"},
				     {"query_failed", bool(error)}});
			} catch (...) {
			}
		}
		if (!runtime->plan_spec.is_null()) {
			try {
				Json operators = Json::array();
				for (auto &action : runtime->plan_spec["actions"]) {
					auto id = action["operator_id"].get<int64_t>();
					auto count = runtime->counts[id];
					operators.push_back(
					    {{"operator_id", id},
					     {"input_rows_or_pairs", count.first},
					     {"output_rows", count.second},
					     {"processing_seconds", runtime->OperatorProcessingSeconds(id)},
					     {"processing_time_basis", "union_of_operator_active_and_http_intervals"},
					     {"llm_calls", runtime->request_counts[id]},
					     {"discarded_requests", runtime->discarded_requests[id]},
					     {"result_reuses", runtime->result_reuses[id] + runtime->response_cache_hits[id]},
					     {"started", runtime->started.count(id) != 0},
					     {"request_latency_total_ms", runtime->latency_totals[id].second},
					     {"selectivity", count.first ? Json(double(count.second) / count.first) : Json(nullptr)}});
				}
				runtime->Log(
				    {{"event", "execution_report"},
				     {"version", 1},
				     {"execution_mode", "data_chunk_pipeline"},
				     {"semantic_join_mode",
				      runtime->query_ir.value("semantic_join_mode", "stream_first_left_then_replay")},
				     {"semantic_response_order",
				      runtime->sampling || (runtime->adaptive && !runtime->adaptive->ChunkMode()) ? "input_order"
				                                                                                  : "ready_first"},
				     {"query_id", query_id},
				     {"status", error ? "error" : "success"},
				     {"plan_spec", runtime->plan_spec},
				     {"actual_start_order", runtime->start_order},
				     {"operators", operators},
				     {"runtime_state", runtime->adaptive ? runtime->adaptive->Report() : Json(nullptr)},
				     {"runtime_observations", runtime->adaptive ? runtime->adaptive->Observations() : Json(nullptr)},
				     {"execution_wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
				                                                                     runtime->execution_start)
				                               .count()},
				     {"accuracy", nullptr},
				     {"cache_metrics", nullptr}});
			} catch (...) {
				if (!error)
					throw;
			}
		}
		std::lock_guard<std::mutex> log_guard(runtime->log_mutex);
		runtime->file.flush();
	}
	runtime.reset();
}

QueryRuntime &QueryState::Ensure(ClientContext &context, const SemanticBindData &binding) {
	std::lock_guard<std::recursive_mutex> guard(mutex);
	// The first semantic operator creates the shared runtime; later operators must
	// agree on the exact query and optimizer decision.
	ValidateExecutionSettings(context);
	if (!runtime) {
		models = binding.models;
		query_id = binding.execution_context.query_id;
		runtime = make_uniq<QueryRuntime>(
		    binding, GetSetting(context, "semantic_telemetry"), GetSetting(context, "semantic_runtime"),
		    std::stoull(GetSetting(context, "semantic_parallelism")), transport_pool,
		    GetSetting(context, "semantic_unrestricted") == "true",
		    std::stoull(GetSetting(context, "semantic_max_concurrent_requests_per_model")),
		    GetSetting(context, "semantic_enforce_model_operator_order") == "true");
		transport_pool = runtime->client.Connections();
		runtime->client.ConfigureResultCache(GetSetting(context, "semantic_response_cache"));
		// Reuse typed results and pending requests before scheduler admission.
		// Both caches are bounded; verification keeps its independent calls.
		runtime->reuse_results = GetSetting(context, "semantic_request_reuse") == "true" &&
		                         !runtime->client.ResultCacheVerificationEnabled();
	} else if (models != binding.models || query_id != binding.execution_context.query_id) {
		throw InvalidInputException("Inconsistent semantic configuration within query");
	}
	return *runtime;
}

QueryRuntime &QueryState::Get(ClientContext &context, const SemanticBindData &binding) {
	Ensure(context, binding);
	std::lock_guard<std::recursive_mutex> guard(runtime->mutex);
	if (!runtime->plan_spec.is_null()) {
		if (!runtime->operator_ids.count(binding.node.semantic.operator_id) ||
		    runtime->finished.count(binding.node.semantic.operator_id))
			throw InvalidInputException("Semantic pipeline used an unknown or finished PlanSpec operator");
	}
	// Start means the first Execute invocation, not physical-state allocation.
	// Operators whose input is empty may never Execute and have no start event.
	if (runtime->started.insert(binding.node.semantic.operator_id).second) {
		runtime->start_order.push_back(binding.node.semantic.operator_id);
		if (runtime->file.is_open())
			runtime->Log({{"event", "operator_start"},
			              {"query_id", query_id},
			              {"operator_id", binding.node.semantic.operator_id},
			              {"operator_kind", semantic::OperatorName(binding.node.kind)},
			              {"model_id", runtime->adaptive ? runtime->adaptive->Current(
			                                                   binding.node.semantic.operator_id)["current"]["model_id"]
			                                             : Json(binding.node.semantic.model_id)},
			              {"next_request_seq", runtime->sequence + 1}});
	}
	return *runtime;
}

void QueryState::Finish(ClientContext &context, const SemanticBindData &binding) {
	auto &query = Ensure(context, binding);
	std::lock_guard<std::recursive_mutex> guard(query.mutex);
	if (query.finished.insert(binding.node.semantic.operator_id).second && query.adaptive)
		query.adaptive->Finish(binding.node.semantic.operator_id);
}

namespace {
Json InputJSON(const Value &value) {
	if (value.IsNull())
		return nullptr;
	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return value.GetValue<bool>();
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
		return value.GetValue<int64_t>();
	case LogicalTypeId::DOUBLE: {
		auto number = value.GetValue<double>();
		if (!std::isfinite(number))
			throw InvalidInputException("Non-finite semantic input number");
		return number;
	}
	case LogicalTypeId::VARCHAR:
		return value.GetValue<string>();
	case LogicalTypeId::LIST: {
		Json result = Json::array();
		for (const auto &child : ListValue::GetChildren(value))
			result.push_back(InputJSON(child));
		return result;
	}
	case LogicalTypeId::STRUCT: {
		Json result = Json::object();
		const auto &fields = StructType::GetChildTypes(value.type());
		const auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < fields.size(); ++i)
			result[fields[i].first] = InputJSON(children[i]);
		return result;
	}
	default:
		throw InvalidInputException("Unsupported semantic input type");
	}
}
}

string SampleRowJSON(const Value &value) {
	std::function<Json(const Value &)> convert = [&](const Value &item) -> Json {
		if (item.IsNull())
			return nullptr;
		if (item.type().id() == LogicalTypeId::STRUCT) {
			Json result = Json::object();
			auto &fields = StructType::GetChildTypes(item.type());
			auto &children = StructValue::GetChildren(item);
			for (idx_t i = 0; i < fields.size(); ++i)
				result[fields[i].first] = convert(children[i]);
			return result;
		}
		if (item.type().id() == LogicalTypeId::LIST) {
			Json result = Json::array();
			for (auto &child : ListValue::GetChildren(item))
				result.push_back(convert(child));
			return result;
		}
		if (item.type().id() == LogicalTypeId::FLOAT)
			return item.GetValue<double>();
		if (item.type().id() == LogicalTypeId::BOOLEAN || item.type().id() == LogicalTypeId::TINYINT ||
		    item.type().id() == LogicalTypeId::SMALLINT || item.type().id() == LogicalTypeId::INTEGER ||
		    item.type().id() == LogicalTypeId::BIGINT || item.type().id() == LogicalTypeId::DOUBLE ||
		    item.type().id() == LogicalTypeId::VARCHAR)
			return InputJSON(item);
		// SQL-only types (dates, decimals, huge integers) round-trip through their
		// typed temporary columns. This does not widen semantic input contracts.
		return item.ToString();
	};
	return convert(value).dump();
}

semantic::Row ReadRow(const Value &value) {
	semantic::Row result;
	auto &fields = StructType::GetChildTypes(value.type());
	const auto *children = value.IsNull() ? nullptr : &StructValue::GetChildren(value);
	for (idx_t i = 0; i < fields.size(); i++) {
		semantic::RowValue converted;
		if (children && !(*children)[i].IsNull()) {
			const auto &item = (*children)[i];
			switch (item.type().id()) {
			case LogicalTypeId::BOOLEAN:
				converted = semantic::RowValue(item.GetValue<bool>());
				break;
			case LogicalTypeId::INTEGER:
			case LogicalTypeId::BIGINT:
				converted = semantic::RowValue(item.GetValue<int64_t>());
				break;
			case LogicalTypeId::DOUBLE:
				converted = semantic::RowValue(item.GetValue<double>());
				break;
			case LogicalTypeId::VARCHAR:
				converted = semantic::RowValue(item.GetValue<string>());
				break;
			case LogicalTypeId::LIST:
			case LogicalTypeId::STRUCT:
				converted = semantic::RowValue::FromJSON(InputJSON(item).dump());
				break;
			default:
				throw InvalidInputException("Unsupported semantic input type");
			}
		}
		result.emplace(fields[i].first, std::move(converted));
	}
	return result;
}

namespace {

Value ParseField(const Json &value, const semantic::Field &field) {
	auto kind = field.type;
	auto type = ToLogicalType(field);
	if (value.is_null())
		return Value(type);
	if (kind == semantic::FieldType::VARCHAR && value.is_string())
		return Value(value.get<string>());
	if (kind == semantic::FieldType::BOOLEAN && value.is_boolean())
		return Value::BOOLEAN(value.get<bool>());
	if ((kind == semantic::FieldType::INTEGER || kind == semantic::FieldType::BIGINT) && value.is_number_integer()) {
		if (value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))
			throw InvalidInputException("Semantic integer overflow");
		auto number = value.get<int64_t>();
		if (kind == semantic::FieldType::BIGINT)
			return Value::BIGINT(number);
		if (number >= std::numeric_limits<int32_t>::min() && number <= std::numeric_limits<int32_t>::max())
			return Value::INTEGER(int32_t(number));
	}
	if (kind == semantic::FieldType::DOUBLE && value.is_number()) {
		auto number = value.get<double>();
		if (std::isfinite(number))
			return Value::DOUBLE(number);
	}
	auto rendered = value.dump();
	if (rendered.size() > 512)
		rendered.resize(512);
	throw InvalidInputException("Semantic output does not match declared %s (JSON type=%s, value=%s)", type.ToString(),
	                            value.type_name(), rendered);
}

}

Value ParseOutput(const string &raw, const SemanticBindData &binding) {
	auto json = semantic::llm_detail::ParseJSON(raw);
	if (binding.node.kind != semantic::NodeKind::SEMANTIC_PROJECTION) {
		if (!json.is_boolean())
			throw InvalidInputException("Semantic predicate requires JSON true or false");
		return Value::BOOLEAN(json.get<bool>());
	}
	auto &schema = binding.node.semantic.output_schema.fields;
	if (!json.is_object())
		throw InvalidInputException("Semantic projection requires a JSON object");
	if (schema.size() != 1)
		throw InvalidInputException("Semantic projection requires exactly one output field");
	const auto &field = schema[0];
	if (!json.contains(field.name))
		throw InvalidInputException("Missing semantic output key: %s", field.name);
	return Value::STRUCT({{field.name, ParseField(json.at(field.name), field)}});
}

}
}
