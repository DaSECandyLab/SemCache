#include "llm/http/http_client_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <thread>

namespace semantic {

using namespace http_detail;

LLMResponse HttpLLMClient::InferHTTP(const LLMRequest &request, const std::function<void()> &admitted) {
	std::string endpoint_name = "unresolved";
	std::string api_key;
	int status = 0;
	bool admission_reported = false;
	try {
		ValidateRequest(request);
		// Resolve the model before creating a connection, so invalid optimizer
		// decisions cannot cause network traffic.
		const auto &endpoint = registry.Resolve(request.model_id);
		endpoint_name = endpoint.base_url + "/v1/chat/completions";
		// Resolve only at the send boundary: configuration/audit retain the variable
		// name, never the credential. Invalid values must fail before admission.
		if (!endpoint.api_key_env.empty()) {
			const auto value = std::getenv(endpoint.api_key_env.c_str());
			if (!value || !*value)
				throw ValidationError("required API credential is missing");
			api_key = value;
			for (unsigned char c : api_key) {
				if (c <= 32 || c >= 127)
					throw ValidationError("API credential contains invalid header characters");
			}
		}
		// OpenAI-compatible metadata is map<string,string>. Convert only at
		// the shared HTTP boundary; scheduler and audit identities stay typed.
		const Json wire_metadata = {{"query_id", request.query_id},
		                            {"operator_id", std::to_string(request.operator_id)},
		                            {"request_seq", std::to_string(request.request_seq)},
		                            {"model_id", request.model_id}};
		auto body = RequestBody(request, endpoint);
		const int base_max_tokens = body.at("max_tokens").get<int>();
		body["metadata"] = wire_metadata;

		auto connection_options = options;
		if (endpoint.connect_timeout_ms)
			connection_options.connect_timeout_ms = endpoint.connect_timeout_ms;
		const auto connection_key = std::make_tuple(endpoint.scheduling_key, connection_options.connect_timeout_ms,
		                                            options.read_timeout_ms, options.write_timeout_ms);
		// Retry incomplete completions within a bounded budget; each retry is a separately admitted and billed request.
		const int incomplete_retries = [] {
			const char *value = std::getenv("SEMCACHE_INCOMPLETE_RETRIES");
			const int parsed = value ? std::atoi(value) : 2;
			return parsed < 0 ? 0 : (parsed > 10 ? 10 : parsed);
		}();
		// Reconnect within the transport retry budget when no response has arrived; zero disables reconnects.
		const int transport_retries = [] {
			const char *value = std::getenv("SEMCACHE_TRANSPORT_RETRIES");
			const int parsed = value ? std::atoi(value) : 1;
			return parsed < 0 ? 0 : (parsed > 3 ? 3 : parsed);
		}();
		const int rate_limit_retries = [] {
			const char *value = std::getenv("SEMCACHE_RATE_LIMIT_RETRIES");
			const int parsed = value ? std::atoi(value) : 0;
			return std::max(0, std::min(20, parsed));
		}();
		int rate_limit_attempts = 0;
		const int server_error_retries = [] {
			const char *value = std::getenv("SEMCACHE_SERVER_ERROR_RETRIES");
			const int parsed = value ? std::atoi(value) : 0;
			return std::max(0, std::min(10, parsed));
		}();
		int server_error_attempts = 0;
		bool repair_completion = false;

		// A retransmission that cannot be admitted must not turn a transport failure
		// into a budget failure: the caller still needs to know what went wrong.
		std::string transport_error;
		for (int transport_attempt = 0, completion_attempt = 0;;) {
			if (repair_completion && request.output_schema.kind == OutputSchema::Kind::OBJECT) {
				// Only invalid/truncated completions get a recovery instruction.
				// Keep the original task/data/layout and system prefix intact so
				// retries preserve KV prefix cache hits on shared prefixes.
				const auto &field = request.output_schema.fields.at(0);
				std::string recovery =
				    "\n\nOutput recovery: Start a fresh answer from the original input. "
				    "Preserve the task's conditions and missing-value rules. ";
				if (field.type == FieldType::VARCHAR) {
					recovery += "Extract the requested information without repeating phrases or cycling through a list. "
					            "Preserve distinct required facts and return only the requested field value as plain text. ";
					body["frequency_penalty"] = 0.2;
				} else if (field.type == FieldType::BOOLEAN) {
					recovery += "Return only one boolean value: true or false, or null when the task requires it. ";
				} else {
					recovery += "Return only one " + TypeName(field) +
					            " numeric value, or null when the task requires it. "
					            "Write the full numeric amount without currency symbols, unit suffixes, "
					            "abbreviations, digit-grouping commas or quotation marks. ";
				}
				body["messages"][1]["content"] =
				    request.user_text + recovery + "Do not add a JSON wrapper, examples or commentary.";
			}
			// Length-terminated responses may retry with a larger output budget, bounded by the configured cap.
			const int attempt_max_tokens =
			    std::min<int64_t>(std::max<int64_t>(4096, base_max_tokens),
			                      int64_t(base_max_tokens) * (int64_t(1) << std::min(completion_attempt, 3)));
			body["max_tokens"] = attempt_max_tokens;
			SharedRequestAdmission shared_admission(
			    endpoint.external ? SharedRequestAdmission::Scope::External : SharedRequestAdmission::Scope::Local);
			std::shared_ptr<ConnectionPool::Entry> connection;
			std::unique_lock<std::mutex> connection_guard;
			{
				std::lock_guard<std::mutex> guard(connections->mutex);
				auto &entries = connections->entries[connection_key];
				for (std::size_t index = 0; index < entries.size();) {
					auto candidate = entries[index];
					std::unique_lock<std::mutex> lease(candidate->mutex, std::try_to_lock);
					if (!lease.owns_lock()) {
						++index;
						continue;
					}
					if (candidate->poisoned.load()) {
						entries.erase(entries.begin() + index);
						continue;
					}
					connection = candidate;
					connection_guard = std::move(lease);
					break;
				}
				if (!connection) {
					connection =
					    std::make_shared<ConnectionPool::Entry>(std::string(endpoint.tls ? "https://" : "http://") +
					                                                endpoint.host + ":" + std::to_string(endpoint.port),
					                                            connection_options);
					connection_guard = std::unique_lock<std::mutex>(connection->mutex);
					entries.push_back(connection);
				}
			}

			httplib::Request wire;
			wire.method = "POST";
			wire.path = endpoint.path_prefix + "/v1/chat/completions";
			wire.set_header("Content-Type", "application/json");
			if (!api_key.empty())
				wire.set_header("Authorization", "Bearer " + api_key);
			for (const auto &header : endpoint.headers)
				wire.set_header(header.first.c_str(), header.second.c_str());
			wire.body = body.dump();
			std::string response_body;
			bool response_started = false;
			wire.response_start_handler = [&]() {
				response_started = true;
			};
			bool too_large = false;
			wire.content_receiver = [&](const char *data, size_t length, uint64_t, uint64_t total) {
				if (length > options.max_response_bytes - response_body.size() || total > options.max_response_bytes) {
					too_large = true;
					return false;
				}
				response_body.append(data, length);
				return true;
			};

			LLMResponse response;
			response.submit_ns = Now();
			// Each concurrent send leases its own reusable connection. Admission is
			// controlled by the shared admission queue and per-model scheduler.
			if (before_send) {
				try {
					before_send(request, request.cache_correction_retry || transport_attempt > 0 ||
					                         completion_attempt > 0 || rate_limit_attempts > 0 ||
					                         server_error_attempts > 0);
				} catch (const SamplingBudgetExhausted &) {
					if ((transport_attempt > 0 || completion_attempt > 0) && !transport_error.empty())
						throw ValidationError(transport_error);
					throw;
				}
			}
			if (!admission_reported) {
				if (admitted)
					admitted();
				admission_reported = true;
			}
			const auto result = connection->client.send(wire);
			response.response_ns = Now();
			shared_admission.Release();
			if (!result) {
				// The socket cannot be trusted after a failed exchange: shut it down
				// and keep it out of the pool so the retry dials a fresh connection.
				connection->client.stop();
				connection->poisoned.store(true);
			}
			connection_guard.unlock();

			if (too_large)
				throw ValidationError("response exceeds configured byte limit");
			if (!result) {
				const auto error = result.error();
				const bool retryable = error == httplib::Error::Connection ||
				                       error == httplib::Error::ConnectionTimeout || error == httplib::Error::Read ||
				                       error == httplib::Error::Write || error == httplib::Error::SSLConnection ||
				                       error == httplib::Error::ProxyConnection;
				// Only retry when the response never started: a partially delivered
				// answer may already have been generated and billed.
				transport_error = "HTTP transport failure/timeout: " + httplib::to_string(error);
				if (retryable && !response_started && response_body.empty() && transport_attempt < transport_retries) {
					std::this_thread::sleep_for(std::chrono::milliseconds(50 * (transport_attempt + 1)));
					++transport_attempt;
					continue;
				}
				throw ValidationError(transport_error);
			}

			status = result->status;
			const bool temporary_server_error = status == 502 || status == 503 || status == 504;
			auto &http_attempts = status == 429 ? rate_limit_attempts : server_error_attempts;
			const int http_retries = status == 429 ? rate_limit_retries : server_error_retries;
			if ((status == 429 || temporary_server_error) && http_attempts < http_retries) {
				const auto header = result->get_header_value("Retry-After");
				char *end = nullptr;
				const long seconds = std::strtol(header.c_str(), &end, 10);
				const bool valid = !header.empty() && end && *end == '\0' && seconds >= 0 && seconds <= 3600;
				const long wait_seconds = valid ? seconds : std::min(60, 1 << std::min(http_attempts, 6));
				++http_attempts;
				const auto jitter = wait_seconds ? (request.request_seq * 17 + http_attempts * 97) % 251 : 0;
				const auto wait_ms = wait_seconds * 1000 + jitter;
				std::cerr << "HTTP " + std::to_string(status) + " retry=" + std::to_string(http_attempts) +
				                 " wait_ms=" + std::to_string(wait_ms) + " query=" + request.query_id +
				                 " request=" + std::to_string(request.request_seq) + "\n";
				std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
				continue;
			}
			if (status == 429 && rate_limit_retries > 0)
				throw ValidationError("HTTP rate limit persisted after " + std::to_string(rate_limit_attempts) +
				                      " retries");
			if (temporary_server_error && server_error_retries > 0)
				throw ValidationError("HTTP upstream error persisted after " + std::to_string(server_error_attempts) +
				                      " retries");
			if (status != 200) {
				auto detail = response_body;
				if (!api_key.empty()) {
					for (size_t at = 0; (at = detail.find(api_key, at)) != std::string::npos; at += 10)
						detail.replace(at, api_key.size(), "[REDACTED]");
				}
				throw ValidationError("HTTP error (redirects and retries disabled): " + detail.substr(0, 2048));
			}

			const auto json = llm_detail::ParseJSON(response_body);
			if (!json.is_object() || !json.contains("choices") || !json.at("choices").is_array() ||
			    json.at("choices").size() != 1)
				throw ValidationError("malformed response choices");
			const auto &choice = json.at("choices")[0];
			const char *missing = !choice.is_object()                 ? "choice"
			                      : !choice.contains("index")         ? "index"
			                      : !choice.contains("finish_reason") ? "finish_reason"
			                      : !choice.contains("message")       ? "message"
			                                                          : nullptr;
			if (missing || !choice.is_object() || !choice.at("index").is_number_integer() || choice.at("index") != 0)
				throw ValidationError(std::string("malformed or incomplete completion (choice field: ") +
				                      (missing ? missing : "index") + ")");
			const auto &finish_reason = choice.at("finish_reason");
			std::string finish = finish_reason.is_string() ? finish_reason.get<std::string>()
			                     : finish_reason.is_null() ? "null"
			                                               : finish_reason.dump();
			const auto &message = choice.at("message");
			const char *message_field = !message.is_object()           ? "message"
			                            : !message.contains("role")    ? "role"
			                            : !message.contains("content") ? "content"
			                                                           : nullptr;
			// Some providers exhaust the completion budget before producing any
			// final content. A null/missing body with an explicit length finish
			// must reach the bounded incomplete-response retry below. Other
			// malformed messages, including null content with stop, stay errors.
			const bool empty_truncated = finish == "length" && message.is_object() && message.contains("role") &&
			                             message.at("role") == "assistant" &&
			                             (!message.contains("content") || message.at("content").is_null());
			if (!empty_truncated &&
			    (message_field || message.at("role") != "assistant" || !message.at("content").is_string()))
				throw ValidationError(std::string("malformed response message (field: ") +
				                      (message_field ? message_field : "content") + ", finish_reason: " + finish + ")");

			response.raw_text = empty_truncated ? "" : message.at("content").get<std::string>();
			if (json.contains("usage") && !json.at("usage").is_null()) {
				if (!json.at("usage").is_object())
					throw ValidationError("malformed usage");
				response.input_tokens = TokenCount(json.at("usage"), "prompt_tokens");
				response.output_tokens = TokenCount(json.at("usage"), "completion_tokens");
				const auto &usage = json.at("usage");
				if (usage.contains("prompt_tokens_details") && !usage.at("prompt_tokens_details").is_null()) {
					const auto &details = usage.at("prompt_tokens_details");
					if (!details.is_object())
						throw ValidationError("malformed prompt token details");
					response.cached_input_tokens = TokenCount(details, "cached_tokens");
					if (response.cached_input_tokens >= 0 &&
					    (response.input_tokens < 0 || response.cached_input_tokens > response.input_tokens))
						throw ValidationError("cached tokens exceed input tokens");
				} else if (usage.contains("prompt_tokens_details") && usage.at("prompt_tokens_details").is_null() &&
				           response.input_tokens >= 0) {
					response.cached_input_tokens = 0;
				}
			}

			if (json.contains("semantic_metadata")) {
				const auto &echo = json.at("semantic_metadata");
				if (!echo.is_object())
					throw ValidationError("malformed server metadata");
				for (auto it = wire_metadata.begin(); it != wire_metadata.end(); ++it) {
					if (!echo.contains(it.key()) || !echo.at(it.key()).is_string() || echo.at(it.key()) != it.value())
						throw ValidationError("server identity echo mismatch");
				}
				response.server_metadata_json = echo.dump();
			}
			// Treat empty or whitespace-only content as incomplete and apply the bounded completion retry policy.
			const bool empty_text_projection = request.output_schema.kind == OutputSchema::Kind::OBJECT &&
			                                   request.output_schema.fields.at(0).type == FieldType::VARCHAR &&
			                                   finish == "stop";
			if (response.raw_text.find_first_not_of(" \t\r\n") == std::string::npos && !empty_text_projection) {
				if (completion_attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (completion_attempt + 1)));
					++completion_attempt;
					continue;
				}
				throw LLMError(request, endpoint_name, status,
				               "malformed or incomplete completion (empty content, finish_reason: " + finish +
				                   ", completion_tokens: " + std::to_string(response.output_tokens) +
				                   ", prompt_tokens: " + std::to_string(response.input_tokens) +
				                   ", max_tokens: " + std::to_string(attempt_max_tokens) + ")",
				               &response);
			}
			// Models return plain values. Convert them once to the typed internal
			// representation shared by SQL execution, sampling and answer caching.
			try {
				std::string normalized;
				if (request.output_schema.kind == OutputSchema::Kind::BOOLEAN) {
					if (endpoint.structured_output) {
						const auto first = response.raw_text.find_first_not_of(" \t\r\n");
						const auto last = response.raw_text.find_last_not_of(" \t\r\n");
						if (first == std::string::npos)
							throw ValidationError("empty constrained predicate response");
						const auto text = response.raw_text.substr(first, last - first + 1);
						if (text != "true" && text != "false")
							throw ValidationError("constrained predicate response must contain only true or false");
					}
					const char *normalization = "single_boolean_token_icase";
					static const std::regex exact_boolean(R"(^\s*(true|false)\s*$)", std::regex::icase);
					static const std::regex negated_boolean(
					    R"(\b(?:not|no|never|isn't|isnt|aren't|arent)\s+(?:true|false)\b)", std::regex::icase);
					static const std::regex boolean_token(R"(\b(true|false)\b)", std::regex::icase);
					std::smatch exact;
					if (std::regex_match(response.raw_text, exact, exact_boolean)) {
						const char initial = exact[1].str()[0];
						normalized = initial == 't' || initial == 'T' ? "true" : "false";
					} else {
						if (std::regex_search(response.raw_text, negated_boolean))
							throw ValidationError("ambiguous negated boolean response");
						// The last standalone true/false token determines the boolean value when non-negated prose appears.
						normalization = "last_boolean_token_icase";
						for (auto match =
						         std::sregex_iterator(response.raw_text.begin(), response.raw_text.end(), boolean_token);
						     match != std::sregex_iterator(); ++match) {
							const char initial = (*match)[1].str()[0];
							normalized = initial == 't' || initial == 'T' ? "true" : "false";
						}
					}
					// Some predicate tasks explicitly request YES/NO. Adapt a whole
					// single-word answer only; prose containing yes/no is ambiguous.
					// Existing true/false tokens always retain their precedence.
					if (normalized.empty()) {
						static const std::regex yes_no_answer(R"(^\s*(yes|no)\s*$)", std::regex::icase);
						std::smatch answer;
						if (std::regex_match(response.raw_text, answer, yes_no_answer)) {
							const char initial = answer[1].str()[0];
							normalized = initial == 'y' || initial == 'Y' ? "true" : "false";
							normalization = "single_yes_no_icase";
						}
					}
					if (normalized.empty())
						throw ValidationError(
						    "response contains no standalone true or false token or single YES/NO answer");
					const auto first = response.raw_text.find_first_not_of(" \t\r\n");
					const auto last = response.raw_text.find_last_not_of(" \t\r\n");
					if (response.raw_text.compare(first, last - first + 1, normalized) != 0) {
						response.original_raw_text = response.raw_text;
						response.output_normalization = normalization;
					}
				} else {
					normalized = NormalizeProjectionText(response.raw_text, request.output_schema);
					response.original_raw_text = response.raw_text;
					response.output_normalization = "plain_projection_value";
				}
				response.raw_text = normalized;
				ValidateStructuredPayload(request, response.raw_text);
			} catch (const std::exception &error) {
				if (completion_attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (completion_attempt + 1)));
					++completion_attempt;
					continue;
				}
				throw LLMError(request, endpoint_name, status,
				               std::string("malformed structured completion (finish_reason: ") + finish +
				                   ", completion_tokens: " + std::to_string(response.output_tokens) +
				                   ", prompt_tokens: " + std::to_string(response.input_tokens) +
				                   ", max_tokens: " + std::to_string(attempt_max_tokens) + "): " + error.what(),
				               &response);
			}
			if (choice.at("finish_reason") != "stop") {
				if (completion_attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (completion_attempt + 1)));
					++completion_attempt;
					continue;
				}
				throw LLMError(request, endpoint_name, status,
				               "malformed or incomplete completion (finish_reason: " + finish +
				                   ", completion_tokens: " + std::to_string(response.output_tokens) +
				                   ", prompt_tokens: " + std::to_string(response.input_tokens) + ")",
				               &response);
			}
			return response;
		}
	} catch (const SamplingBudgetExhausted &) {
		throw;
	} catch (const LLMError &) {
		throw;
	} catch (const std::exception &error) {
		// Parser/HTTP libraries may echo response bodies.  Never let a provider
		// echo of the bearer token cross the error or audit boundary.
		std::string reason = error.what();
		if (!api_key.empty()) {
			size_t pos = 0;
			while ((pos = reason.find(api_key, pos)) != std::string::npos) {
				reason.replace(pos, api_key.size(), "[redacted]");
				pos += 10;
			}
		}
		throw LLMError(request, endpoint_name, status, reason);
	}
}

}
