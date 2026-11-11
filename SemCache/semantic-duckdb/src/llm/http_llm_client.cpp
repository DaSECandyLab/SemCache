// Keep HTTP concerns at this boundary: build one chat request, enforce resource
// limits, then return a transport-neutral response to the execution layer.
#include "llm/http_llm_client.hpp"
#include "llm/request_scheduler.hpp"
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <random>
#include <openssl/evp.h>
#include <thread>
#include <tuple>
// Retry transport failures only before any response byte arrives; a partial response may already be billed.
#define CPPHTTPLIB_NO_SSL_PEER_RETRY
#include "cpp-httplib/httplib.h"
#include "llm/json_utils.hpp"

namespace semantic {
namespace {

using Json = llm_detail::Json;

int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

int64_t TokenCount(const Json &usage, const char *name) {
	if (!usage.contains(name))
		return -1;
	const auto &count = usage.at(name);
	if (!count.is_number_integer() ||
	    (count.is_number_unsigned() && count.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
	    (!count.is_number_unsigned() && count.get<int64_t>() < 0))
		throw ValidationError("invalid usage token count");
	return count.get<int64_t>();
}

// Mirror the operator's declared value contract at the retry boundary. In
// particular, a nullable field is legal; a null Projection object is not.
void ValidateOutputField(const Json &value, const Field &field) {
	if (value.is_null())
		return;
	bool valid = false;
	switch (field.type) {
	case FieldType::BOOLEAN:
		valid = value.is_boolean();
		break;
	case FieldType::INTEGER:
	case FieldType::BIGINT:
		if (value.is_number_integer() &&
		    !(value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))) {
			const auto number = value.get<int64_t>();
			valid = field.type == FieldType::BIGINT ||
			        (number >= std::numeric_limits<int32_t>::min() && number <= std::numeric_limits<int32_t>::max());
		}
		break;
	case FieldType::DOUBLE:
		valid = value.is_number() && std::isfinite(value.get<double>());
		break;
	case FieldType::VARCHAR:
		valid = value.is_string();
		break;
	case FieldType::LIST:
	case FieldType::STRUCT:
		throw ValidationError("semantic projection requires a scalar output field");
	}
	if (!valid)
		throw ValidationError("structured field " + field.name + " must match " + TypeName(field) +
		                      " (JSON type=" + value.type_name() + ")");
}

void ValidateStructuredResult(const Json &value, const OutputSchema &schema) {
	ValidateOutputSchema(schema);
	if (schema.kind == OutputSchema::Kind::BOOLEAN) {
		if (!value.is_boolean())
			throw ValidationError("predicate response must be true or false");
		return;
	}
	const auto &field = schema.fields[0];
	if (!value.is_object() || !value.contains(field.name))
		throw ValidationError("response missing declared field: " + field.name);
	ValidateOutputField(value.at(field.name), field);
}

void ValidateStructuredPayload(const LLMRequest &request, const std::string &raw) {
	const auto parsed = llm_detail::ParseJSON(raw);
	ValidateStructuredResult(parsed, request.output_schema);
}

std::string CacheContract(const LLMRequest &request) {
	if (request.output_schema.kind == OutputSchema::Kind::BOOLEAN)
		return Json::array({"plain-predicate-v1"}).dump();
	const auto &field = request.output_schema.fields.at(0);
	return Json::array({"plain-projection-v1", field.name, TypeName(field)}).dump();
}

std::string NormalizeProjectionText(const std::string &raw, const OutputSchema &schema) {
	const auto first = raw.find_first_not_of(" \t\r\n");
	const auto last = raw.find_last_not_of(" \t\r\n");
	const auto text = first == std::string::npos ? std::string() : raw.substr(first, last - first + 1);
	const auto &field = schema.fields.at(0);
	Json value;
	if (text == "NULL" || text == "null") {
		value = nullptr;
	} else if (field.type == FieldType::VARCHAR) {
		value = text;
	} else {
		// Parse the entire plain value. Never extract a number from an explanation.
		value = llm_detail::ParseJSON(text);
	}
	ValidateOutputField(value, field);
	// This typed envelope is internal; the provider returns one plain value.
	return Json {{field.name, value}}.dump();
}

void AppendKeyPart(std::string &key, const std::string &part) {
	const uint64_t size = part.size();
	for (size_t i = 0; i < 8; ++i)
		key += char((size >> (i * 8)) & 255);
	key += part;
}
std::string ResultDigest(const std::string &value) {
	std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
	unsigned char digest[EVP_MAX_MD_SIZE];
	unsigned int size = 0;
	if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
	    EVP_DigestUpdate(context.get(), value.data(), value.size()) != 1 ||
	    EVP_DigestFinal_ex(context.get(), digest, &size) != 1)
		throw ValidationError("cache result digest failed");
	const char hex[] = "0123456789abcdef";
	std::string result;
	result.reserve(size * 2);
	for (unsigned int i = 0; i < size; ++i) {
		result += hex[digest[i] >> 4];
		result += hex[digest[i] & 15];
	}
	return result;
}

Json RequestBody(const LLMRequest &request, const ModelEndpoint &endpoint) {
	const int base_max_tokens = endpoint.max_tokens ? endpoint.max_tokens : request.max_tokens;
	Json body = {{"model", endpoint.model},
	             {"temperature", request.temperature},
	             {"max_tokens", base_max_tokens},
	             {"stream", false},
	             {"messages", Json::array({{{"role", "system"}, {"content", request.system_text}},
	                                       {{"role", "user"}, {"content", request.user_text}}})}};
	// Local structured endpoints support regex-constrained plain text. Keep
	// predicate answers boolean-only without introducing a JSON wrapper.
	if (endpoint.structured_output && request.output_schema.kind == OutputSchema::Kind::BOOLEAN)
		body["regex"] = "(true|false)";
	if (endpoint.has_enable_thinking) {
		body["chat_template_kwargs"] = {{"enable_thinking", endpoint.enable_thinking}};
	}
	if (!endpoint.thinking.empty())
		body["thinking"] = {{"type", endpoint.thinking}};
	if (!endpoint.reasoning_effort.empty())
		body["reasoning_effort"] = endpoint.reasoning_effort;

	return body;
}

}

struct HttpLLMClient::ConnectionPool {
	struct Entry {
		explicit Entry(std::string url, const HTTPOptions &options) : client(std::move(url)) {
			client.enable_server_certificate_verification(true);
			client.enable_server_hostname_verification(true);
			client.set_connection_timeout(std::chrono::milliseconds(options.connect_timeout_ms));
			client.set_read_timeout(std::chrono::milliseconds(options.read_timeout_ms));
			client.set_write_timeout(std::chrono::milliseconds(options.write_timeout_ms));
			client.set_follow_location(false);
			client.set_keep_alive(true);
		}
		httplib::Client client;
		std::mutex mutex;
		// A socket that failed one exchange is never handed out again, even when
		// httplib still considers it alive.
		std::atomic<bool> poisoned {false};
	};
	std::mutex mutex;
	std::map<std::tuple<std::string, int, int, int>, std::vector<std::shared_ptr<Entry>>> entries;
};

// Coordinators share the storage instance's lifetime but retain only active
// requests. Completed values live exclusively in the bounded result cache.
struct HttpLLMClient::CacheFlights {
	std::mutex mutex;
	std::unordered_map<std::string, std::shared_future<LLMResponse>> requests;
	static std::shared_ptr<CacheFlights> Open(const std::shared_ptr<PromptResultCache> &cache) {
		static std::mutex registry_mutex;
		static std::map<PromptResultCache *, std::weak_ptr<CacheFlights>> instances;
		std::lock_guard<std::mutex> guard(registry_mutex);
		for (auto entry = instances.begin(); entry != instances.end();) {
			if (entry->second.expired())
				entry = instances.erase(entry);
			else
				++entry;
		}
		auto &slot = instances[cache.get()];
		auto shared = slot.lock();
		if (!shared) {
			shared = std::make_shared<CacheFlights>();
			slot = shared;
		}
		return shared;
	}
};

HttpLLMClient::HttpLLMClient(ModelRegistry registry, HTTPOptions options,
                             std::function<void(const LLMRequest &, bool retransmit)> before_send,
                             std::shared_ptr<ConnectionPool> pool, std::function<void(const LLMRequest &)> on_cache_hit)
    : registry(std::move(registry)), options(options), before_send(std::move(before_send)),
      on_cache_hit(std::move(on_cache_hit)) {
	if (options.connect_timeout_ms <= 0 || options.read_timeout_ms <= 0 || options.write_timeout_ms <= 0 ||
	    options.max_response_bytes == 0 || options.max_response_bytes > 16 * 1024 * 1024)
		throw ValidationError("invalid HTTP timeout/response limit");
	connections = pool ? std::move(pool) : std::make_shared<ConnectionPool>();
	const auto configuration = std::getenv("SEMCACHE_RESPONSE_CACHE");
	if (configuration && *configuration)
		ConfigureResultCache(configuration);
}

HttpLLMClient::~HttpLLMClient() = default;

std::string HttpLLMClient::SchedulingKey(const LLMRequest &request) const {
	const auto &endpoint = registry.Resolve(request.model_id);
	return endpoint.scheduling_key;
}

void HttpLLMClient::ConfigureResultCache(const std::string &configuration) {
	if (configuration.empty() || configuration == result_cache_configuration)
		return;
	const auto settings = llm_detail::ParseJSON(configuration);
	if (!settings.is_object())
		throw ValidationError("response_cache must be an object");
	// Prepare the replacement before releasing the current instance. Open shares
	// matching live caches, so equivalent settings retain their populated L1.
	std::string next_configuration = configuration;
	bool next_verification = true;
	if (settings.contains("verify_on_second_use")) {
		const auto &value = settings.at("verify_on_second_use");
		if (!value.is_boolean())
			throw ValidationError("response_cache.verify_on_second_use must be a boolean");
		next_verification = value.get<bool>();
	}
	size_t next_retries = 5;
	if (settings.contains("max_correction_retries")) {
		const auto &value = settings.at("max_correction_retries");
		if (!value.is_number_integer() || value < 0 || value > 64)
			throw ValidationError("response_cache.max_correction_retries must be between 0 and 64");
		next_retries = value.get<size_t>();
	}
	if (!settings.contains("directory") || settings.at("directory").is_null()) {
		result_cache.reset();
		cache_flights.reset();
		verify_on_second_use = next_verification;
		max_correction_retries = next_retries;
		result_cache_configuration.swap(next_configuration);
		return;
	}
	if (!settings.at("directory").is_string() || settings.at("directory").get<std::string>().empty())
		throw ValidationError("response_cache.directory must be a nonempty path or null");
	PromptCacheOptions limits;
	limits.directory = settings.at("directory").get<std::string>();
	auto number = [&](const char *name, size_t &target, bool zero) {
		if (!settings.contains(name))
			return;
		const auto &value = settings.at(name);
		if (!value.is_number_integer() || value < (zero ? 0 : 1) ||
		    (value.is_number_unsigned() && value.get<uint64_t>() > std::numeric_limits<size_t>::max()))
			throw ValidationError(std::string("invalid response_cache.") + name);
		target = value.get<size_t>();
	};
	number("max_disk_bytes", limits.max_disk_bytes, false);
	number("max_memory_bytes", limits.max_memory_bytes, true);
	number("max_entries", limits.max_entries, false);
	number("max_entry_bytes", limits.max_entry_bytes, false);
	if (limits.max_entry_bytes < 56)
		throw ValidationError("response_cache.max_entry_bytes must include at least the 56-byte entry header");
	auto next_cache = PromptResultCache::Open(limits);
	auto next_flights = CacheFlights::Open(next_cache);
	result_cache.swap(next_cache);
	cache_flights.swap(next_flights);
	verify_on_second_use = next_verification;
	max_correction_retries = next_retries;
	result_cache_configuration.swap(next_configuration);
}

void HttpLLMClient::ConfigureRequest(LLMRequest &request) const {
	PrepareRequest(request, true);
}

size_t HttpLLMClient::ConfigureCacheProbe(LLMRequest &request) const {
	PrepareRequest(request, false);
	if (request.cached_response)
		return 0;
	if (!result_cache || request.result_cache_key.empty())
		return 1;
	if (!result_cache->CanStoreKey(request.result_cache_key))
		return std::numeric_limits<size_t>::max();
	LLMResponse response;
	bool confirmed = false;
	if (LookupCached(request, response, confirmed, false, false))
		return confirmed ? 0 : 1;
	// A cold semantic key needs an initial response and, on a later use,
	// independent confirmation. A singleton input still makes only one call.
	return verify_on_second_use && !request.semantic_cache_key.empty() ? 2 : 1;
}

void HttpLLMClient::PrepareRequest(LLMRequest &request, bool erase_invalid) const {
	const auto &endpoint = registry.Resolve(request.model_id);
	if (endpoint.max_tokens)
		request.max_tokens = endpoint.max_tokens;
	if (request.result_cache_checked)
		return;
	request.result_cache_checked = true;
	if (!result_cache || (request.semantic_cache_key.empty() && request.temperature != 0))
		return;
	if (!request.semantic_cache_key.empty()) {
		request.result_cache_key = "semantic-model-result-v4";
		AppendKeyPart(request.result_cache_key, endpoint.scheduling_key);
		// A changed system protocol must not reuse answers produced under an
		// older one, even when task, data and output shape are identical.
		AppendKeyPart(request.result_cache_key, request.system_text);
		AppendKeyPart(request.result_cache_key, request.semantic_cache_key);
		// Different output schemas need independent candidates and
		// confirmations. Keep layout out of this storage contract so compatible
		// renderings of the same semantic input can still share their result.
		AppendKeyPart(request.result_cache_key, CacheContract(request));
	} else {
		request.result_cache_key =
		    Json::array({1, endpoint.base_url, endpoint.model, endpoint.api_key_env, endpoint.headers,
		                 RequestBody(request, endpoint), CacheContract(request), 1})
		        .dump();
	}
	LLMResponse response;
	bool confirmed = false;
	if (LookupCached(request, response, confirmed, false, erase_invalid) && confirmed)
		request.cached_response = std::make_shared<const LLMResponse>(std::move(response));
}

bool HttpLLMClient::LookupCached(const LLMRequest &request, LLMResponse &response, bool &confirmed,
                                 bool refresh_provisional, bool erase_invalid) const {
	if (!result_cache)
		return false;
	auto decode = [&](std::string stored) {
		try {
			confirmed = true;
			if (!request.semantic_cache_key.empty()) {
				const auto envelope = llm_detail::ParseJSON(stored);
				if (!envelope.is_object() || envelope.value("kind", std::string()) != "semantic-result-cache" ||
				    envelope.at("version") != 3 || !envelope.at("confirmed").is_boolean())
					throw ValidationError("invalid semantic result cache envelope");
				// Also validate the stored contract before using its result.
				// An incompatible consumer must not erase another valid contract.
				if (envelope.at("contract") != CacheContract(request))
					return false;
				confirmed = envelope.at("confirmed").get<bool>() || !verify_on_second_use;
				response.input_tokens = TokenCount(envelope, "prompt_tokens");
				stored = envelope.at("result").dump();
			}
			ValidateStructuredPayload(request, stored);
		} catch (...) {
			if (erase_invalid)
				result_cache->Erase(request.result_cache_key);
			return false;
		}
		response.raw_text = std::move(stored);
		response.cache_result_confirmed = confirmed && !request.semantic_cache_key.empty();
		response.result_cache_hit = confirmed;
		response.submit_ns = response.response_ns = 0;
		return true;
	};
	std::string stored;
	const bool found =
	    result_cache->Lookup(request.result_cache_key, stored, !erase_invalid) && decode(std::move(stored));
	// Only an owner about to verify a provisional candidate checks disk. Hot
	// confirmed entries retain their normal O(1) memory lookup path.
	if (refresh_provisional && (!found || !confirmed)) {
		std::string authoritative;
		if (result_cache->LookupDisk(request.result_cache_key, authoritative))
			return decode(std::move(authoritative));
	}
	return found;
}

LLMResponse HttpLLMClient::Infer(const LLMRequest &request) {
	try {
		return InferWithAdmission(request, {});
	} catch (const SamplingBudgetExhausted &) {
		throw;
	} catch (const LLMError &) {
		throw;
	} catch (const std::exception &error) {
		throw LLMError(request, "unresolved", 0, error.what());
	}
}

LLMResponse HttpLLMClient::InferWithAdmission(const LLMRequest &original, const std::function<void()> &admitted) {
	ValidateRequest(original);
	// SQL execution prepares requests before scheduler admission. Reuse that
	// immutable request instead of copying every prompt and cache key again.
	LLMRequest prepared;
	if (!original.result_cache_checked) {
		prepared = original;
		ConfigureRequest(prepared);
		ValidateRequest(prepared);
	}
	const auto &request = original.result_cache_checked ? original : prepared;
	auto reuse = [&](const LLMResponse &source) {
		LLMResponse cached;
		cached.raw_text = source.raw_text;
		cached.result_cache_hit = true;
		cached.input_tokens = source.input_tokens;
		cached.submit_ns = cached.response_ns = 0;
		if (on_cache_hit)
			on_cache_hit(request);
		return cached;
	};
	if (!request.semantic_cache_key.empty() && result_cache && !request.result_cache_key.empty())
		return InferSemanticCached(request, admitted);
	if (request.cached_response)
		return reuse(*request.cached_response);
	if (!result_cache || request.result_cache_key.empty())
		return InferHTTP(request, admitted);
	// A request can wait in the scheduler after its first lookup. Recheck so
	// a result completed during that wait never causes another HTTP send.
	std::string existing;
	if (result_cache->Lookup(request.result_cache_key, existing)) {
		try {
			ValidateStructuredPayload(request, existing);
			LLMResponse cached;
			cached.raw_text = std::move(existing);
			return reuse(cached);
		} catch (const ValidationError &) {
			result_cache->Erase(request.result_cache_key);
		}
	}

	std::shared_ptr<std::promise<LLMResponse>> owner;
	std::shared_future<LLMResponse> future;
	{
		std::lock_guard<std::mutex> guard(cache_flights->mutex);
		const auto found = cache_flights->requests.find(request.result_cache_key);
		if (found != cache_flights->requests.end()) {
			future = found->second;
		} else {
			owner = std::make_shared<std::promise<LLMResponse>>();
			future = owner->get_future().share();
			cache_flights->requests.emplace(request.result_cache_key, future);
		}
	}
	if (!owner) {
		auto response = future.get();
		ValidateStructuredPayload(request, response.raw_text);
		return reuse(response);
	}
	try {
		LLMResponse response;
		// Close the handoff race: another owner may publish and remove its
		// flight between our earlier lookup and creation of this flight.
		std::string published;
		bool hit = result_cache->Lookup(request.result_cache_key, published);
		if (hit) {
			try {
				ValidateStructuredPayload(request, published);
			} catch (...) {
				result_cache->Erase(request.result_cache_key);
				hit = false;
			}
		}
		if (hit) {
			LLMResponse cached;
			cached.raw_text = std::move(published);
			response = reuse(cached);
		} else {
			response = InferHTTP(request, admitted);
			// Only complete, schema-validated responses reach storage.
			result_cache->Store(request.result_cache_key, response.raw_text);
		}
		owner->set_value(response);
		std::lock_guard<std::mutex> guard(cache_flights->mutex);
		cache_flights->requests.erase(request.result_cache_key);
		return response;
	} catch (...) {
		owner->set_exception(std::current_exception());
		std::lock_guard<std::mutex> guard(cache_flights->mutex);
		cache_flights->requests.erase(request.result_cache_key);
		throw;
	}
}

LLMResponse HttpLLMClient::InferSemanticCached(const LLMRequest &request, const std::function<void()> &admitted) {
	auto reuse = [&](const LLMResponse &source) {
		ValidateStructuredPayload(request, source.raw_text);
		LLMResponse cached;
		cached.raw_text = source.raw_text;
		cached.cache_result_confirmed = true;
		cached.result_cache_hit = true;
		cached.input_tokens = source.input_tokens;
		cached.submit_ns = cached.response_ns = 0;
		if (on_cache_hit)
			on_cache_hit(request);
		return cached;
	};
	if (request.cached_response && request.cached_response->cache_result_confirmed)
		return reuse(*request.cached_response);
	const auto contract = CacheContract(request);
	auto flight_key = request.result_cache_key;
	// Clients sharing storage may choose different verification policies.
	AppendKeyPart(flight_key, verify_on_second_use ? "verify" : "reuse");
	LLMResponse provisional;
	bool have_provisional = false;
	for (;;) {
		LLMResponse stored;
		bool confirmed = false;
		if (LookupCached(request, stored, confirmed)) {
			if (confirmed)
				return reuse(stored);
			provisional = std::move(stored);
			have_provisional = true;
		}
		std::shared_ptr<std::promise<LLMResponse>> owner;
		std::shared_future<LLMResponse> future;
		{
			std::lock_guard<std::mutex> guard(cache_flights->mutex);
			auto found = cache_flights->requests.find(flight_key);
			if (found != cache_flights->requests.end()) {
				future = found->second;
			} else {
				owner = std::make_shared<std::promise<LLMResponse>>();
				future = owner->get_future().share();
				cache_flights->requests.emplace(flight_key, future);
			}
		}
		if (!owner) {
			auto completed = future.get();
			ValidateStructuredPayload(request, completed.raw_text);
			if (completed.cache_result_confirmed)
				return reuse(completed);
			// A cold owner has supplied only the first observation. A follower
			// must acquire the next phase and make an independent model call.
			provisional = std::move(completed);
			have_provisional = true;
			continue;
		}
		auto finish = [&]() {
			std::lock_guard<std::mutex> guard(cache_flights->mutex);
			cache_flights->requests.erase(flight_key);
		};
		try {
			// Close publication/removal races before admitting another call.
			if (LookupCached(request, stored, confirmed, have_provisional)) {
				if (confirmed) {
					auto response = reuse(stored);
					finish();
					owner->set_value(response);
					return response;
				}
				provisional = std::move(stored);
				have_provisional = true;
			}
			auto response = InferHTTP(request, admitted);
			response.cache_result_confirmed = !verify_on_second_use;
			auto latest = llm_detail::ParseJSON(response.raw_text);
			Json winner = latest;
			Json summary;
			if (have_provisional) {
				struct Candidate {
					Json value;
					size_t votes;
				};
				std::vector<Candidate> candidates;
				size_t observations = 0;
				auto vote = [&](const Json &value) {
					++observations;
					for (auto &candidate : candidates) {
						if (candidate.value == value) {
							++candidate.votes;
							return;
						}
					}
					candidates.push_back({value, 1});
				};
				auto majority = [&]() -> const Candidate * {
					for (const auto &candidate : candidates)
						if (candidate.votes > observations / 2)
							return &candidate;
					return nullptr;
				};
				Json checks = Json::array();
				auto record = [&](const LLMResponse &result) {
					checks.push_back({{"result_digest", ResultDigest(llm_detail::ParseJSON(result.raw_text).dump())},
					                  {"submit_ns", result.submit_ns},
					                  {"response_ns", result.response_ns},
					                  {"input_tokens", result.input_tokens},
					                  {"output_tokens", result.output_tokens}});
				};
				const auto initial = llm_detail::ParseJSON(provisional.raw_text);
				vote(initial);
				vote(latest);
				record(response);
				size_t retries = 0;
				while (!majority() && retries < max_correction_retries) {
					auto retry = request;
					retry.cache_correction_retry = true;
					// Each send passes the query ledger as a retransmission. Only
					// the initial check emits scheduler admission for this request.
					response = InferHTTP(retry, {});
					latest = llm_detail::ParseJSON(response.raw_text);
					vote(latest);
					record(response);
					++retries;
				}
				auto selected = majority();
				bool fallback = !selected;
				if (!selected) {
					size_t most = 0;
					std::vector<size_t> tied;
					for (size_t i = 0; i < candidates.size(); ++i) {
						if (candidates[i].votes > most) {
							most = candidates[i].votes;
							tied.clear();
						}
						if (candidates[i].votes == most)
							tied.push_back(i);
					}
					static thread_local std::mt19937_64 random(std::random_device {}());
					std::uniform_int_distribution<size_t> choose(0, tied.size() - 1);
					selected = &candidates[tied[choose(random)]];
				}
				winner = selected->value;
				response.cache_result_confirmed = true;
				response.cache_correction_fallback = fallback;
				response.cache_correction_calls = retries + 1;
				summary = {{"observations", observations},
				           {"winner_votes", selected->votes},
				           {"fallback", fallback},
				           {"correction_retries", retries}};
				response.cache_correction_metadata_json = Json({{"provisional_digest", ResultDigest(initial.dump())},
				                                                {"checks", checks},
				                                                {"summary", summary}})
				                                              .dump();
			}
			const Json envelope {{"kind", "semantic-result-cache"},
			                     {"version", 3},
			                     {"prompt_tokens", response.input_tokens},
			                     {"contract", contract},
			                     // Unchecked values remain provisional on disk so
			                     // enabling verification later still checks them.
			                     {"confirmed", verify_on_second_use && response.cache_result_confirmed},
			                     {"result", winner},
			                     {"verification", summary}};
			const auto proposed = envelope.dump();
			const auto published =
			    result_cache->Publish(request.result_cache_key, proposed, [&](const std::string &existing) {
				    try {
					    const auto candidate = llm_detail::ParseJSON(existing);
					    if (candidate.is_object() &&
					        candidate.value("kind", std::string()) == "semantic-result-cache" &&
					        candidate.at("version") == 3 && candidate.at("contract") == contract &&
					        candidate.at("confirmed").is_boolean() &&
					        (candidate.at("confirmed").get<bool>() || !verify_on_second_use)) {
						    ValidateStructuredPayload(request, candidate.at("result").dump());
						    return existing;
					    }
				    } catch (...) {
				    }
				    return proposed;
			    });
			if (published != proposed) {
				const auto authoritative = llm_detail::ParseJSON(published);
				winner = authoritative.at("result");
				ValidateStructuredPayload(request, winner.dump());
				response.cache_result_confirmed = authoritative.at("confirmed").get<bool>() || !verify_on_second_use;
				const auto verification = authoritative.value("verification", Json::object());
				response.cache_correction_fallback = verification.is_object() && verification.value("fallback", false);
				auto details = response.cache_correction_metadata_json.empty()
				                   ? Json::object()
				                   : llm_detail::ParseJSON(response.cache_correction_metadata_json);
				details["publication"] = {{"adopted_confirmed", response.cache_result_confirmed},
				                          {"result_digest", ResultDigest(winner.dump())}};
				response.cache_correction_metadata_json = details.dump();
			}
			const auto canonical = winner.dump();
			if (canonical != response.raw_text) {
				if (response.original_raw_text.empty())
					response.original_raw_text = response.raw_text;
				response.output_normalization =
				    winner == latest ? "semantic_cache_canonical_json" : "cache_correction_vote";
				response.raw_text = canonical;
			}
			finish();
			owner->set_value(response);
			return response;
		} catch (...) {
			// Failed verification leaves the original provisional entry intact.
			// In particular, budget exhaustion never invents a confirmation.
			auto error = std::current_exception();
			finish();
			owner->set_exception(error);
			std::rethrow_exception(error);
		}
	}
}

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
		for (int attempt = 0;;) {
			if (repair_completion && request.output_schema.kind == OutputSchema::Kind::OBJECT) {
				// Only invalid/truncated completions get a recovery instruction.
				// Keep the original task/data/layout intact and never accept or
				// truncate a partial value as a successful result.
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
				body["messages"][0]["content"] =
				    request.system_text + recovery + "Do not add a JSON wrapper, examples or commentary.";
			}
			// Length-terminated responses may retry with a larger output budget, bounded by the configured cap.
			const int attempt_max_tokens =
			    std::min<int64_t>(std::max<int64_t>(4096, base_max_tokens),
			                      int64_t(base_max_tokens) * (int64_t(1) << std::min(attempt, 3)));
			body["max_tokens"] = attempt_max_tokens;
			SharedRequestAdmission shared_admission;
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
					before_send(request, request.cache_correction_retry || attempt > 0 || rate_limit_attempts > 0 ||
					                         server_error_attempts > 0);
				} catch (const SamplingBudgetExhausted &) {
					if (attempt > 0 && !transport_error.empty())
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
				if (retryable && !response_started && response_body.empty() && attempt < transport_retries) {
					std::this_thread::sleep_for(std::chrono::milliseconds(50 * (attempt + 1)));
					++attempt;
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
				if (attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (attempt + 1)));
					++attempt;
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
						const auto text = response.raw_text.substr(first, last - first + 1);
						if (text != "true" && text != "false")
							throw ValidationError("constrained predicate response must contain only true or false");
					}
					// The last standalone true/false token determines the boolean value, regardless of case or wrapper.
					const char *normalization = "last_boolean_token_icase";
					static const std::regex boolean_token(R"(\b(true|false)\b)", std::regex::icase);
					for (auto match =
					         std::sregex_iterator(response.raw_text.begin(), response.raw_text.end(), boolean_token);
					     match != std::sregex_iterator(); ++match) {
						const char initial = (*match)[1].str()[0];
						normalized = initial == 't' || initial == 'T' ? "true" : "false";
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
				if (attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (attempt + 1)));
					++attempt;
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
				if (attempt < incomplete_retries) {
					repair_completion = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(150 * (attempt + 1)));
					++attempt;
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
