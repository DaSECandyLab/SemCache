// Keep HTTP concerns at this boundary: build one chat request, enforce resource
// limits, then return a transport-neutral response to the execution layer.
#include "llm/http/http_client_internal.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <openssl/evp.h>

namespace semantic {
namespace http_detail {

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

} // namespace http_detail

using namespace http_detail;

std::shared_ptr<HttpLLMClient::CacheFlights> HttpLLMClient::CacheFlights::Open(
    const std::shared_ptr<PromptResultCache> &cache) {
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

} // namespace semantic
