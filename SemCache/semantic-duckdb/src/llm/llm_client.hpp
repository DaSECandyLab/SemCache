#pragma once

// Transport-neutral requests keep operators independent of HTTP details.

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <string>
#include <functional>
#include "plan/semantic_config.hpp"

namespace semantic {
struct LLMResponse;
struct LLMRequest {
	std::string query_id;
	int64_t operator_id = -1;
	uint64_t request_seq = 0;
	std::string model_id;
	std::string system_text;
	std::string user_text;
	OutputSchema output_schema;
	double temperature = 0.0;
	int64_t max_tokens = 64;
	// Versioned semantic identity from the renderer: operator protocol, rebound
	// task and rendered data. Transport adds the system and physical-model identity.
	std::string semantic_cache_key;
	// Internal correction sends retain one logical request/admission identity.
	bool cache_correction_retry = false;
	// Filled only by the transport layer; never part of the model request.
	bool result_cache_checked = false;
	std::string result_cache_key;
	std::shared_ptr<const LLMResponse> cached_response;
};

struct LLMResponse {
	bool result_cache_hit = false;
	bool cache_result_confirmed = false;
	bool cache_correction_fallback = false;
	size_t cache_correction_calls = 0;
	std::string cache_correction_metadata_json;
	std::string raw_text;
	// Preserve the provider completion when its output is normalized.
	std::string original_raw_text;
	std::string output_normalization;
	int64_t submit_ns = -1;
	int64_t response_ns = -1;
	// Unknown counts/timestamps stay unknown, never estimated from text length.
	int64_t server_start_ns = -1;
	int64_t input_tokens = -1;
	int64_t output_tokens = -1;
	// Serving-reported prefix-cache reads; absent metadata remains unknown.
	int64_t cached_input_tokens = -1;
	std::string server_metadata_json;
};

class LLMError : public std::runtime_error {
public:
	LLMError(const LLMRequest &request, const std::string &endpoint, int http_status, const std::string &reason,
	         const LLMResponse *confirmed_response = nullptr);
	// A validated HTTP envelope with an incomplete completion is a confirmed
	// physical cache source, but never a valid result/quality observation.
	std::shared_ptr<const LLMResponse> confirmed_response;
};

void ValidateRequest(const LLMRequest &request);

class LLMClient {
public:
	virtual std::string SchedulingKey(const LLMRequest &request) const {
		return request.model_id;
	}
	virtual void ConfigureRequest(LLMRequest &) const {
	}
	virtual LLMResponse Infer(const LLMRequest &request) = 0;
	virtual LLMResponse InferWithAdmission(const LLMRequest &request, const std::function<void()> &admitted) {
		if (admitted)
			admitted();
		return Infer(request);
	}
	virtual ~LLMClient() = default;
};
}
