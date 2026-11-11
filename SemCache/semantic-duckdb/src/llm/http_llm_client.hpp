#pragma once


#include <cstddef>
#include <functional>
#include <exception>
#include <memory>
#include <mutex>
#include <future>
#include <unordered_map>
#include "llm/prompt_result_cache.hpp"
#include "llm/llm_client.hpp"
#include "llm/model_registry.hpp"

namespace semantic {
struct SamplingBudgetExhausted : std::exception {
	const char *what() const noexcept override {
		return "Native sampling HTTP request allowance exhausted";
	}
};

struct HTTPOptions {
	int connect_timeout_ms = 1000;
	int read_timeout_ms = 60000;
	int write_timeout_ms = 1000;
	size_t max_response_bytes = 1024 * 1024;
};

// Non-streaming HTTP/verified HTTPS chat adapter. No redirects; a transport
// failure that never produced a response byte is retried a bounded number of
// times, and the admission hook is told which attempts are retransmissions.
class HttpLLMClient : public LLMClient {
public:
	struct ConnectionPool;
	explicit HttpLLMClient(ModelRegistry registry, HTTPOptions options = HTTPOptions(),
	                       std::function<void(const LLMRequest &, bool retransmit)> before_send = {},
	                       std::shared_ptr<ConnectionPool> pool = nullptr,
	                       std::function<void(const LLMRequest &)> on_cache_hit = {});
	~HttpLLMClient() override;
	HttpLLMClient(const HttpLLMClient &) = delete;
	HttpLLMClient &operator=(const HttpLLMClient &) = delete;
	void ConfigureRequest(LLMRequest &request) const override;
	// Populate confirmed cache evidence without sending or erasing an entry.
	size_t ConfigureCacheProbe(LLMRequest &request) const;
	std::string SchedulingKey(const LLMRequest &request) const override;
	LLMResponse Infer(const LLMRequest &request) override;
	LLMResponse InferWithAdmission(const LLMRequest &request, const std::function<void()> &admitted) override;
	void ConfigureResultCache(const std::string &configuration);
	bool ResultCacheEnabled() const {
		return bool(result_cache);
	}
	bool ResultCacheVerificationEnabled() const {
		return result_cache && verify_on_second_use;
	}
	std::set<std::string> KnownModelIds() const {
		return registry.KnownModelIds();
	}
	std::shared_ptr<ConnectionPool> Connections() const {
		return connections;
	}

private:
	LLMResponse InferHTTP(const LLMRequest &request, const std::function<void()> &admitted);
	LLMResponse InferSemanticCached(const LLMRequest &request, const std::function<void()> &admitted);
	bool LookupCached(const LLMRequest &request, LLMResponse &response, bool &confirmed,
	                  bool refresh_provisional = false, bool erase_invalid = true) const;
	void PrepareRequest(LLMRequest &request, bool erase_invalid) const;
	struct CacheFlights;
	const ModelRegistry registry;
	const HTTPOptions options;
	const std::function<void(const LLMRequest &, bool retransmit)> before_send;
	std::shared_ptr<ConnectionPool> connections;
	std::function<void(const LLMRequest &)> on_cache_hit;
	std::shared_ptr<PromptResultCache> result_cache;
	std::string result_cache_configuration;
	bool verify_on_second_use = true;
	size_t max_correction_retries = 5;
	std::shared_ptr<CacheFlights> cache_flights;
};
}
