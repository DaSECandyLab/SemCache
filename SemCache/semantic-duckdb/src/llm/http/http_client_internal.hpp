#pragma once

#include "llm/http_llm_client.hpp"
#include "llm/request_scheduler.hpp"
#include "llm/json_utils.hpp"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
// Retry transport failures only before any response byte arrives; a partial response may already be billed.
#define CPPHTTPLIB_NO_SSL_PEER_RETRY
#include "cpp-httplib/httplib.h"

namespace semantic {
namespace http_detail {

using Json = llm_detail::Json;

int64_t Now();
int64_t TokenCount(const Json &usage, const char *name);
void ValidateOutputField(const Json &value, const Field &field);
void ValidateStructuredResult(const Json &value, const OutputSchema &schema);
void ValidateStructuredPayload(const LLMRequest &request, const std::string &raw);
std::string CacheContract(const LLMRequest &request);
std::string NormalizeProjectionText(const std::string &raw, const OutputSchema &schema);
void AppendKeyPart(std::string &key, const std::string &part);
std::string ResultDigest(const std::string &value);
Json RequestBody(const LLMRequest &request, const ModelEndpoint &endpoint);

} // namespace http_detail

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
	static std::shared_ptr<CacheFlights> Open(const std::shared_ptr<PromptResultCache> &cache);
};

} // namespace semantic
