#pragma once


#include <map>
#include <set>
#include <string>

namespace semantic {
struct ModelEndpoint {
	std::string base_url;
	std::string model;
	std::string host;
	int port;
	bool tls = false;
	std::string path_prefix;
	std::string scheduling_key;
	std::string api_key_env;
	// Provider-specific routing headers (for example a session id a gateway
	// requires). Authorization stays reserved for api_key_env.
	std::map<std::string, std::string> headers;
	std::string thinking;
	std::string reasoning_effort;
	bool structured_output = false;
	bool has_enable_thinking = false;
	bool enable_thinking = false;
	int max_tokens = 0;
	// Zero inherits the client default; external providers can allow longer connections.
	int connect_timeout_ms = 0;
};

class ModelRegistry {
public:
	static ModelRegistry Parse(const std::string &json);
	const ModelEndpoint &Resolve(const std::string &model_id) const;
	std::set<std::string> KnownModelIds() const;

private:
	std::map<std::string, ModelEndpoint> models;
};
}
