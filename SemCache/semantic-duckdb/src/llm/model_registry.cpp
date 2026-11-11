// Reject ambiguous or unsupported endpoint configuration before execution
// starts, rather than failing after some rows have already been processed.
#include "llm/model_registry.hpp"
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include "llm/json_utils.hpp"

namespace semantic {
ModelRegistry ModelRegistry::Parse(const std::string &text) {
	try {
		const auto json = llm_detail::ParseJSON(text);
		if (!json.is_object() || json.empty())
			throw ValidationError("model registry must be a nonempty object");

		ModelRegistry registry;
		// Keep the local numeric-IPv4 HTTP contract, and allow DNS names for
		// certificate-verified HTTPS providers. Userinfo/query/fragment are forbidden.
		static const std::regex url_pattern(
		    "^(http|https)://([A-Za-z0-9][A-Za-z0-9.-]*)(:([0-9]{1,5}))?(/[A-Za-z0-9/_-]*)?$");
		static const std::regex numeric_host("^[0-9.]+$");
		static const std::regex ipv4("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$");
		static const std::regex dns_label("^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?$");
		static const std::regex env_name("^[A-Za-z_][A-Za-z0-9_]*$");
		for (auto it = json.begin(); it != json.end(); ++it) {
			const auto &value = it.value();
			if (it.key().empty() || !value.is_object() || !value.contains("base_url") || !value.contains("model") ||
			    !value.at("base_url").is_string() || !value.at("model").is_string())
				throw ValidationError("registry entry requires base_url/model strings");
			for (auto field = value.begin(); field != value.end(); ++field) {
				if (field.key() != "base_url" && field.key() != "model" && field.key() != "structured_output" &&
				    field.key() != "enable_thinking" && field.key() != "max_tokens" && field.key() != "api_key_env" &&
				    field.key() != "thinking" && field.key() != "reasoning_effort" &&
				    field.key() != "connect_timeout_ms" && field.key() != "headers")
					throw ValidationError("unknown registry entry field: " + field.key());
			}

			ModelEndpoint endpoint;
			endpoint.base_url = value.at("base_url").get<std::string>();
			endpoint.model = value.at("model").get<std::string>();
			if (value.contains("api_key_env")) {
				if (!value.at("api_key_env").is_string() ||
				    !std::regex_match(value.at("api_key_env").get<std::string>(), env_name))
					throw ValidationError("api_key_env must be an environment variable name");
				endpoint.api_key_env = value.at("api_key_env").get<std::string>();
			}
			if (value.contains("headers")) {
				const auto &headers = value.at("headers");
				if (!headers.is_object() || headers.empty())
					throw ValidationError("headers must be a nonempty object");
				for (auto header = headers.begin(); header != headers.end(); ++header) {
					if (!header.value().is_string() || header.key().empty())
						throw ValidationError("headers must map names to string values");
					std::string name = header.key();
					std::transform(name.begin(), name.end(), name.begin(),
					               [](unsigned char c) { return std::tolower(c); });
					if (name == "authorization")
						throw ValidationError("headers must not set authorization");
					endpoint.headers[header.key()] = header.value().get<std::string>();
				}
			}
			if (value.contains("reasoning_effort")) {
				if (!value.at("reasoning_effort").is_string() ||
				    value.at("reasoning_effort").get<std::string>().empty())
					throw ValidationError("reasoning_effort must be a nonempty string");
				endpoint.reasoning_effort = value.at("reasoning_effort").get<std::string>();
			}
			if (value.contains("thinking")) {
				if (!value.at("thinking").is_string() ||
				    (value.at("thinking") != "disabled" && value.at("thinking") != "enabled"))
					throw ValidationError("thinking must be enabled or disabled");
				if (value.contains("enable_thinking"))
					throw ValidationError("thinking and enable_thinking cannot both be configured");
				endpoint.thinking = value.at("thinking").get<std::string>();
			}
			if (value.contains("structured_output")) {
				if (!value.at("structured_output").is_boolean())
					throw ValidationError("structured_output must be boolean");
				endpoint.structured_output = value.at("structured_output").get<bool>();
			}
			if (value.contains("enable_thinking")) {
				if (!value.at("enable_thinking").is_boolean())
					throw ValidationError("enable_thinking must be boolean");
				endpoint.has_enable_thinking = true;
				endpoint.enable_thinking = value.at("enable_thinking").get<bool>();
			}
			if (value.contains("max_tokens")) {
				const auto &limit = value.at("max_tokens");
				if (!limit.is_number_integer() || limit < 1 || limit > 262144)
					throw ValidationError("max_tokens must be an integer in [1, 262144]");
				endpoint.max_tokens = limit.get<int>();
			}
			if (value.contains("connect_timeout_ms")) {
				const auto &limit = value.at("connect_timeout_ms");
				if (!limit.is_number_integer() || limit < 1 || limit > 600000)
					throw ValidationError("connect_timeout_ms must be an integer in [1, 600000]");
				endpoint.connect_timeout_ms = limit.get<int>();
			}
			std::smatch match;
			if (endpoint.model.empty() || !std::regex_match(endpoint.base_url, match, url_pattern))
				throw ValidationError(
				    "invalid model or base_url: expected http://IPv4:port[/prefix] or https://host[:port][/prefix]");

			endpoint.tls = match[1].str() == "https";
			endpoint.host = match[2].str();
			endpoint.port = match[4].matched ? std::stoi(match[4].str()) : 443;
			if (endpoint.port < 1 || endpoint.port > 65535)
				throw ValidationError("invalid endpoint port");
			const bool numeric = std::regex_match(endpoint.host, numeric_host);
			if (!endpoint.tls && (!numeric || !match[4].matched))
				throw ValidationError("plain HTTP requires numeric IPv4 and explicit port");
			if ((numeric && !std::regex_match(endpoint.host, ipv4)) || endpoint.host.size() > 253 ||
			    endpoint.host.back() == '.')
				throw ValidationError("invalid endpoint host");

			std::istringstream parts(endpoint.host);
			std::string part;
			while (std::getline(parts, part, '.')) {
				if (numeric && (part.size() > 3 || (part.size() > 1 && part[0] == '0') || std::stoi(part) > 255))
					throw ValidationError("invalid numeric IPv4 address");
				if (!numeric && (part.size() > 63 || !std::regex_match(part, dns_label)))
					throw ValidationError("invalid HTTPS hostname");
			}

			endpoint.path_prefix = match[5].str();
			if (!endpoint.path_prefix.empty() && endpoint.path_prefix.back() == '/')
				endpoint.path_prefix.pop_back();
			endpoint.scheduling_key = llm_detail::Json::array({endpoint.tls, endpoint.host, endpoint.port,
			                                                   endpoint.path_prefix, endpoint.model})
			                              .dump();
			registry.models.emplace(it.key(), std::move(endpoint));
		}
		return registry;
	} catch (const llm_detail::Json::exception &error) {
		throw ValidationError(std::string("invalid model registry JSON: ") + error.what());
	}
}

const ModelEndpoint &ModelRegistry::Resolve(const std::string &id) const {
	const auto entry = models.find(id);
	if (entry == models.end())
		throw ValidationError("unknown model_id: " + id);
	return entry->second;
}

std::set<std::string> ModelRegistry::KnownModelIds() const {
	std::set<std::string> ids;
	for (const auto &entry : models)
		ids.insert(entry.first);
	return ids;
}
}
