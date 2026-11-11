// Validate every value crossing the model boundary before it can affect a
// relational result; a successful HTTP status alone is insufficient.
#include "llm/llm_client.hpp"
#include <cmath>
#include "plan/semantic_config.hpp"
#include "prompt/row_serializer.hpp"

namespace semantic {
LLMError::LLMError(const LLMRequest &request, const std::string &endpoint, int http_status, const std::string &reason,
                   const LLMResponse *response)
    : std::runtime_error("query=" + request.query_id + " operator=" + std::to_string(request.operator_id) +
                         " request=" + std::to_string(request.request_seq) + " endpoint=" + endpoint +
                         " status=" + std::to_string(http_status) + ": " + reason),
      confirmed_response(response ? std::make_shared<const LLMResponse>(*response) : nullptr) {
}

void ValidateRequest(const LLMRequest &request) {
	// These fields are echoed by the server and identify one row-level call in
	// the query audit stream.
	if (request.query_id.empty() || request.operator_id < 0 || request.request_seq == 0 || request.model_id.empty())
		throw ValidationError("invalid request identity");
	if (!std::isfinite(request.temperature) || request.temperature < 0 || request.temperature > 2 ||
	    request.max_tokens <= 0 || request.max_tokens > 262144)
		throw ValidationError("invalid temperature/max_tokens");

	ValidateUTF8(request.query_id);
	ValidateUTF8(request.model_id);
	ValidateUTF8(request.system_text);
	ValidateUTF8(request.user_text);
	ValidateOutputSchema(request.output_schema);
}
}
