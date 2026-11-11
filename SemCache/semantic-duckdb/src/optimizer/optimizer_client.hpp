#pragma once

// Narrow client interface between DuckDB planning and the external optimizer.
// The wire protocol carries JSON only and never exposes DuckDB object pointers.

#include "llm/json_utils.hpp"

namespace semantic {
// Version 1: one length-framed JSON request/response per connection, no retries.
class UnixSocketOptimizerClient {
public:
	UnixSocketOptimizerClient(std::string path, int timeout_ms);
	llm_detail::Json Optimize(const llm_detail::Json &query);

private:
	std::string path;
	int timeout_ms;
};
}
