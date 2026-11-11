#pragma once

#include "llm/json_utils.hpp"

namespace duckdb {
namespace semantic_extension {

// QueryIR scheduling metadata follows produced fields and actual input flow.
semantic::llm_detail::Json SemanticDependencies(const semantic::llm_detail::Json &query);

}
}
