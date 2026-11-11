#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "operators/semantic_operators.hpp"

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(semantic, loader) {
	duckdb::RegisterSemanticOperators(loader);
}
}
