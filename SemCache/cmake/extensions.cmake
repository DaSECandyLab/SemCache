get_filename_component(SEMCACHE_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.."
                       ABSOLUTE)
duckdb_extension_load(semantic DONT_LINK SOURCE_DIR
                      "${SEMCACHE_SOURCE_ROOT}/semantic-duckdb")
duckdb_extension_load(beam_search DONT_LINK SOURCE_DIR
                      "${SEMCACHE_SOURCE_ROOT}/beam-search-duckdb")
duckdb_extension_load(tpch DONT_LINK)
