# Production source lists; target configuration belongs to the extension root.
set(SEMANTIC_SOURCES
    extension/semantic_extension.cpp
    operators/binding/semantic_bind.cpp
    operators/binding/semantic_sql_bind.cpp
    optimizer/optimizer_client.cpp
    optimizer/semantic_optimizer_sampling.cpp
    optimizer/semantic_optimizer_extract.cpp
    optimizer/semantic_optimizer.cpp
    optimizer/semantic_dependencies.cpp
    operators/execution/semantic_execution.cpp
    runtime/result_capture.cpp
    operators/execution/semantic_sampling.cpp
    operators/execution/semantic_unary.cpp
    operators/execution/join/physical_join_source.cpp
    operators/execution/join/left_major_cross.cpp
    operators/execution/join/logical_join.cpp
    operators/plan/semantic_logical_operator.cpp
    operators/semantic_operators.cpp
    operators/plan/rewrite/semantic_predicates.cpp
    operators/plan/semantic_plan_tree.cpp
    operators/plan/semantic_plan_tree_passes.cpp
    operators/plan/rewrite/semantic_projection_lineage.cpp
    operators/plan/semantic_rewriter.cpp
    runtime/query/runtime_core.cpp
    runtime/query/runtime_reopt.cpp
    runtime/query/query_state_rows.cpp
    runtime/runtime_progress.cpp
    runtime/runtime_plan_state.cpp)

set(SEMANTIC_PROMPT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/plan/semantic_config.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/prompt/row_serializer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/prompt/prompt_renderer.cpp")
set(SEMANTIC_LLM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/llm_client.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/cache/cache_storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/cache/prompt_result_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/model_registry.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/http/http_client_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/http/http_client_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/http/http_client_transport.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/llm/request_scheduler.cpp")

list(TRANSFORM SEMANTIC_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND SEMANTIC_SOURCES ${SEMANTIC_PROMPT_SOURCES} ${SEMANTIC_LLM_SOURCES})
foreach(group SEMANTIC_SOURCES SEMANTIC_PROMPT_SOURCES SEMANTIC_LLM_SOURCES)
  set(${group}
      ${${group}}
      PARENT_SCOPE)
endforeach()
