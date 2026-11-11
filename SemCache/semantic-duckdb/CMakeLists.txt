cmake_minimum_required(VERSION 3.16)
project(semantic LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NativeComponents.cmake")
add_subdirectory(src)
if(NOT COMMAND build_loadable_extension)
  message(
    FATAL_ERROR
      "Build semantic through cmake/extensions.cmake using the pinned DuckDB tree"
  )
endif()

# Both extensions are loaded dynamically; no redundant static extension copy.
build_loadable_extension(semantic "" ${SEMANTIC_SOURCES})
semcache_native_target(semantic_loadable_extension 11)
semcache_native_https(semantic_loadable_extension)
target_include_directories(semantic_loadable_extension
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(semantic_loadable_extension SYSTEM
                           PRIVATE "${SEM_ROOT}/third_party")
