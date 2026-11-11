cmake_minimum_required(VERSION 3.16)
project(semcache_statistics LANGUAGES CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NativeComponents.cmake")
add_library(
  semcache_statistics SHARED
  src/bottom_k/prefix_sketch.cpp "${SEM_MBED}/library/sha256.cpp"
  "${SEM_MBED}/library/platform_util.cpp")
set_target_properties(
  semcache_statistics
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shared")
semcache_native_target(semcache_statistics 17)
target_include_directories(semcache_statistics
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(semcache_statistics PRIVATE "${SEM_ROOT}/third_party"
                                                       "${SEM_MBED}/include")
target_compile_definitions(semcache_statistics
                           PRIVATE SEMCACHE_STATISTICS_BUILD)
target_compile_options(semcache_statistics PRIVATE -ffp-contract=off)

find_package(Threads REQUIRED)
target_link_libraries(semcache_statistics PRIVATE Threads::Threads)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(semcache_statistics_build src/offline/main.cpp)
  semcache_native_target(semcache_statistics_build 17)
  target_include_directories(
    semcache_statistics_build
    PRIVATE "${SEM_ROOT}/third_party"
            "${SEM_ROOT}/third_party/duckdb/src/include")
  target_link_libraries(semcache_statistics_build PRIVATE duckdb
                                                          semcache_statistics)
  target_compile_definitions(semcache_statistics_build
                             PRIVATE SEM_ROOT_PATH="${SEM_ROOT}")
  set_target_properties(
    semcache_statistics_build
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
               BUILD_RPATH "$ORIGIN/../src;$ORIGIN/../shared")
endif()
