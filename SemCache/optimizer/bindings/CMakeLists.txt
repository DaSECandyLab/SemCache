cmake_minimum_required(VERSION 3.16)
project(semcache_python_bindings LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/NativeComponents.cmake")
set(BEAM_SRC "${SEM_ROOT}/beam-search-duckdb/src")
find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
if(NOT DEFINED SEM_DUCKDB_BUILD)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print((p.parents[2]/json.loads(p.read_text())['runtime']['build']).resolve())"
      "${SEM_ROOT}/config/system/runtime.json"
    OUTPUT_VARIABLE SEM_DEFAULT_DUCKDB_BUILD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SEM_CONFIG_STATUS)
  if(NOT SEM_CONFIG_STATUS EQUAL 0)
    message(FATAL_ERROR "Cannot read the default DuckDB build from config/system/runtime.json")
  endif()
endif()
set(SEM_DUCKDB_BUILD
    "${SEM_DEFAULT_DUCKDB_BUILD}"
    CACHE PATH "Pinned DuckDB build containing beam_search")
set(SEM_PYTHON_MODULE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/overlay/planning/kernels"
    CACHE PATH "Output directory for the Python planning module")
set(BEAM_BINARY
    "${SEM_DUCKDB_BUILD}/extension/beam_search/beam_search.duckdb_extension")
if(NOT EXISTS "${BEAM_BINARY}")
  message(
    FATAL_ERROR
      "Build the beam_search_loadable_extension target in ${SEM_DUCKDB_BUILD} before building Python bindings"
  )
endif()
add_library(beam_core SHARED IMPORTED GLOBAL)
set_target_properties(beam_core PROPERTIES IMPORTED_LOCATION "${BEAM_BINARY}")
set(STATISTICS_BINARY "${SEM_DUCKDB_BUILD}/shared/libsemcache_statistics.so")
if(NOT EXISTS "${STATISTICS_BINARY}")
  message(
    FATAL_ERROR
      "Build semcache_statistics in ${SEM_DUCKDB_BUILD} before building Python bindings"
  )
endif()
add_library(statistics_core SHARED IMPORTED GLOBAL)
set_target_properties(
  statistics_core
  PROPERTIES IMPORTED_LOCATION "${STATISTICS_BINARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${SEM_ROOT}/statistics/include")
find_package(Threads REQUIRED)

if(NOT SEM_PYBIND_INCLUDE)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import importlib.util,pathlib; s=importlib.util.find_spec('pybind11'); t=importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent/'include' if s else pathlib.Path(t.origin).parent/'include' if t else '')"
    OUTPUT_VARIABLE SEM_PYBIND_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT EXISTS "${SEM_PYBIND_INCLUDE}/pybind11/pybind11.h")
  message(FATAL_ERROR "Selected Python needs pybind11 or Torch headers")
endif()
python_add_library(
  _search_core
  MODULE
  WITH_SOABI
  search_bindings.cpp
  state_bindings.cpp
  resident_bindings.cpp
  resident_engine_bindings.cpp
  cache_terms.cpp
  features/feature_buffers.cpp
  features/native_features_stats.cpp
  features/native_features_build.cpp
  features/native_features_pack.cpp)
semcache_native_target(_search_core 17)
set_target_properties(
  _search_core
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SEM_PYTHON_MODULE_DIR}"
    BUILD_RPATH
    "${SEM_DUCKDB_BUILD}/extension/beam_search;${SEM_DUCKDB_BUILD}/shared")
target_include_directories(
  _search_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${BEAM_SRC}" "${BEAM_SRC}/search"
                       "${BEAM_SRC}/search/drivers" "${BEAM_SRC}/search/pruning"
                       "${SEM_PYBIND_INCLUDE}" "${SEM_ROOT}/third_party")
target_link_libraries(_search_core PRIVATE beam_core statistics_core
                                           Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(_search_core PRIVATE -Wall -Wextra -ffp-contract=off)
