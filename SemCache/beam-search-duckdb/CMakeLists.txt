cmake_minimum_required(VERSION 3.16)
project(beam_search LANGUAGES CXX)

if(NOT COMMAND build_loadable_extension)
  message(
    FATAL_ERROR
      "Build beam_search through cmake/extensions.cmake using the pinned DuckDB tree"
  )
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NativeComponents.cmake")
set(BEAM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
find_package(Threads REQUIRED)
find_path(BEAM_BOOST_INCLUDE boost/multiprecision/cpp_int.hpp REQUIRED)
add_subdirectory("${SEM_ROOT}/statistics"
                 "${CMAKE_CURRENT_BINARY_DIR}/statistics")

# Core objects are built once into this loadable extension. The Python shim
# links this exact artifact; no static extension or second planning core.
add_subdirectory(src)

build_loadable_extension(beam_search "" ${BEAM_SOURCES})
semcache_native_target(beam_search_loadable_extension 17)
set_target_properties(
  beam_search_loadable_extension
  PROPERTIES CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN ON
             BUILD_RPATH "$ORIGIN/../../shared")
target_include_directories(
  beam_search_loadable_extension
  PRIVATE "${BEAM_SRC}" "${BEAM_SRC}/search" "${BEAM_BOOST_INCLUDE}"
          "${SEM_ROOT}/statistics/include" "${SEM_ROOT}/third_party"
          "${SEM_MBED}/include")
target_link_libraries(beam_search_loadable_extension Threads::Threads
                      ${CMAKE_DL_LIBS} semcache_statistics)
target_compile_options(beam_search_loadable_extension PRIVATE -ffp-contract=off)
