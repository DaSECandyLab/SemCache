get_filename_component(SEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(SEM_MBED "${SEM_ROOT}/third_party/duckdb/third_party/mbedtls")
include_guard(GLOBAL)

function(semcache_native_target target standard)
  set_target_properties(${target} PROPERTIES CXX_STANDARD ${standard}
                                             CXX_STANDARD_REQUIRED ON)
endfunction()

# cpp-httplib is header-only; every transport target must compile with the same
# TLS support and link OpenSSL for certificate and hostname verification.
function(semcache_native_https target)
  # Development headers come from the tool virtualenv; runtime OpenSSL libraries come from the host.
  if(NOT OPENSSL_INCLUDE_DIR
     AND EXISTS "${SEM_ROOT}/.venv-tools/include/openssl/ssl.h")
    set(OPENSSL_INCLUDE_DIR
        "${SEM_ROOT}/.venv-tools/include"
        CACHE PATH "OpenSSL include directory" FORCE)
  endif()
  if(NOT OPENSSL_SSL_LIBRARY)
    find_library(OPENSSL_SSL_LIBRARY NAMES ssl libssl.so.3)
  endif()
  if(NOT OPENSSL_CRYPTO_LIBRARY)
    find_library(OPENSSL_CRYPTO_LIBRARY NAMES crypto libcrypto.so.3)
  endif()
  find_package(OpenSSL REQUIRED)
  target_compile_definitions(${target} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
  # DuckDB extension targets use CMake's plain linkage signature; standalone
  # static libraries pass PUBLIC to propagate their link dependencies.
  target_link_libraries(${target} ${ARGN} OpenSSL::SSL OpenSSL::Crypto)
endfunction()
