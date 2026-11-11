#pragma once
#include <stddef.h>

#if defined(_WIN32)
#if defined(SEMCACHE_STATISTICS_BUILD)
#define SEM_STATISTICS_API __declspec(dllexport)
#else
#define SEM_STATISTICS_API __declspec(dllimport)
#endif
#else
#define SEM_STATISTICS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Version 1, UTF-8 character-domain statistics. No exception crosses this ABI.
 * Strings use explicit byte lengths, including embedded NULs. Handles own their
 * memory and must be destroyed with the matching destroy function. Independent
 * handles may run concurrently; access to one handle must be serialized.
 */
SEM_STATISTICS_API int sem_prefix_version(void);
/* Thread-local last failure; valid until the next failure on this thread.
 * Successful calls need not clear it. Never free the returned string.
 */
SEM_STATISTICS_API const char *sem_prefix_error(void);
/* Capacity >= 2; strictly increasing anchors start at 1 and end at maximum.
 * Failure returns NULL. The input anchor array is copied during this call.
 */
SEM_STATISTICS_API void *sem_prefix_create(size_t capacity, size_t maximum,
                                           const size_t *anchors, size_t count);
/* Returns 0 on success, -1 on failure. A failed add invalidates the handle:
 * subsequent add/json calls fail, and only destroy remains permitted.
 * Input strings need remain valid only for this synchronous call.
 */
SEM_STATISTICS_API int sem_prefix_add(void *handle, const char *const *strings,
                                      const size_t *lengths, size_t count);
/* Owned UTF-8 JSON buffer, valid until next json call or destroy. NULL on
 * error. */
SEM_STATISTICS_API const char *sem_prefix_json(void *handle);
SEM_STATISTICS_API void sem_prefix_destroy(void *handle);

/* Fetch selected zero-based row indices from a zero-based field. Fill strings
 * and byte lengths; buffers must live until the next callback or build return.
 * Return 0 on success, nonzero on error. The callback must not throw.
 */
typedef int (*sem_adaptive_fetch)(size_t field, const size_t *indices,
                                  size_t count, const char **strings,
                                  size_t *lengths, void *context);
/* Config is a JSON object; names is a JSON field-name array matching counts.
 * Sources must remain immutable/replayable during the synchronous build.
 * Returns an owned result or NULL; no partially built result is published.
 */
SEM_STATISTICS_API void *
sem_adaptive_build(const char *config, const char *names, const size_t *counts,
                   size_t fields, sem_adaptive_fetch fetch, void *context);
/* Explicit OFFLINE table build. Field arrays share one row ordinal domain.
 * Adds conditional row samples and a deduplicated row bank to the result.
 * Ordinary sem_adaptive_build never collects these samples. Optional config
 * build_parallelism (1..64) bounds field workers and is not part of statistical
 * identity. Callbacks remain serialized; processing their copied batches may
 * overlap. Ordinary sem_adaptive_build remains serial. */
SEM_STATISTICS_API void *sem_table_build(const char *config, const char *names,
                                         const size_t *counts, size_t fields,
                                         sem_adaptive_fetch fetch,
                                         void *context);
/* Owned UTF-8 JSON buffer, valid until result destruction. NULL for NULL
 * result. */
SEM_STATISTICS_API const char *sem_adaptive_json(void *result);
SEM_STATISTICS_API void sem_adaptive_destroy(void *result);

#ifdef __cplusplus
}
#endif
