#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace semantic {

struct PromptCacheOptions {
	std::string directory;
	size_t max_disk_bytes = 268435456;
	size_t max_memory_bytes = 16777216;
	size_t max_entries = 32768;
	size_t max_entry_bytes = 262144;
};

// Validated responses share bounded memory and sharded disk storage. Exact keys resolve hash collisions.
// Pending writes share the memory budget; a background writer persists them, with backpressure when full.
// Shard generations invalidate disk-usage snapshots; nanosecond timestamps provide LRU ordering.
class PromptResultCache {
public:
	static std::shared_ptr<PromptResultCache> Open(const PromptCacheOptions &options);
	~PromptResultCache();
	bool Lookup(const std::string &key, std::string &response, bool read_only = false);
	bool CanStoreKey(const std::string &key) const;
	// Bypass stale L1 state once, then refresh L1 from the authoritative disk.
	// A miss leaves the local value intact; memory-only caches use Lookup.
	bool LookupDisk(const std::string &key, std::string &response);
	// Synchronous read/select/publication under the shard's process/file locks.
	// select receives disk state, else local state, else an empty string, and
	// must be pure and must not call this cache. Failures return proposed.
	std::string Publish(const std::string &key, const std::string &proposed,
	                    const std::function<std::string(const std::string &existing)> &select);
	void Store(const std::string &key, const std::string &response);
	void Erase(const std::string &key);
	void Flush();

private:
	explicit PromptResultCache(const PromptCacheOptions &options);
	struct Impl;
	std::unique_ptr<Impl> impl;
};

}
