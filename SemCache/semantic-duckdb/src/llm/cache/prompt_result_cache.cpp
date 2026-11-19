#include "llm/cache/prompt_cache_internal.hpp"

#include <fcntl.h>
#include <map>
#include <tuple>
#include <unistd.h>

namespace semantic {

using namespace prompt_cache_detail;

PromptResultCache::PromptResultCache(const PromptCacheOptions &options) : impl(new Impl(options)) {
}
PromptResultCache::~PromptResultCache() = default;

std::shared_ptr<PromptResultCache> PromptResultCache::Open(const PromptCacheOptions &options) {
	static std::mutex mutex;
	using Identity = std::tuple<std::string, size_t, size_t, size_t, size_t>;
	static std::map<Identity, std::weak_ptr<PromptResultCache>> instances;
	std::lock_guard<std::mutex> guard(mutex);
	for (auto entry = instances.begin(); entry != instances.end();) {
		if (entry->second.expired())
			entry = instances.erase(entry);
		else
			++entry;
	}
	const Identity identity(options.directory, options.max_disk_bytes, options.max_memory_bytes, options.max_entries,
	                        options.max_entry_bytes);
	auto found = instances.find(identity);
	if (found != instances.end()) {
		auto shared = found->second.lock();
		if (shared)
			return shared;
	}
	auto result = std::shared_ptr<PromptResultCache>(new PromptResultCache(options));
	instances[identity] = result;
	return result;
}

bool PromptResultCache::CanStoreKey(const std::string &key) const {
	return impl->Fits(key, std::string());
}

bool PromptResultCache::Lookup(const std::string &key, std::string &response, bool read_only) {
	try {
		if (!impl->Fits(key, std::string()))
			return false;
		std::string disk_digest;
		if (impl->Recall(key, response, disk_digest, read_only)) {
			if (!read_only && !disk_digest.empty())
				impl->Touch(disk_digest);
			return true;
		}
		if (impl->options.directory.empty() || !impl->shards)
			return false;
		const auto digest = Digest(key), directory = impl->Shard(digest);
		std::lock_guard<std::mutex> local(impl->shard_mutexes[impl->ShardIndex(digest)]);
		if (impl->Recall(key, response, disk_digest, read_only)) {
			if (!read_only && !disk_digest.empty())
				impl->Touch(disk_digest);
			return true;
		}
		struct stat info;
		if (stat(directory.c_str(), &info) || !S_ISDIR(info.st_mode))
			return false;
		ShardLock lock(directory);
		if (!impl->Read(directory + "/" + Hex(digest) + ".entry", key, response, lock, read_only))
			return false;
		{
			std::lock_guard<std::mutex> guard(impl->mutex);
			// Store can enqueue while the disk read is in progress. Prefer its
			// newer value and never overwrite a queued entry with an old read.
			auto found = impl->memory.find(key);
			if (found != impl->memory.end()) {
				response = found->second->response;
				impl->lru.splice(impl->lru.begin(), impl->lru, found->second);
			} else {
				impl->Remember(key, response);
			}
		}
		return true;
	} catch (...) {
		return false;
	}
}

bool PromptResultCache::LookupDisk(const std::string &key, std::string &response) {
	try {
		if (!impl->Fits(key, std::string()))
			return false;
		if (impl->options.directory.empty() || !impl->shards)
			return Lookup(key, response);
		const auto digest = Digest(key), directory = impl->Shard(digest);
		const auto shard_index = impl->ShardIndex(digest);
		for (;;) {
			std::unique_lock<std::mutex> local(impl->shard_mutexes[shard_index]);
			bool restore_pending;
			{
				std::unique_lock<std::mutex> guard(impl->mutex);
				if (impl->Writing(key)) {
					// The writer may itself be waiting for our shard mutex.
					local.unlock();
					impl->changed.wait(guard, [&]() { return !impl->Writing(key); });
					continue;
				}
				restore_pending = impl->BeginPublication(key, shard_index);
			}
			Impl::Publication publication {*impl, key, shard_index, restore_pending};
			impl->changed.notify_all();
			struct stat info;
			if (stat(directory.c_str(), &info) || !S_ISDIR(info.st_mode))
				return false;
			ShardLock lock(directory);
			if (!impl->Read(directory + "/" + Hex(digest) + ".entry", key, response, lock))
				return false;
			{
				std::lock_guard<std::mutex> guard(impl->mutex);
				// Cancel the stale queued value rather than letting it overwrite
				// the authoritative state after this explicit refresh returns.
				impl->Forget(key);
				impl->Remember(key, response);
			}
			return true;
		}
	} catch (...) {
		return false;
	}
}

std::string PromptResultCache::Publish(const std::string &key, const std::string &proposed,
                                       const std::function<std::string(const std::string &existing)> &select) {
	try {
		if (!impl->Fits(key, std::string()))
			return proposed;
		if (impl->options.directory.empty() || !impl->shards) {
			std::lock_guard<std::mutex> guard(impl->mutex);
			auto found = impl->memory.find(key);
			const auto selected = select(found == impl->memory.end() ? std::string() : found->second->response);
			if (!impl->Fits(key, selected))
				return proposed;
			impl->Forget(key);
			impl->Remember(key, selected);
			return selected;
		}
		const auto digest = Digest(key), directory = impl->Shard(digest);
		const auto shard_index = impl->ShardIndex(digest);
		for (;;) {
			std::unique_lock<std::mutex> local(impl->shard_mutexes[shard_index]);
			bool restore_pending;
			{
				std::unique_lock<std::mutex> guard(impl->mutex);
				if (impl->Writing(key)) {
					local.unlock();
					impl->changed.wait(guard, [&]() { return !impl->Writing(key); });
					continue;
				}
				restore_pending = impl->BeginPublication(key, shard_index);
			}
			Impl::Publication publication {*impl, key, shard_index, restore_pending};
			impl->changed.notify_all();
			MakeDirectories(directory);
			ShardLock lock(directory);
			const auto path = directory + "/" + Hex(digest) + ".entry";
			std::string existing;
			const bool disk_hit = impl->Read(path, key, existing, lock);
			if (!disk_hit) {
				std::lock_guard<std::mutex> guard(impl->mutex);
				auto found = impl->memory.find(key);
				if (found != impl->memory.end())
					existing = found->second->response;
			}
			// Only a small pure selector runs here, never network/model work.
			const auto selected = select(existing);
			if (!impl->Fits(key, selected))
				return proposed;
			if ((!disk_hit || selected != existing) && !impl->Write(directory, path, key, selected, lock, shard_index))
				return proposed;
			{
				std::lock_guard<std::mutex> guard(impl->mutex);
				impl->Forget(key);
				impl->Remember(key, selected);
			}
			return selected;
		}
	} catch (...) {
		// Cache/selector failures must not change the normal query result.
		return proposed;
	}
}

void PromptResultCache::Store(const std::string &key, const std::string &response) {
	try {
		if (!impl->Fits(key, response))
			return;
		const bool disk = !impl->options.directory.empty() && impl->shards;
		const auto digest = disk ? Digest(key) : std::string();
		const size_t shard_index = disk ? impl->ShardIndex(digest) : 0;
		const auto weight = Weight(key, response);
		if (!disk || weight <= impl->options.max_memory_bytes) {
			std::unique_lock<std::mutex> guard(impl->mutex);
			if (disk)
				impl->EnsureWriterLocked();
			if (!disk || impl->writer.joinable()) {
				for (;;) {
					impl->changed.wait(guard, [&]() {
						return !impl->Writing(key) && (!disk || !impl->publishing[shard_index]);
					});
					impl->Forget(key);
					if (!disk || impl->MakeRoom(weight))
						break;
					// The queue and L1 share one budget. Only persisted entries may
					// be evicted, so apply backpressure until the writer frees one.
					impl->changed.wait(guard);
				}
				if (impl->Remember(key, response) && disk) {
					impl->Enqueue(impl->lru.front());
					impl->changed.notify_all();
				}
				return;
			}
		}
		const auto directory = impl->Shard(digest);
		for (;;) {
			std::unique_lock<std::mutex> local(impl->shard_mutexes[impl->ShardIndex(digest)]);
			std::unique_lock<std::mutex> guard(impl->mutex);
			if (impl->Writing(key)) {
				// A writer can be pinned while waiting for this shard. Release
				// the shard before waiting, so the active write can complete.
				local.unlock();
				impl->changed.wait(guard, [&]() { return !impl->Writing(key); });
				continue;
			}
			impl->Forget(key);
			impl->Remember(key, response);
			guard.unlock();
			impl->changed.notify_all();
			MakeDirectories(directory);
			ShardLock lock(directory);
			impl->Write(directory, directory + "/" + Hex(digest) + ".entry", key, response, lock,
			            impl->ShardIndex(digest));
			return;
		}
	} catch (...) {
		// Cache availability must never change SQL success or failure.
		impl->changed.notify_all();
	}
}

void PromptResultCache::Erase(const std::string &key) {
	try {
		if (impl->options.directory.empty() || !impl->shards || !impl->Fits(key, std::string())) {
			std::lock_guard<std::mutex> guard(impl->mutex);
			impl->Forget(key);
			return;
		}
		const auto digest = Digest(key), directory = impl->Shard(digest);
		for (;;) {
			std::unique_lock<std::mutex> local(impl->shard_mutexes[impl->ShardIndex(digest)]);
			std::unique_lock<std::mutex> guard(impl->mutex);
			if (impl->Writing(key)) {
				local.unlock();
				impl->changed.wait(guard, [&]() { return !impl->Writing(key); });
				continue;
			}
			impl->Forget(key);
			guard.unlock();
			impl->changed.notify_all();
			struct stat info;
			if (stat(directory.c_str(), &info) || !S_ISDIR(info.st_mode))
				return;
			ShardLock lock(directory);
			const auto path = directory + "/" + Hex(digest) + ".entry";
			std::string response;
			if (impl->Read(path, key, response, lock)) {
				lock.Advance();
				unlink(path.c_str());
			}
			// A later Store may already have queued its replacement. The shard
			// lock makes its publication follow this deletion; leave L1 intact.
			return;
		}
	} catch (...) {
	}
}

void PromptResultCache::Flush() {
	try {
		std::unique_lock<std::mutex> guard(impl->mutex);
		impl->changed.wait(guard, [this]() { return !impl->pending_count; });
	} catch (...) {
	}
}
}
