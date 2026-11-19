#pragma once

#include "llm/cache/prompt_result_cache.hpp"
#include <array>
#include <condition_variable>
#include <cstdint>
#include <dirent.h>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace semantic {
namespace prompt_cache_detail {

const size_t HEADER_BYTES = 56;
const size_t DIGEST_BYTES = 32;
extern const char MAGIC[8];

std::string Digest(const std::string &first, const std::string &second = std::string());
std::string Hex(const std::string &digest);
void MakeDirectories(const std::string &path);

struct Descriptor {
	int fd;
	explicit Descriptor(int value);
	~Descriptor();
	Descriptor(const Descriptor &) = delete;
	Descriptor &operator=(const Descriptor &) = delete;
};

struct ShardLock {
	Descriptor descriptor;
	explicit ShardLock(const std::string &directory);
	struct stat Stamp() const;
	struct stat Advance(const struct stat &before) const;
	void Advance() const;
};

struct CloseDirectory {
	void operator()(DIR *directory) const;
};

void Transfer(int fd, char *data, size_t size, bool writing);
void PutSize(char *target, uint64_t value);
uint64_t GetSize(const char *source);
bool EntryName(const char *name);
size_t Weight(const std::string &key, const std::string &response);
bool SameTime(const timespec &left, const timespec &right);
void TouchDescriptor(int fd);
size_t Allocated(const struct stat &info);

} // namespace prompt_cache_detail

struct PromptResultCache::Impl {
	explicit Impl(const PromptCacheOptions &value);
	~Impl();
	struct Entry {
		Entry(const std::string &key_value, const std::string &response_value, size_t weight_value,
		      const std::string &digest_value)
		    : key(key_value), response(response_value), weight(weight_value), pending(false), writing(false),
		      publishing(false), queue_previous(nullptr), queue_next(nullptr) {
			if (!digest_value.empty())
				std::copy(digest_value.begin(), digest_value.end(), digest.begin());
		}
		std::string key, response;
		std::array<char, prompt_cache_detail::DIGEST_BYTES> digest = {};
		size_t weight;
		int64_t last_touch_sec = 0;
		bool pending, writing, publishing;
		Entry *queue_previous, *queue_next;
	};
	struct ShardUsage {
		bool valid = false;
		size_t bytes = 0, entries = 0, allocation_block = 0;
		struct stat stamp = {}, generation = {};
		bool Matches(const struct stat &info, const struct stat &lock) const {
			return valid && stamp.st_dev == info.st_dev && stamp.st_ino == info.st_ino &&
			       prompt_cache_detail::SameTime(stamp.st_mtim, info.st_mtim) &&
			       prompt_cache_detail::SameTime(stamp.st_ctim, info.st_ctim) &&
			       generation.st_dev == lock.st_dev && generation.st_ino == lock.st_ino &&
			       prompt_cache_detail::SameTime(generation.st_mtim, lock.st_mtim);
		}
	};
	struct Publication {
		Impl &owner;
		const std::string &key;
		size_t shard_index;
		bool restore_pending;
		~Publication();
	};
	// Fixed metadata only: no second per-entry index competing with the L1 budget.
	// All access is protected by the corresponding in-process shard mutex.
	std::array<ShardUsage, 256> disk_usage;
	PromptCacheOptions options;
	size_t shards = 0, memory_bytes = 0;
	std::mutex mutex;
	// Fixed per-shard coordination prevents an old disk read from repopulating
	// L1 after Erase, without serializing independent shards in this process.
	std::array<std::mutex, 256> shard_mutexes;
	std::array<bool, 256> publishing = {};
	std::condition_variable changed;
	std::list<Entry> lru;
	std::unordered_map<std::string, std::list<Entry>::iterator> memory;
	size_t pending_count = 0;
	Entry *queue_head = nullptr, *queue_tail = nullptr;
	bool stopping = false;
	std::thread writer;

	bool Fits(const std::string &key, const std::string &response) const;
	void Unqueue(Entry &entry);
	void EnsureWriterLocked();
	void Enqueue(Entry &entry);
	void Forget(const std::string &key);
	bool MakeRoom(size_t weight);
	bool Remember(const std::string &key, const std::string &response);
	bool Recall(const std::string &key, std::string &response, std::string &disk_digest, bool read_only = false);
	void Touch(const std::string &digest);
	bool BeginPublication(const std::string &key, size_t shard_index);
	bool Writing(const std::string &key) const;
	size_t ShardIndex(const std::string &digest) const;
	std::string Shard(const std::string &digest) const;
	void RunWriter();
	bool Read(const std::string &path, const std::string &key, std::string &response,
	          const prompt_cache_detail::ShardLock &lock, bool read_only = false);
	bool Write(const std::string &directory, const std::string &path, const std::string &key,
	           const std::string &response, const prompt_cache_detail::ShardLock &lock, size_t shard_index);
};

} // namespace semantic
