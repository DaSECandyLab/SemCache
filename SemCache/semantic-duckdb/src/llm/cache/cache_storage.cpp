#include "llm/cache/prompt_cache_internal.hpp"

#include <openssl/evp.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <vector>

namespace semantic {
namespace prompt_cache_detail {

const char MAGIC[8] = {'S', 'E', 'M', 'P', 'R', 'C', '0', '1'};

std::string Digest(const std::string &first, const std::string &second) {
	std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
	unsigned char bytes[EVP_MAX_MD_SIZE];
	unsigned int count = 0;
	if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
	    EVP_DigestUpdate(context.get(), first.data(), first.size()) != 1 ||
	    EVP_DigestUpdate(context.get(), second.data(), second.size()) != 1 ||
	    EVP_DigestFinal_ex(context.get(), bytes, &count) != 1 || count != 32)
		throw std::runtime_error("prompt cache digest failure");
	return std::string(reinterpret_cast<const char *>(bytes), count);
}
std::string Hex(const std::string &digest) {
	static const char hex[] = "0123456789abcdef";
	std::string result;
	result.reserve(digest.size() * 2);
	for (unsigned char byte : digest) {
		result += hex[byte >> 4];
		result += hex[byte & 15];
	}
	return result;
}

void MakeDirectories(const std::string &path) {
	for (size_t index = 1; index <= path.size(); ++index) {
		if (index != path.size() && path[index] != '/')
			continue;
		const auto part = path.substr(0, index);
		if (!part.empty() && mkdir(part.c_str(), 0700) != 0 && errno != EEXIST)
			throw std::runtime_error("prompt cache directory unavailable");
	}
}


Descriptor::Descriptor(int value) : fd(value) {
	if (fd < 0)
		throw std::runtime_error("prompt cache file unavailable");
}
Descriptor::~Descriptor() {
	close(fd);
}

ShardLock::ShardLock(const std::string &directory)
    : descriptor(open((directory + "/.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600)) {
	while (flock(descriptor.fd, LOCK_EX) != 0) {
		if (errno != EINTR)
			throw std::runtime_error("prompt cache lock unavailable");
	}
}
struct stat ShardLock::Stamp() const {
	struct stat info;
	if (fstat(descriptor.fd, &info))
		throw std::runtime_error("prompt cache generation unavailable");
	return info;
}
struct stat ShardLock::Advance(const struct stat &before) const {
		timespec times[2] = {};
		times[0].tv_nsec = UTIME_OMIT;
		times[1] = before.st_mtim;
		if (++times[1].tv_nsec == 1000000000) {
			if (times[1].tv_sec == std::numeric_limits<time_t>::max())
				throw std::runtime_error("prompt cache generation exhausted");
			++times[1].tv_sec;
			times[1].tv_nsec = 0;
		}
		// Directory timestamps may be coalesced within one filesystem clock tick.
		// An explicit generation on the existing empty lock inode avoids both a
		// stale cross-process usage snapshot and a journal consuming disk blocks.
		if (futimens(descriptor.fd, times))
			throw std::runtime_error("prompt cache generation update failed");
		const auto after = Stamp();
		if (before.st_mtim.tv_sec == after.st_mtim.tv_sec && before.st_mtim.tv_nsec == after.st_mtim.tv_nsec)
			throw std::runtime_error("prompt cache generation precision unavailable");
		return after;
	}

void ShardLock::Advance() const {
	Advance(Stamp());
}
void CloseDirectory::operator()(DIR *directory) const {
	closedir(directory);
}
void Transfer(int fd, char *data, size_t size, bool writing) {
	while (size) {
		const auto count = writing ? write(fd, data, size) : read(fd, data, size);
		if (count < 0 && errno == EINTR)
			continue;
		if (count <= 0)
			throw std::runtime_error("prompt cache incomplete IO");
		data += count;
		size -= static_cast<size_t>(count);
	}
}

void PutSize(char *target, uint64_t value) {
	for (size_t index = 0; index < 8; ++index)
		target[index] = static_cast<char>((value >> (index * 8)) & 255);
}

uint64_t GetSize(const char *source) {
	uint64_t value = 0;
	for (size_t index = 0; index < 8; ++index)
		value |= uint64_t(static_cast<unsigned char>(source[index])) << (index * 8);
	return value;
}

bool EntryName(const char *name) {
	if (std::strlen(name) != 70 || std::strcmp(name + 64, ".entry") != 0)
		return false;
	for (size_t index = 0; index < 64; ++index)
		if (!((name[index] >= '0' && name[index] <= '9') || (name[index] >= 'a' && name[index] <= 'f')))
			return false;
	return true;
}

size_t Weight(const std::string &key, const std::string &response) {
	const size_t overhead =
	    sizeof(std::string) * 4 + sizeof(void *) * 14 + sizeof(size_t) + sizeof(bool) * 3 + DIGEST_BYTES;
	// Both the LRU node and the hash index own a copy of the complete key.
	const size_t available = std::numeric_limits<size_t>::max() - overhead;
	if (key.size() > available / 2 || response.size() > available - key.size() * 2)
		return std::numeric_limits<size_t>::max();
	return overhead + key.size() * 2 + response.size();
}

bool SameTime(const timespec &left, const timespec &right) {
	return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

void TouchDescriptor(int fd) {
	// atime may be suppressed by mount options; mtime is our explicit LRU clock.
	// Keep nanoseconds and refresh every read, including reads less than 30s apart.
	timespec times[2] = {};
	times[0].tv_nsec = UTIME_OMIT;
	times[1].tv_nsec = UTIME_NOW;
	futimens(fd, times);
}

size_t Allocated(const struct stat &info) {
	if (info.st_blocks < 0 || uint64_t(info.st_blocks) > std::numeric_limits<size_t>::max() / 512)
		return std::numeric_limits<size_t>::max();
	return static_cast<size_t>(info.st_blocks) * 512;
}

} // namespace prompt_cache_detail

using namespace prompt_cache_detail;

PromptResultCache::Impl::Impl(const PromptCacheOptions &value) : options(value) {
	if (!options.directory.empty()) {
		while (options.directory.size() > 1 && options.directory.back() == '/')
			options.directory.pop_back();
	}
	if (options.max_disk_bytes && options.max_entries && options.max_entry_bytes) {
		shards =
		    std::min(size_t(256), std::min(options.max_entries,
		                                   std::max(size_t(1), options.max_disk_bytes / options.max_entry_bytes)));
	}
}

PromptResultCache::Impl::~Impl() {
	{
		std::lock_guard<std::mutex> guard(mutex);
		stopping = true;
	}
	changed.notify_all();
	if (writer.joinable())
		writer.join();
}


PromptResultCache::Impl::Publication::~Publication() {
	{
		std::lock_guard<std::mutex> guard(owner.mutex);
		auto found = owner.memory.find(key);
		if (found != owner.memory.end() && found->second->publishing) {
			found->second->publishing = false;
			// A failed transaction/disk miss retains the original local
			// entry and must not silently discard its queued persistence.
			if (restore_pending)
				owner.Enqueue(*found->second);
		}
		owner.publishing[shard_index] = false;
	}
	owner.changed.notify_all();
}


bool PromptResultCache::Impl::Fits(const std::string &key, const std::string &response) const {
	return options.max_entries && options.max_entry_bytes >= HEADER_BYTES &&
	       key.size() <= options.max_entry_bytes - HEADER_BYTES &&
	       response.size() <= options.max_entry_bytes - HEADER_BYTES - key.size();
}

void PromptResultCache::Impl::Unqueue(Entry &entry) {
	if (entry.queue_previous)
		entry.queue_previous->queue_next = entry.queue_next;
	else
		queue_head = entry.queue_next;
	if (entry.queue_next)
		entry.queue_next->queue_previous = entry.queue_previous;
	else
		queue_tail = entry.queue_previous;
	entry.queue_previous = entry.queue_next = nullptr;
}

void PromptResultCache::Impl::EnsureWriterLocked() {
	if (!stopping && !writer.joinable() && !options.directory.empty() && shards && options.max_memory_bytes) {
		try {
			writer = std::thread([this]() { RunWriter(); });
		} catch (...) {
		}
	}
}

void PromptResultCache::Impl::Enqueue(Entry &entry) {
	EnsureWriterLocked();
	entry.pending = true;
	entry.queue_previous = queue_tail;
	if (queue_tail)
		queue_tail->queue_next = &entry;
	else
		queue_head = &entry;
	queue_tail = &entry;
	++pending_count;
}

void PromptResultCache::Impl::Forget(const std::string &key) {
	auto found = memory.find(key);
	if (found == memory.end())
		return;
	if (found->second->pending) {
		Unqueue(*found->second);
		--pending_count;
	}
	memory_bytes -= found->second->weight;
	lru.erase(found->second);
	memory.erase(found);
}

bool PromptResultCache::Impl::MakeRoom(size_t weight) {
	while (memory_bytes > options.max_memory_bytes - weight || memory.size() >= options.max_entries) {
		auto victim = lru.end();
		while (victim != lru.begin()) {
			--victim;
			if (!victim->pending && !victim->publishing)
				break;
		}
		if (victim == lru.end() || victim->pending || victim->publishing)
			return false;
		Forget(victim->key);
	}
	return true;
}

bool PromptResultCache::Impl::Remember(const std::string &key, const std::string &response) {
	const auto weight = Weight(key, response);
	if (!Fits(key, response) || !options.max_memory_bytes || weight > options.max_memory_bytes || !MakeRoom(weight))
		return false;
	lru.emplace_front(key, response, weight, options.directory.empty() || !shards ? std::string() : Digest(key));
	try {
		memory.emplace(lru.front().key, lru.begin());
	} catch (...) {
		lru.pop_front();
		throw;
	}
	memory_bytes += weight;
	return true;
}

bool PromptResultCache::Impl::Recall(const std::string &key, std::string &response, std::string &disk_digest,
                                     bool read_only) {
	std::lock_guard<std::mutex> guard(mutex);
	auto found = memory.find(key);
	if (found == memory.end())
		return false;
	response = found->second->response;
	lru.splice(lru.begin(), lru, found->second);
	// A queued write will publish a fresh timestamp by itself. Keep its
	// precomputed locator inline and budgeted to avoid hashing a long key on
	// every L1 hit; throttle disk mtime touches to at most once per 30s.
	if (!read_only && !found->second->pending && !options.directory.empty() && shards) {
		const auto now_sec =
		    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch())
		        .count();
		if (found->second->last_touch_sec == 0 || now_sec - found->second->last_touch_sec >= 30) {
			found->second->last_touch_sec = now_sec;
			disk_digest.assign(found->second->digest.data(), DIGEST_BYTES);
		}
	}
	return true;
}

void PromptResultCache::Impl::Touch(const std::string &digest) {
	if (options.directory.empty() || !shards)
		return;
	try {
		const auto path = Shard(digest) + "/" + Hex(digest) + ".entry";
		const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
		if (fd < 0)
			return;
		Descriptor descriptor(fd);
		// No shard flock: an L1 hit must not wait for persistence. Touching an
		// inode concurrently replaced/evicted is harmless and never recreates it.
		TouchDescriptor(fd);
	} catch (...) {
	}
}

bool PromptResultCache::Impl::BeginPublication(const std::string &key, size_t shard_index) {
	bool restore_pending = false;
	auto found = memory.find(key);
	if (found != memory.end()) {
		auto &entry = *found->second;
		restore_pending = entry.pending;
		if (entry.pending) {
			Unqueue(entry);
			entry.pending = false;
			--pending_count;
		}
		entry.publishing = true;
	}
	publishing[shard_index] = true;
	return restore_pending;
}

bool PromptResultCache::Impl::Writing(const std::string &key) const {
	auto found = memory.find(key);
	return found != memory.end() && found->second->writing;
}

size_t PromptResultCache::Impl::ShardIndex(const std::string &digest) const {
	return static_cast<unsigned char>(digest[0]) % shards;
}

std::string PromptResultCache::Impl::Shard(const std::string &digest) const {
	return options.directory + "/" + std::to_string(ShardIndex(digest));
}

void PromptResultCache::Impl::RunWriter() {
	for (;;) {
		Entry *entry = nullptr;
		{
			std::unique_lock<std::mutex> guard(mutex);
			changed.wait(guard, [this]() { return stopping || pending_count; });
			if (!pending_count && stopping)
				return;
			// Intrusive links keep queue operations O(1), with no second
			// payload or queue-node allocation beyond the shared L1 budget.
			entry = queue_head;
			Unqueue(*entry);
			entry->writing = true;
		}
		try {
			const std::string digest(entry->digest.data(), DIGEST_BYTES);
			const auto directory = Shard(digest);
			std::lock_guard<std::mutex> local(shard_mutexes[ShardIndex(digest)]);
			MakeDirectories(directory);
			ShardLock lock(directory);
			Write(directory, directory + "/" + Hex(digest) + ".entry", entry->key, entry->response, lock,
			      ShardIndex(digest));
		} catch (...) {
			// Disk and worker failures must never escape to a SQL caller.
		}
		{
			std::lock_guard<std::mutex> guard(mutex);
			entry->pending = entry->writing = false;
			--pending_count;
		}
		changed.notify_all();
	}
}

bool PromptResultCache::Impl::Read(const std::string &path, const std::string &key, std::string &response,
                                   const ShardLock &lock, bool read_only) {
	const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	if (fd < 0)
		return false;
	Descriptor descriptor(fd);
	struct stat info;
	char header[HEADER_BYTES];
	try {
		if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size < static_cast<off_t>(HEADER_BYTES) ||
		    uint64_t(info.st_size) > options.max_entry_bytes)
			throw std::runtime_error("prompt cache invalid length");
		Transfer(fd, header, HEADER_BYTES, false);
		const auto key_bytes = GetSize(header + 8), response_bytes = GetSize(header + 16);
		if (std::memcmp(header, MAGIC, 8) || key_bytes > options.max_entry_bytes - HEADER_BYTES ||
		    response_bytes > options.max_entry_bytes - HEADER_BYTES - key_bytes ||
		    key_bytes + response_bytes + HEADER_BYTES != uint64_t(info.st_size))
			throw std::runtime_error("prompt cache invalid header");
		// A hash collision is a miss, never a result for another binary key.
		if (key_bytes != key.size())
			return false;
		std::string stored_key(static_cast<size_t>(key_bytes), '\0');
		Transfer(fd, &stored_key[0], stored_key.size(), false);
		if (stored_key != key)
			return false;
		std::string payload(static_cast<size_t>(response_bytes), '\0');
		Transfer(fd, &payload[0], payload.size(), false);
		response = std::move(payload);
		if (!read_only)
			TouchDescriptor(fd);
		return true;
	} catch (...) {
		if (!read_only) {
			lock.Advance();
			unlink(path.c_str());
		}
		return false;
	}
}

bool PromptResultCache::Impl::Write(const std::string &directory, const std::string &path, const std::string &key,
                                    const std::string &response, const ShardLock &lock, size_t shard_index) {
	auto &usage = disk_usage[shard_index];
	const size_t budget = options.max_disk_bytes / shards, capacity = options.max_entries / shards;
	const size_t bytes = HEADER_BYTES + key.size() + response.size();
	struct stat directory_info;
	if (stat(directory.c_str(), &directory_info))
		return false;
	// The explicit lock generation catches even mutations in the same clock
	// tick. Entry read touches leave it unchanged, keeping this O(1) fast path.
	const auto previous_generation = lock.Stamp();
	const bool unchanged = usage.Matches(directory_info, previous_generation);
	usage.valid = false;
	const auto generation = lock.Advance(previous_generation);
	usage.valid = unchanged;
	if (!usage.allocation_block || usage.stamp.st_dev != directory_info.st_dev ||
	    usage.stamp.st_ino != directory_info.st_ino) {
		struct statvfs filesystem;
		if (statvfs(directory.c_str(), &filesystem))
			return false;
		const auto block = std::max(filesystem.f_bsize, filesystem.f_frsize);
		if (!block || uint64_t(block) > std::numeric_limits<size_t>::max())
			return false;
		usage.allocation_block = static_cast<size_t>(block);
	}
	const auto block = usage.allocation_block;
	if (bytes > std::numeric_limits<size_t>::max() - (block - 1))
		return false;
	const size_t reservation = ((bytes + block - 1) / block) * block;
	if (reservation > budget || !capacity)
		return false;
	const auto temporary = directory + "/.tmp";
	if (unlink(temporary.c_str()) && errno != ENOENT)
		return false;
	// Remove the replacement before reserving its temporary file, so the
	// committed entries plus the single temporary never exceed the budget.
	struct stat replaced;
	if (lstat(path.c_str(), &replaced) == 0) {
		if (unlink(path.c_str())) {
			usage.valid = false;
			return false;
		}
		const auto allocated = Allocated(replaced);
		if (usage.valid && S_ISREG(replaced.st_mode) && usage.entries && allocated <= usage.bytes) {
			usage.bytes -= allocated;
			--usage.entries;
		} else {
			usage.valid = false;
		}
	} else if (errno != ENOENT) {
		usage.valid = false;
		return false;
	}
	if (!usage.valid || usage.entries >= capacity || usage.bytes > budget - reservation) {
		// Scan only after another process changed the shard, on first use, or
		// when eviction is actually required. Under-capacity stores are O(1).
		usage.valid = false;
		struct File {
			std::string path;
			size_t bytes;
			timespec modified;
		};
		std::vector<File> files;
		size_t total = 0;
		std::unique_ptr<DIR, CloseDirectory> entries(opendir(directory.c_str()));
		if (!entries)
			return false;
		for (;;) {
			errno = 0;
			auto entry = readdir(entries.get());
			if (!entry) {
				if (errno)
					return false;
				break;
			}
			if (!EntryName(entry->d_name))
				continue;
			const auto file = directory + "/" + entry->d_name;
			struct stat info;
			if (lstat(file.c_str(), &info)) {
				if (errno == ENOENT)
					continue;
				return false;
			}
			const auto allocated = Allocated(info);
			if (!S_ISREG(info.st_mode) || info.st_size < static_cast<off_t>(HEADER_BYTES) ||
			    uint64_t(info.st_size) > options.max_entry_bytes || allocated > budget) {
				if (unlink(file.c_str()))
					return false;
				continue;
			}
			if (allocated > std::numeric_limits<size_t>::max() - total)
				return false;
			files.push_back({file, allocated, info.st_mtim});
			total += allocated;
		}
		// Heap construction is linear, then only actual victims cost log(n).
		// Compare nanoseconds so recent reads/writes in the same second count.
		const auto newer = [](const File &left, const File &right) {
			if (left.modified.tv_sec != right.modified.tv_sec)
				return left.modified.tv_sec > right.modified.tv_sec;
			if (left.modified.tv_nsec != right.modified.tv_nsec)
				return left.modified.tv_nsec > right.modified.tv_nsec;
			return left.path > right.path;
		};
		std::make_heap(files.begin(), files.end(), newer);
		while (!files.empty() && (files.size() >= capacity || total > budget - reservation)) {
			std::pop_heap(files.begin(), files.end(), newer);
			const auto &victim = files.back();
			if (unlink(victim.path.c_str()))
				return false;
			total -= victim.bytes;
			files.pop_back();
		}
		usage.bytes = total;
		usage.entries = files.size();
	}
	usage.valid = false; // A partial write must force reconciliation next time.
	Descriptor descriptor(open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
	char header[HEADER_BYTES];
	std::memcpy(header, MAGIC, 8);
	PutSize(header + 8, key.size());
	PutSize(header + 16, response.size());
	// Keep the fixed header layout; the trailing metadata bytes are unused.
	std::memset(header + 24, 0, DIGEST_BYTES);
	try {
		Transfer(descriptor.fd, header, HEADER_BYTES, true);
		Transfer(descriptor.fd, const_cast<char *>(key.data()), key.size(), true);
		Transfer(descriptor.fd, const_cast<char *>(response.data()), response.size(), true);
		struct stat published;
		if (fstat(descriptor.fd, &published) || Allocated(published) > reservation)
			throw std::runtime_error("prompt cache allocation exceeds reservation");
		if (rename(temporary.c_str(), path.c_str()))
			throw std::runtime_error("prompt cache publish failed");
		usage.bytes += Allocated(published);
		++usage.entries;
		usage.generation = generation;
		usage.valid = stat(directory.c_str(), &usage.stamp) == 0;
		return true;
	} catch (...) {
		unlink(temporary.c_str());
		throw;
	}
}


} // namespace semantic
