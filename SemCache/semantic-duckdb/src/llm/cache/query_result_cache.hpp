#pragma once

// Query-owned typed results. Stable contract buckets share one bounded
// FIFO; an oversized contract disables reuse instead of sharing a false key.
#include "duckdb/common/types/value.hpp"
#include "llm/llm_client.hpp"
#include "llm/json_utils.hpp"
#include <deque>
#include <map>
#include <tuple>
#include <unordered_map>

namespace duckdb {
namespace semantic_extension {

class SemanticResultCache {
public:
	struct Result {
		Value value;
		uint64_t source_request_seq;
	};
	class Bucket {
	public:
		using Map = std::unordered_map<string, Result>;
		using iterator = Map::iterator;
		explicit Bucket(SemanticResultCache *owner = nullptr) : owner(owner) {
		}
		bool Enabled() const {
			return owner != nullptr;
		}
		iterator find(const string &key) {
			return values.find(key);
		}
		iterator end() {
			return values.end();
		}
		std::pair<iterator, bool> emplace(string key, Result result) {
			return owner ? owner->Store(*this, std::move(key), std::move(result)) : std::make_pair(values.end(), false);
		}

	private:
		friend class SemanticResultCache;
		SemanticResultCache *owner;
		Map values;
	};

	explicit SemanticResultCache(size_t max_bytes = 16777216, size_t max_entries = 32768,
	                             size_t max_entry_bytes = 262144)
	    : max_bytes(max_bytes), max_entries(max_entries), max_entry_bytes(max_entry_bytes) {
	}
	Bucket &Disabled() {
		return disabled;
	}

	Bucket &Bind(const semantic::LLMRequest &request) {
		using Json = semantic::llm_detail::Json;
		auto key = Json::array({request.model_id, request.system_text, int(request.output_schema.kind),
		                        Fields(request.output_schema.fields), request.temperature, request.max_tokens})
		               .dump();
		auto found = contexts.find(key);
		if (found != contexts.end())
			return found->second;
		const auto weight = key.size() + sizeof(Bucket) + 128;
		if (weight > max_entry_bytes || !MakeRoom(weight))
			return disabled;
		auto inserted = contexts.emplace(std::piecewise_construct, std::forward_as_tuple(std::move(key)),
		                                 std::forward_as_tuple(this));
		used_bytes += weight;
		return inserted.first->second;
	}

private:
	struct Entry {
		Bucket *bucket;
		string key;
		size_t weight;
	};
	bool MakeRoom(size_t weight) {
		while (!fifo.empty() && (weight > max_bytes - used_bytes || contexts.size() + fifo.size() >= max_entries)) {
			auto &old = fifo.front();
			old.bucket->values.erase(old.key);
			auto &values = old.bucket->values;
			if (values.empty()) {
				Bucket::Map empty;
				values.swap(empty);
			} else if (values.bucket_count() / 4 > values.size()) {
				values.rehash(values.size() * 2);
			}
			used_bytes -= old.weight;
			fifo.pop_front();
		}
		return weight <= max_bytes - used_bytes && contexts.size() + fifo.size() < max_entries;
	}
	std::pair<Bucket::iterator, bool> Store(Bucket &bucket, string key, Result result) {
		auto found = bucket.values.find(key);
		if (found != bucket.values.end())
			return {found, false};
		// Both the dictionary and FIFO retain the exact key. Counting one Value
		// per rendered character deliberately overestimates nested value storage.
		size_t weight = sizeof(Result) + sizeof(Entry) + 128;
		if (key.size() > max_entry_bytes / 2)
			return {bucket.end(), false};
		weight += key.size() * 2;
		const auto rendered = result.value.ToString();
		if (rendered.size() > max_entry_bytes / sizeof(Value))
			return {bucket.end(), false};
		const auto value_weight = sizeof(Value) * (rendered.size() + 1);
		if (weight > max_entry_bytes || value_weight > max_entry_bytes - weight)
			return {bucket.end(), false};
		weight += value_weight;
		if (weight > max_entry_bytes || !MakeRoom(weight))
			return {bucket.end(), false};
		auto inserted = bucket.values.emplace(key, std::move(result));
		fifo.push_back({&bucket, std::move(key), weight});
		used_bytes += weight;
		return inserted;
	}
	static semantic::llm_detail::Json Fields(const semantic::Schema &fields) {
		auto result = semantic::llm_detail::Json::array();
		for (const auto &field : fields)
			result.push_back({field.name, int(field.type), Fields(field.children)});
		return result;
	}
	size_t max_bytes, max_entries, max_entry_bytes, used_bytes = 0;
	std::map<string, Bucket> contexts;
	std::deque<Entry> fifo;
	Bucket disabled;
};

}
}
