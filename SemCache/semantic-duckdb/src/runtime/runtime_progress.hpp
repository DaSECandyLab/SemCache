#pragma once

#include "llm/json_utils.hpp"
#include <map>

namespace semantic {

// Initial-plan cardinalities weight actual completed semantic inputs. This
// scheduler never uses elapsed time, HTTP counts or inference predictions.
class RuntimeProgress {
public:
	using Json = llm_detail::Json;
	RuntimeProgress(const Json &query, const Json &estimates, uint64_t limit);
	void Observe(int64_t id);
	void Finish(int64_t id);
	double Fraction() const;
	// Consume crossed checkpoints, including checks that cannot run while the
	// previous search is busy. Never catch up with a burst at the same progress.
	uint64_t TakeDue();
	uint64_t Limit() const {
		return limit;
	}
	Json Snapshot(uint64_t checkpoint) const;

private:
	struct Entry {
		double expected = 1;
		uint64_t completed = 0;
		bool done = false;
	};
	std::map<int64_t, Entry> entries;
	uint64_t limit, next = 1;
	double total = 0, processed = 0;
};

}
