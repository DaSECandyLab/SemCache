#include "runtime/runtime_progress.hpp"
#include <algorithm>
#include <cmath>

namespace semantic {

RuntimeProgress::RuntimeProgress(const Json &query, const Json &estimates, uint64_t limit) : limit(limit) {
	if (!limit || limit > 128 || !estimates.is_object())
		throw ValidationError("Invalid runtime progress budget/estimates");
	std::map<int64_t, Json> nodes;
	for (const auto &node : query.at("operators"))
		nodes.emplace(node.at("node_id").get<int64_t>(), node);
	for (const auto &op : query.at("semantic_operators")) {
		const auto id = op.at("operator_id").get<int64_t>();
		const auto key = std::to_string(id);
		double expected = 0;
		if (estimates.contains(key)) {
			if (!estimates.at(key).is_number())
				throw ValidationError("Runtime input cardinalities must be numeric");
			expected = estimates.at(key).get<double>();
		} else {
			const auto &node = nodes.at(op.at("node_id").get<int64_t>());
			const auto &children = node.at("children");
			if (!children.empty()) {
				expected = 1;
				for (const auto &child : children)
					expected *= nodes.at(child.get<int64_t>()).value("estimated_rows", 0.0);
			}
		}
		if (!std::isfinite(expected) || expected < 0)
			throw ValidationError("Runtime input cardinalities must be finite and nonnegative");
		// A zero estimate cannot make an operator disappear from the schedule.
		Entry entry;
		entry.expected = std::max(1.0, expected);
		total += entry.expected;
		entries.emplace(id, entry);
	}
	for (auto it = estimates.begin(); it != estimates.end(); ++it) {
		bool known = false;
		for (const auto &entry : entries)
			known |= it.key() == std::to_string(entry.first);
		if (!known)
			throw ValidationError("Runtime input cardinality names an unknown operator");
	}
	if (!std::isfinite(total))
		throw ValidationError("Runtime total input cardinality overflow");
}

void RuntimeProgress::Observe(int64_t id) {
	auto &entry = entries.at(id);
	if (entry.done)
		throw ValidationError("Runtime input completed after operator finalization");
	const auto before = std::min(double(entry.completed), entry.expected);
	++entry.completed;
	processed += std::min(double(entry.completed), entry.expected) - before;
}

void RuntimeProgress::Finish(int64_t id) {
	auto &entry = entries.at(id);
	if (!entry.done) {
		processed += entry.expected - std::min(double(entry.completed), entry.expected);
		entry.done = true;
	}
}

double RuntimeProgress::Fraction() const {
	return total ? std::min(1.0, std::max(0.0, processed / total)) : 1.0;
}

uint64_t RuntimeProgress::TakeDue() {
	uint64_t due = 0;
	while (next <= limit && Fraction() + 1e-12 >= double(next) / (limit + 1))
		due = next++;
	return due;
}

RuntimeProgress::Json RuntimeProgress::Snapshot(uint64_t checkpoint) const {
	Json operators = Json::array();
	for (const auto &entry : entries)
		operators.push_back({{"operator_id", entry.first},
		                     {"estimated_input_rows", entry.second.expected},
		                     {"completed_input_rows", entry.second.completed},
		                     {"done", entry.second.done}});
	return {{"index", checkpoint},
	        {"count", limit},
	        {"target", double(checkpoint) / (limit + 1)},
	        {"observed", Fraction()},
	        {"estimated_total_input_rows", total},
	        {"operators", operators},
	        {"basis", "actual completed semantic inputs weighted by initial-plan input cardinalities"}};
}

}
