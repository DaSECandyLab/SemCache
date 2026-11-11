#pragma once

// Limit different requests independently of their mapped original rows. A
// repeated prompt owns one request slot, while row/byte limits bound lookahead.
#include "operators/execution/semantic_execution.hpp"
#include <unordered_map>

namespace duckdb {
namespace semantic_extension {

class SemanticPendingWindow {
public:
	bool CanAdmit(idx_t rows, idx_t max_requests) const {
		return rows < MaxValue<idx_t>(STANDARD_VECTOR_SIZE, max_requests) && retained_bytes < MAX_BYTES &&
		       flights.size() < max_requests;
	}
	template <class Row>
	void Add(Row &row) {
		row.retained_bytes = row.pending.retained_bytes + 2048;
		for (const auto &value : row.output_values)
			row.retained_bytes += ValueBytes(value);
		retained_bytes += row.retained_bytes;
		if (row.pending.shared_flight)
			++flights[row.pending.shared_flight.get()];
	}
	template <class Row>
	void Remove(const Row &row) {
		D_ASSERT(retained_bytes >= row.retained_bytes);
		retained_bytes -= row.retained_bytes;
		if (row.pending.shared_flight) {
			auto found = flights.find(row.pending.shared_flight.get());
			D_ASSERT(found != flights.end() && found->second);
			if (!--found->second)
				flights.erase(found);
		}
	}
	void Clear() {
		flights.clear();
		retained_bytes = 0;
	}

private:
	static size_t ValueBytes(const Value &value) {
		size_t bytes = sizeof(Value);
		if (value.IsNull())
			return bytes;
		const vector<Value> *children = nullptr;
		switch (value.type().InternalType()) {
		case PhysicalType::VARCHAR:
			return bytes + StringValue::Get(value).size();
		case PhysicalType::STRUCT:
			children = &StructValue::GetChildren(value);
			break;
		case PhysicalType::LIST:
			children = &ListValue::GetChildren(value);
			break;
		case PhysicalType::ARRAY:
			children = &ArrayValue::GetChildren(value);
			break;
		default:
			break;
		}
		if (children)
			for (const auto &child : *children)
				bytes += ValueBytes(child);
		return bytes;
	}
	// This is a soft limit: at most one additional row can cross it. Accepting
	// that row ensures progress even when one input itself exceeds the budget.
	static constexpr size_t MAX_BYTES = 64 * 1024 * 1024;
	size_t retained_bytes = 0;
	std::unordered_map<const InFlightSemanticResult *, idx_t> flights;
};

}
}
