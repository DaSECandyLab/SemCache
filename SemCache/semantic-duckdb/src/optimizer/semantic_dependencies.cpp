#include "optimizer/semantic_dependencies.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include <functional>
#include <map>
#include <set>

namespace duckdb {
namespace semantic_extension {
using Json = semantic::llm_detail::Json;
namespace {
int64_t Id(const Json &value) {
	return value.get<int64_t>();
}
}

// Export generated-field dependencies inside reorder regions, and genuine
// data-flow boundaries between regions. Traversal order between independent
// branches is not a dependency.
Json SemanticDependencies(const Json &query) {
	std::map<int64_t, const Json *> nodes;
	std::map<int64_t, int64_t> semantic_nodes, cte_sources;
	std::map<int64_t, string> regions;
	std::map<string, std::vector<int64_t>> region_bases;
	for (auto &node : query["operators"]) {
		auto id = Id(node["node_id"]);
		nodes.emplace(id, &node);
		if (node.contains("semantic_operator_id"))
			semantic_nodes.emplace(Id(node["semantic_operator_id"]), id);
		if (node["type"] == "CTE" && node.contains("cte_index") && !node["children"].empty())
			cte_sources.emplace(Id(node["cte_index"]), Id(node["children"][0]));
	}
	for (idx_t i = 0; i < query["order_groups"].size(); ++i)
		for (auto &op : query["order_groups"][i])
			regions[Id(op)] = "group:" + std::to_string(i);
	for (const auto *kind : {"tree_regions", "predicate_regions"})
		if (query.contains(kind))
			for (auto &region : query[kind]) {
				auto key = string(kind) + ":" + region["region_id"].dump();
				region_bases[key] = region.contains("base_nodes") ? region["base_nodes"].get<std::vector<int64_t>>()
				                                                  : std::vector<int64_t> {Id(region["base_node_id"])};
				for (auto &id : region["nodes"]) {
					auto &node = *nodes.at(Id(id));
					if (node.contains("semantic_operator_id"))
						regions[Id(node["semantic_operator_id"])] = key;
				}
			}
	std::map<int64_t, std::set<int64_t>> upstream;
	std::set<int64_t> visiting;
	std::function<const std::set<int64_t> &(int64_t)> collect = [&](int64_t id) -> const std::set<int64_t> & {
		auto cached = upstream.find(id);
		if (cached != upstream.end())
			return cached->second;
		if (!visiting.insert(id).second)
			throw BinderException("Cyclic semantic input dependencies");
		auto &node = *nodes.at(id);
		std::set<int64_t> result;
		auto children = node["children"].get<std::vector<int64_t>>();
		if (node["type"] == "CTE_SCAN" && node.contains("cte_index")) {
			auto source = cte_sources.find(Id(node["cte_index"]));
			if (source != cte_sources.end())
				children.push_back(source->second);
		}
		for (auto child : children) {
			auto &values = collect(child);
			result.insert(values.begin(), values.end());
		}
		if (node.contains("semantic_operator_id"))
			result.insert(Id(node["semantic_operator_id"]));
		visiting.erase(id);
		return upstream.emplace(id, std::move(result)).first->second;
	};
	std::map<std::pair<string, string>, int64_t> producers;
	for (auto &op : semantic_nodes) {
		auto &node = *nodes.at(op.second);
		std::set<string> inputs;
		for (auto &side : node["input_schema"])
			for (auto &field : side)
				inputs.insert(field["binding"].dump());
		for (auto &field : node["output_schema"])
			if (!inputs.count(field["binding"].dump()))
				producers.emplace(std::make_pair(regions.at(op.first), field["binding"].dump()), op.first);
	}
	std::map<std::pair<int64_t, int64_t>, string> edges;
	for (auto &after : semantic_nodes) {
		auto &node = *nodes.at(after.second);
		std::set<string> required;
		for (auto &field : node["required_columns"])
			required.insert(field.dump());
		auto bases = region_bases.find(regions.at(after.first));
		std::vector<int64_t> inputs;
		if (bases == region_bases.end())
			inputs.push_back(after.second);
		else
			for (auto base : bases->second) {
				bool needed = bases->second.size() == 1;
				for (auto &field : (*nodes.at(base))["output_schema"])
					needed |= required.count(field["binding"].dump()) != 0;
				if (needed)
					inputs.push_back(base);
			}
		// A filter placed on one side of a movable join need not wait for
		// semantic work on the other side. Its bound input fields select bases.
		for (auto input : inputs)
			for (auto before : collect(input))
				if (before != after.first && regions.at(before) != regions.at(after.first))
					edges[{before, after.first}] = "relational_boundary";
		for (auto &field : node["required_columns"]) {
			auto producer = producers.find({regions.at(after.first), field.dump()});
			if (producer != producers.end() && producer->second != after.first)
				edges[{producer->second, after.first}] = "generated_field";
		}
	}
	Json result = Json::array();
	for (auto &edge : edges)
		result.push_back({{"before", edge.first.first}, {"after", edge.first.second}, {"reason", edge.second}});
	return result;
}
}
}
