"""Compile a query once and dispatch its graph to the selected native backend."""

from pathlib import Path

from experiments.core.io import save
from experiments.system_comparison.backends.abacus.graph import Abacus
from experiments.system_comparison.backends.lotus.graph import Lotus
from experiments.system_comparison.backends.nirvana.graph import Nirvana
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.runtime.native_graph import Compiler, describe


def run(system, sql, tables, job):
    compiler = Compiler(tables)
    with stage("planning", "compile source SQL to native graph"):
        relation = compiler.compile(sql)
        save(Path(job["output"]) / "logical-graph.json", describe(relation.node))
    backend = {"abacus": Abacus, "nirvana": Nirvana, "lotus": Lotus}[system](job)
    output = backend.execute(relation.node)
    output = output[[internal for _, internal in relation.columns]].copy()
    output.columns = [label for label, _ in relation.columns]

    def constrained_joins(node):
        return int(
            node.options.get("candidate_relation") == "ordinary_inner_join"
        ) + sum(constrained_joins(child) for child in node.children)

    constrained = constrained_joins(relation.node)
    contract = {
        "system": system,
        "single_semantic_stages_forced": False,
        "binary_semantic_joins_preserved": constrained == 0,
        "equality_constrained_semantic_joins_as_pair_filters": constrained,
        "semantic_join_candidates": "ordinary equality predicates precede semantic evaluation",
        "explicit_sql_context_barriers": True,
        "native_optimizer_scope": "complete graph between source SQL DISTINCT/ORDER/LIMIT boundaries",
        "whole_query_native_sql_engine": False,
        "cte_cache_hits": compiler.cte_cache_hits,
        "cte_definitions_compiled": len(compiler.cte_cache),
    }
    save(Path(job["output"]) / "native-graph-contract.json", contract)
    return output
