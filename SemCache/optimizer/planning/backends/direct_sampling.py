"""Draw final operator cohorts from base SQL, without a reservoir of spares."""

import math

from planning.state import digest


def load_direct_sources(problem, config, load):
    semantic_rows = config.get("operator_sample_rows", 20)
    relational_rows = config.get("relational_sample_rows", 200)
    limits = sorted({semantic_rows, relational_rows})
    sources, populations, cohorts = {}, {}, {}
    for relation in problem.query["relations"]:
        sql = relation.get("source_sql")
        if not sql:
            raise ValueError("Direct sampling requires bound source_sql")
        population = relation.get("estimated_rows")
        if (
            type(population) not in (int, float)
            or not math.isfinite(population)
            or population < 0
        ):
            raise ValueError("DuckDB source cardinality must be finite and nonnegative")
        rid = relation["node_id"]
        cohorts[str(rid)] = {}
        registry = {}
        for limit in limits:
            seed = int(digest([config.get("seed", 42), rid, limit])[:8], 16) % (2**31)
            # Both sizes are independent draws from the table. In particular,
            # semantic inputs are never selected from the larger SQL cohort.
            sampled = load(
                rid,
                f"SELECT * FROM ({sql}) USING SAMPLE reservoir({limit} ROWS) "
                f"REPEATABLE ({seed}) ORDER BY hash(__sample_id, {seed})",
            )
            cohorts[str(rid)][str(limit)] = sampled
            for row in sampled:
                identity = row["id"] if isinstance(row, dict) else row.identity
                registry[identity] = row
        sources[rid] = list(registry.values())
        populations[rid] = max(population, len(registry))
    for rid, prior in problem.profile.get("cardinality_only_sources", {}).items():
        node_id = next(nid for nid in problem.logical if str(nid) == rid)
        sources[node_id] = []
        populations[node_id] = prior["estimated_rows"]
        cohorts[str(node_id)] = {str(limit): [] for limit in limits}
    return sources, populations, cohorts
