"""Bound native training inputs using ordinary SQL keys, never model labels.

The complete production Dataset is unchanged. The optional train_dataset is a
small key-aligned validation-input pool accepted by Abacus's public API. This
avoids materialising an unbounded cross product in the sentinel's index lists.
"""

from __future__ import annotations

import math

from experiments.core.io import save
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.runtime.native_graph import Scalar, q


class EmptyCandidates(ValueError):
    pass


def training_set(dataset, job):
    import duckdb
    import palimpzest as pz

    roots = dataset._get_root_datasets()
    limit = int(job.get("abacus_optimizer", {}).get("max_sampling_index_space", 500000))
    if len(roots) == 1 or math.prod(len(root) for root in roots.values()) <= limit:
        return None
    identifiers = {key: f"sampleRoot{i}" for i, key in enumerate(roots)}
    definitions = []
    cache = {}
    with (
        stage("sampling", "prepare key-aligned native Abacus training inputs"),
        duckdb.connect(config={"threads": 1}) as con,
    ):
        for index, (key, root) in enumerate(roots.items()):
            if not hasattr(root, "vals"):
                raise ValueError(
                    "Native sampling pool needs an indexable in-memory root"
                )
            con.register(f"frame{index}", root.vals)
            con.execute(
                f"CREATE TABLE {q('source' + str(index))} AS SELECT row_number() OVER ()-1 AS {q(identifiers[key])},* FROM {q('frame' + str(index))}"
            )

        def visit(current):
            if id(current) in cache:
                return cache[id(current)]
            output = list(current.schema.model_fields)
            if current.is_root:
                index = list(roots).index(current.id)
                source = "SELECT * FROM " + q("source" + str(index))
                ids = [identifiers[current.id]]
            else:
                parents = [visit(child) for child in current._sources]
                op = current._operator
                ids = [name for _, keys in parents for name in keys]
                if len(parents) == 2:
                    left, right = current._sources
                    selection = ",".join(
                        ("l." if name in left.schema.model_fields else "r.")
                        + q(name)
                        + " AS "
                        + q(name)
                        for name in output
                    )
                    selection += "," + ",".join(
                        ("l." if name in parents[0][1] else "r.")
                        + q(name)
                        + " AS "
                        + q(name)
                        for name in ids
                    )
                    keys = getattr(op, "on", None) or []
                    on = (
                        " AND ".join("l." + q(key) + "=r." + q(key) for key in keys)
                        or "TRUE"
                    )
                    source = f"SELECT {selection} FROM {q(parents[0][0])} l JOIN {q(parents[1][0])} r ON {on}"
                else:
                    parent = parents[0][0]
                    udf = getattr(op, "udf", None)
                    filter_fn = getattr(getattr(op, "filter", None), "filter_fn", None)
                    filter_obj = getattr(op, "filter_obj", None)
                    filter_fn = filter_fn or getattr(filter_obj, "filter_fn", None)
                    if isinstance(udf, Scalar):
                        expressions = ",".join(
                            text + " AS " + q(name)
                            for name, text in udf.expressions.items()
                        )
                        source = f"SELECT *,{expressions} FROM {q(parent)}"
                    elif filter_fn is not None and hasattr(filter_fn, "scalar"):
                        expression = filter_fn.scalar.expressions["keep"]
                        source = f"SELECT * FROM {q(parent)} WHERE {expression}"
                    else:
                        # Semantic predicates are deliberately TRUE for input
                        # sampling. No prediction or final reference is read.
                        projection = ",".join(q(name) for name in output + ids)
                        source = f"SELECT {projection} FROM {q(parent)}"
            name = f"node{len(definitions)}"
            definitions.append((name, source))
            cache[id(current)] = (name, ids)
            return name, ids

        final, ids = visit(dataset)
        sql = "WITH " + ",".join(
            q(name) + " AS (" + body + ")" for name, body in definitions
        )
        sql += (
            " SELECT "
            + ",".join(q(identifiers[key]) for key in roots)
            + " FROM "
            + q(final)
            + " LIMIT 8"
        )
        witnesses = con.execute(sql).fetchall()
        if not witnesses:
            raise EmptyCandidates(
                "Ordinary SQL conditions have no possible output even with every semantic predicate TRUE"
            )
        while (
            math.prod(len({row[i] for row in witnesses}) for i in range(len(roots)))
            > limit
        ):
            witnesses = witnesses[:-1]
        pools = {}
        evidence = {
            "method": "key-aligned ordinary-SQL witness inputs; semantic predicates TRUE",
            "model_labels_read": False,
            "production_inputs_cropped": False,
            "roots": [],
        }
        for index, (key, root) in enumerate(roots.items()):
            indices = sorted({int(row[index]) for row in witnesses})
            pool = root.vals.iloc[indices].copy().reset_index(drop=True)
            pools[key] = pz.MemoryDataset(id=key, vals=pool, schema=root.schema)
            evidence["roots"].append(
                {
                    "id": key,
                    "full_rows": len(root),
                    "training_indices": indices,
                    "training_rows": len(pool),
                }
            )
        evidence["sampling_index_space"] = math.prod(
            len(pool) for pool in pools.values()
        )
        save(
            __import__("pathlib").Path(job["output"]) / "sampling-inputs.json", evidence
        )
        return pools
