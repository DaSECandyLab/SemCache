"""Select native Sema switches from the exposed filter groups, not query IDs."""

import json
import math

from sqlglot import exp


def transport_binary(source, target, timeout):
    """Extend two libcurl deadlines in a verified copy of the paper binary.

    The released artifact exposes no SQL setting for these deadlines. Its
    ConfigureReusableOptions uses CURLOPT_TIMEOUT=60 and LOW_SPEED_TIME=15.
    Only their immediate operands change; native query operators stay intact.
    """
    from experiments.core.io import digest, save

    expected = "15a534667a668a152524a8756fe5d66c0fa9c9822e73d26029e1a71693a56ec3"
    seconds = math.ceil(float(timeout))
    if not 60 <= seconds <= 2147483647:
        raise ValueError(
            "Sema transport timeout must be between 60 and 2147483647 seconds"
        )
    if digest(source) != expected:
        raise ValueError(
            "Sema transport compatibility requires the verified paper binary"
        )
    content = bytearray(source.read_bytes())
    patches = []
    for offset, old, option in (
        (0xAB47C8, 60, "CURLOPT_TIMEOUT"),
        (0xAB480F, 15, "CURLOPT_LOW_SPEED_TIME"),
    ):
        # mov edx,<seconds>; xor eax,eax OR mov esi,<option> follows.
        original = b"\xba" + old.to_bytes(4, "little")
        if content[offset : offset + 5] != original:
            raise ValueError(
                "Sema transport instruction does not match the verified artifact"
            )
        content[offset + 1 : offset + 5] = seconds.to_bytes(4, "little")
        patches.append(
            {"option": option, "offset": offset + 1, "old": old, "new": seconds}
        )
    target.write_bytes(content)
    target.chmod(0o700)
    save(
        target.parent / "transport-compatibility.json",
        {
            "source_binary": str(source),
            "source_sha256": expected,
            "runtime_binary": str(target),
            "runtime_sha256": digest(target),
            "patches": patches,
            "scope": "HTTP deadlines only; query operators and concurrency unchanged",
            "whole_query_deadline_seconds": seconds,
        },
    )
    return target


def _chained_scalar_map_filter(tree):
    """Find a bare semantic map whose generated alias feeds an outer filter."""
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        source = select.args.get("from_")
        if where is None or source is None or not isinstance(source.this, exp.Subquery):
            continue
        inner = source.this.this
        if not isinstance(inner, exp.Select):
            continue
        generated = {
            projection.alias.casefold()
            for projection in inner.expressions
            if isinstance(projection, exp.Alias)
            and isinstance(projection.this, exp.Anonymous)
            and projection.this.name.casefold()
            in {"semantic_filter", "semantic_project"}
        }
        for predicate in where.find_all(exp.Anonymous):
            if predicate.name.casefold() != "semantic_filter":
                continue
            if any(
                column.name.casefold() in generated
                for column in predicate.find_all(exp.Column)
            ):
                return True
    return False


def options(sql):
    from experiments.validation.audits.source import native_tree

    tree = native_tree(sql, "sema")
    groups = []
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        if where is None:
            continue
        count = sum(
            isinstance(node, exp.Anonymous) and node.name.lower() == "semantic_filter"
            for node in where.this.walk()
        )
        if count:
            groups.append(count)
    aqe = any(count > 1 for count in groups)
    # Disable chained Filter/Projection fusion for bare semantic maps whose native initializer asserts.
    chained_map_filter = _chained_scalar_map_filter(tree)
    return {
        "filter_group_sizes": groups,
        "aqe_eligible": aqe,
        "chained_map_filter_fusion_blocked": chained_map_filter,
        "static_operator_fusion": not aqe and not chained_map_filter,
        "reason": "Preserve individual predicates for native AQE fusion exploration"
        if aqe
        else "Bypass unsupported chained scalar Map/Filter fusion in the published binary"
        if chained_map_filter
        else "No multi-predicate filter group; retain native static operator fusion",
    }


def settings(sql):
    selected = options(sql)
    return [
        "SET enable_semantic_filter_multiplexer=true;",
        "SET enable_semantic_filter_dp=true;",
        "SET enable_semantic_operator_fusion="
        + str(selected["static_operator_fusion"]).lower()
        + ";",
        "SET semantic_filter_enable_fusion=true;",
        "SET semantic_filter_enable_batch=true;",
    ], selected


def profile_evidence(path, selected):
    operators = []

    def walk(node):
        if node.get("operator_name") == "SEM_FILTER_MULTIPLEXER":
            extra = node.get("extra_info") or {}
            operators.append(
                {
                    key: extra.get(key)
                    for key in (
                        "expression_exploration_tuples",
                        "path_exploration_tuples",
                        "exploitation_tuples",
                        "paths_count",
                        "best_path_index",
                        "best_path_expression",
                        "expression_exploration_time_ms",
                        "path_exploration_time_ms",
                        "exploitation_time_ms",
                    )
                }
            )
        for child in node.get("children", []):
            walk(child)

    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or "children" not in profile:
        raise RuntimeError("Sema did not emit a valid native operator profile; inspect profile.json")
    walk(profile)
    if selected["aqe_eligible"] and not operators:
        raise RuntimeError(
            "Sema multi-predicate region lost its native AQE operator; inspect profile.json"
        )
    return {
        **selected,
        "multiplexer_nodes": len(operators),
        "operators": operators,
        "expected_operator_verified": True,
        "reference_accuracy_evaluated": False,
    }
