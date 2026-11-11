"""Static prompt prefixes, history features and upstream cache-reuse graph edges."""

import math

from .schema import HISTORY_NUMERIC, known, log_value
from .statistics import feature_reliability

HISTORY_DISTANCES = ("age_seconds", "requests_since", "characters_since")


def static_prompt_features(op, action, profile):
    """A bound prompt's identity/length do not require a sampled input field."""
    from planning.observations.statistics import prefix, prefix_length

    system = profile.get("system_prefix")
    specification = profile.get("operator_specifications", {}).get(op["kind"])
    if not isinstance(system, str) or not isinstance(specification, str):
        return None, None
    namespace = prefix(op, action, system, specification)
    return namespace, prefix_length(namespace)


def bind_prompt_features(stat, namespace, length):
    """A newly bound namespace replaces a sampled layout's inherited length."""
    from planning.observations.statistics import prefix_length

    if stat.get("pre") is None and namespace is not None:
        stat["pre"], stat["pre_length"] = namespace, length
    if stat.get("pre_length") is None:
        stat["pre_length"] = prefix_length(stat.get("pre"))
        if stat["pre_length"] is None:
            stat["pre_length"] = length


def history_representation(packed, estimate):
    """Extend the native six values/six masks with the §4.2 reliability vector.

    Catalog counts and recorded load distances are known facts. Mean/max prefix
    matches inherit the cache estimate's reliability; an empty catalog proves
    zero external reuse independently of sample strength.
    """
    values, masks = list(packed[:6]), list(packed[6:])
    if not estimate.get("valid", {}).get("l_existing", True):
        for index in (0, 1, 3, 4, 5):
            values[index] = masks[index] = 0.0
    confidence = (
        1.0
        if masks[2] and values[2] == 0
        else feature_reliability(estimate, "l_existing", values[0])
    )
    reliability = [confidence * masks[0], confidence * masks[1], *masks[2:]]
    return values + reliability + masks


def build_history_feature_index(history):
    """Index one frozen catalogue without repeating model/prefix scans per node."""
    result = {}
    for entry in history:
        model = entry["model_id"]
        if model not in result:
            result[model] = {
                "unknown_pre": False,
                "by_pre": {},
                "scales": dict.fromkeys(HISTORY_DISTANCES, 0),
            }
        group = result[model]
        pre = entry.get("pre")
        if pre is None:
            group["unknown_pre"] = True
        else:
            group["by_pre"].setdefault(pre, []).append(entry)
        for key in HISTORY_DISTANCES:
            value = entry.get(key)
            if known(value) and value >= 0:
                group["scales"][key] = max(group["scales"][key], value)
    return result


def history_pairs(problem, action, estimate):
    """Deterministic §4.2 history statistics; unknown matches stay masked.

    Count all compatible sources, including those with an unknown match. Their
    mean/max and winning source cannot be identified until every match is known.
    """
    if estimate.get("pre") is None:
        return [0.0] * HISTORY_NUMERIC
    index = getattr(problem, "history_feature_index", None)
    if index is not None:
        group = index.get(action["model_id"])
        if group is not None and group["unknown_pre"]:
            # Its model is known but compatibility is not. Neither the exact
            # compatible count nor the winning source can be established.
            return [0.0] * HISTORY_NUMERIC
        compatible = group["by_pre"].get(estimate["pre"], ()) if group else ()
        scales = group["scales"] if group else dict.fromkeys(HISTORY_DISTANCES, 0)
    else:
        entries = [
            e for e in problem.cache_history if e["model_id"] == action["model_id"]
        ]
        compatible = []
        for entry in entries:
            if entry.get("pre") is None:
                return [0.0] * HISTORY_NUMERIC
            if entry["pre"] == estimate["pre"]:
                compatible.append(entry)
        scales = {
            k: max((e[k] for e in entries if known(e.get(k)) and e[k] >= 0), default=0)
            for k in HISTORY_DISTANCES
        }
    matches = estimate.get("l_existing", {})
    complete = all(
        known(matches.get(e["id"])) and matches[e["id"]] >= 0 for e in compatible
    )
    values, masks = (
        [0.0, 0.0, log_value(len(compatible)), 0.0, 0.0, 0.0],
        [float(complete), float(complete), 1.0, 0.0, 0.0, 0.0],
    )
    if complete and compatible:
        # Stable identity breaks equal-match ties independently of catalogue order.
        strongest = min(compatible, key=lambda e: (-matches[e["id"]], e["id"]))
        values[0] = log_value(
            math.fsum(matches[e["id"]] for e in compatible) / len(compatible)
        )
        values[1] = log_value(matches[strongest["id"]])
        for index, key in enumerate(HISTORY_DISTANCES, 3):
            distance = strongest.get(key)
            if known(distance) and distance >= 0:
                values[index] = distance / scales[key] if scales[key] else 0.0
                masks[index] = 1.0
    return history_representation(values + masks, estimate)


def history_features(problem, action, estimate):
    """Extend the native twelve-value reduction without scanning history again.

    Older profiles and the pure-Python training path do not contain the packed
    field; they retain the reference implementation for compatibility.
    """
    packed = estimate.get("l_existing_features")
    if isinstance(packed, list) and len(packed) == 12:
        if all(known(value) for value in packed) and packed[8] == 1:
            return history_representation([float(value) for value in packed], estimate)
    return history_pairs(problem, action, estimate)


def cache_relations(rows, by_operator, chosen, estimates, paths):
    """§4.2.3 compatibility defines an edge even if its strength is unknown."""
    for oid, nid in by_operator.items():
        stat, action = estimates[oid], chosen[oid]
        pre = stat.get("pre")
        if pre is None:
            continue
        matches = stat.get("l_previous", {})
        for source_id, source_node in by_operator.items():
            if (
                source_node not in paths[nid]
                or chosen[source_id]["model_id"] != action["model_id"]
                or estimates[source_id].get("pre") != pre
            ):
                continue
            value = matches.get(str(source_id), matches.get(source_id))
            rows[nid]["l_previous"][source_node] = (
                log_value(value)
                if known(value)
                and value >= 0
                and stat.get("valid", {}).get("l_previous", True)
                else None
            )
