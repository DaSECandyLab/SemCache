"""Assign each query to one teacher, preserving published answer provenance."""

import json


def assign_teachers(root, workloads, teachers, *, allow_routing_update=False):
    names = list(teachers)
    previous = root / "contract.json"
    frozen = json.loads(previous.read_text()) if previous.exists() else {}
    assignments = frozen.get("query_teachers", {})
    previously_active = frozen.get(
        "active_teacher_routes", list(frozen.get("teachers", {}))
    )
    redistribute = allow_routing_update and set(previously_active) != set(teachers)
    signatures = {name: teacher.configuration() for name, teacher in teachers.items()}
    result, next_lane = {}, 0
    for workload in workloads:
        identity = workload["id"]
        answer = root / identity / "answer.json"
        if answer.exists():
            saved = json.loads(answer.read_text())
            matching = [
                name
                for name, config in signatures.items()
                if saved.get("teacher") == config
            ]
            if len(matching) != 1:
                old_name = assignments.get(identity)
                overrides = frozen.get("query_teacher_configs", {})
                expected = overrides.get(
                    identity, frozen.get("teachers", {}).get(old_name)
                )
                if (
                    not matching
                    and isinstance(old_name, str)
                    and saved.get("teacher") == expected
                    and (
                        allow_routing_update
                        or identity in overrides
                        or old_name not in previously_active
                    )
                ):
                    result[identity] = old_name
                else:
                    raise ValueError(
                        f"Published answer has no unique matching teacher: {identity}"
                    )
            else:
                result[identity] = matching[0]
        elif identity in assignments and not redistribute:
            name = assignments[identity]
            if name not in teachers or (
                frozen["teachers"][name] != signatures[name]
                and not allow_routing_update
            ):
                raise ValueError(f"Frozen query teacher differs: {identity}")
            result[identity] = name
        else:
            result[identity] = names[next_lane % len(names)]
            next_lane += 1
    return result
