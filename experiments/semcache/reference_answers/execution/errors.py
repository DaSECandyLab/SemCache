"""Recover the first native cause instead of a secondary scheduler failure."""

import json


def query_failure(folder, error):
    if "query already failed" not in str(error):
        return {"reason": str(error), "telemetry": None}
    candidates = [
        *folder.glob("execution/native/run-*/telemetry.jsonl"),
        *folder.glob("attempt-*/native/run-*/telemetry.jsonl"),
    ]
    if candidates:
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        with path.open() as stream:
            for line in stream:
                if '"error"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                reason = event.get("error", "")
                if (
                    event.get("event") == "error"
                    and isinstance(reason, str)
                    and reason
                    and "query already failed" not in reason
                ):
                    return {
                        "reason": reason,
                        "telemetry": str(path),
                        "request_seq": event.get("request_seq"),
                    }
    return {"reason": str(error), "telemetry": None}
