"""Run-owned model/kind statistics; query snapshots commit atomically once."""

import copy
import json
from pathlib import Path

from planning.state import digest


def shared_profile(profile):
    """Never share operator IDs, task selectivity, or query cache evidence."""
    if profile.get("calibration_protocol") != 2:
        raise ValueError("Global calibration requires typed API agreement protocol 2")
    result = copy.deepcopy(
        {
            key: profile[key]
            for key in (
                "calibration_protocol",
                "request_timing_protocol",
                "accuracy",
                "accuracy_observations",
                "request_profiles",
                "online_request_profiles",
                "relational_profiles",
            )
            if key in profile
        }
    )
    relational = result.setdefault("relational_profiles", {})
    if isinstance(relational, dict):
        # Query-local node IDs must never become global calibration identities.
        result["relational_profiles"] = {
            key: value
            for key, value in relational.items()
            if key == "default" or key.isupper()
        }
    return result


class GlobalCalibration:
    def __init__(self, path, models, teacher, initialize=None):
        self.path = Path(path)
        contract = {"version": 2, "models": models, "teacher": teacher.configuration()}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["contract"] != contract:
                raise ValueError("Global calibration model/teacher contract changed")
            if not self.state["profile"].get("relational_profiles"):
                if initialize is None:
                    raise ValueError(
                        "Missing global relational calibration; cannot restore training statistics"
                    )
                refreshed = shared_profile(initialize())
                refreshed["accuracy"] = self.state["profile"].get(
                    "accuracy", refreshed["accuracy"]
                )
                refreshed["request_profiles"] = self.state["profile"].get(
                    "request_profiles", refreshed["request_profiles"]
                )
                self.state = {**self.state, "profile": refreshed}
                self._save(self.state)
        else:
            if initialize is None:
                raise ValueError(
                    "Missing global calibration; cannot restore training statistics"
                )
            self.state = {
                "contract": contract,
                "profile": shared_profile(initialize()),
                "applied": {},
            }
            self._save(self.state)

    def _save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temporary.replace(self.path)

    def snapshot(self):
        return shared_profile(self.state["profile"])

    def merge_saved_request_timings(self, phases):
        """One-time upgrade of a legacy bank, using already recorded requests."""
        from planning.observations.request_profiles import (
            REQUEST_TIMING_PROTOCOL,
            merge_request_profiles,
        )

        if (
            self.state["profile"].get("request_timing_protocol")
            == REQUEST_TIMING_PROTOCOL
        ):
            return False
        profile = self.snapshot()
        scopes = set()
        for phase in phases:
            scope = phase["scope"]
            if scope in scopes:
                continue
            scopes.add(scope)
            merge_request_profiles(profile, phase)
        profile["request_timing_protocol"] = REQUEST_TIMING_PROTOCOL
        state = {
            **self.state,
            "profile": shared_profile(profile),
            "request_timing_upgrade": {"saved_phase_scopes": sorted(scopes)},
        }
        self._save(state)
        self.state = state
        return True

    def commit(self, token, before, after):
        before, after = shared_profile(before), shared_profile(after)
        identity = digest(after)
        if token in self.state["applied"]:
            if self.state["applied"][token] != identity:
                raise ValueError(
                    "Query already committed different calibration statistics"
                )
            return
        if digest(before) != digest(self.state["profile"]):
            raise ValueError("Query calibration has a stale parent snapshot")
        state = {
            **self.state,
            "profile": after,
            "applied": {**self.state["applied"], token: identity},
        }
        self._save(state)
        self.state = state
