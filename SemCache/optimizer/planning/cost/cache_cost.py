"""Character-domain reuse potential and externally calibrated latency priors.

No value here asserts physical KV residence. Combination ranges assume exact
marginals; with sketches/survival approximations they are estimates, not confidence
intervals. Cross terms already use the target's full request denominator.
"""

import math
from dataclasses import dataclass

REUSE_BASIS = "self-first-cross-v1"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


@dataclass(frozen=True, slots=True)
class ReusePrior:
    lower: float
    upper: float | None

    @property
    def point(self):
        """Strongest-known-source prior only when every component is known."""
        return self.lower if self.upper is not None else None


def reuse_prior(info):
    """Add repeated visits to a range for the union of first-visit sources.

    max(cross) and sum(cross) do not assume independence or identify overlap.
    Source enumeration alone cannot certify unknown component values. Legacy
    records without enumeration metadata cannot establish a complete union.
    """
    complete = info.get("cache_sources_complete") is True
    valid = info.get("valid", {})
    own = info.get("l_self") if valid.get("l_self", True) else None
    if own is None:
        complete, own = False, 0.0
    elif not _number(own):
        raise ValueError("Invalid self reuse potential")
    cross = []
    for key in ("l_existing", "l_previous"):
        entries = info.get(key, {})
        if not isinstance(entries, dict):
            raise ValueError("Invalid cross-source reuse potential")
        if not valid.get(key, True):
            complete = False
            continue
        for value in entries.values():
            if value is None:
                complete = False
            elif not _number(value):
                raise ValueError("Invalid cross-source reuse potential")
            else:
                cross.append(value)
    lower = own + max(cross, default=0.0)
    upper = own + math.fsum(cross) if complete else None
    if not _number(lower) or upper is not None and not _number(upper):
        raise ValueError("Invalid combined reuse potential")
    return ReusePrior(lower, upper)


def request_seconds(calibration, info, *, audit=None):
    """Use observed mean unless one complete, centered external fit is supplied.

    The fit groups its reference latency, reference potential, slope and identity
    so a new warm average is never accidentally used as a zero-reuse intercept.
    Its effective slope relates potential to external latency, not to live KV.
    """
    mean = calibration.get("seconds_per_request")
    basis = "observed average request latency"
    if mean is not None and not _number(mean):
        raise ValueError(
            "Invalid sampled model request time for estimated training target"
        )
    legacy_rate = calibration.get("cached_character_seconds", 0.0)
    if not _number(legacy_rate):
        raise ValueError("Invalid cached-character time calibration")
    fit = calibration.get("cache_time_prior")
    if fit is not None:
        if (
            not isinstance(fit, dict)
            or fit.get("reuse_basis") != REUSE_BASIS
            or not isinstance(fit.get("calibration_id"), str)
            or not fit["calibration_id"].strip()
            or not all(
                _number(fit.get(key))
                for key in (
                    "reference_seconds_per_request",
                    "reference_reuse_characters",
                    "cached_character_seconds",
                )
            )
        ):
            raise ValueError("Invalid centered cache time calibration")
        if mean is None:
            mean = fit["reference_seconds_per_request"]
            basis = "external calibrated reference latency"
    # The ordinary Beam path does not create bounds/audit objects or inspect
    # summaries. It only reads the existing scalar latency mean.
    if fit is None and audit is None:
        return mean
    potential = reuse_prior(info)
    reason = "no centered cache calibration"
    seconds = mean
    if fit is not None:
        if potential.point is None:
            reason = "incomplete cache evidence"
        else:
            seconds = max(
                0.0,
                fit["reference_seconds_per_request"]
                - fit["cached_character_seconds"]
                * (potential.point - fit["reference_reuse_characters"]),
            )
            if not _number(seconds):
                raise ValueError("Invalid centered cache time prediction")
            reason, basis = None, "centered external cache-potential prior"
    elif legacy_rate:
        reason = "uncentered legacy cache coefficient ignored"
    if audit is not None:
        audit.update(
            basis=basis,
            seconds_per_request=seconds,
            fallback_reason=reason,
            reuse_basis=REUSE_BASIS,
            reuse_potential={
                "point": potential.point,
                "combination_range": [potential.lower, potential.upper],
                "source_enumeration_complete": info.get("cache_sources_complete")
                is True,
            },
        )
        if fit is not None:
            audit["calibration_id"] = fit["calibration_id"]
    return seconds
