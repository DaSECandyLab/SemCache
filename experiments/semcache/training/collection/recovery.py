"""Bounded retries of cold queries; never invent a restored serving KV state."""

import copy
import re
import time
from datetime import datetime

from training.artifacts import save


def is_network_failure(error):
    """Recognize transport failures, not SQL timeouts or malformed responses."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "http transport failure",
            "connection timed out",
            "connection reset",
            "connection refused",
            "remote end closed connection",
            "temporary failure in name resolution",
            "model endpoints unavailable:",
        )
    ) or bool(re.search(r"\bstatus=(408|429|502|503|504)\b", message))


def scientific_protocol(protocol):
    """Exclude operational retry and fitting settings from collection identity."""
    result = copy.deepcopy(protocol)
    # Serving endpoints are loaded from the run's model metadata, not compared
    # with newly discovered services or changed teacher API files at startup.
    result.pop("models", None)
    result.pop("teacher", None)
    result.get("configuration", {}).get("training", {}).pop("network_retry", None)
    configuration = result.get("configuration", {})
    configuration.pop("models", None)
    configuration.pop("teacher_config", None)
    # CUDA chunking controls memory and dispatch overhead, not the candidate
    # domain, learned weights, objectives or Beam width.
    configuration.get("inference", {}).pop("gpu_batch_size", None)
    configuration.get("search", {}).pop("workers", None)
    configuration.get("training", {}).pop("checkpoint_epochs", None)
    configuration.get("training", {}).pop("final_update", None)
    configuration.get("training", {}).pop("round_update", None)
    learning = configuration.get("learning", {})
    learning.pop("batch_size", None)
    return result


def archive_attempt(destination, run_dir):
    if not destination.exists():
        return None
    root = run_dir / "network-retries" / destination.relative_to(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt-{index}").exists():
        index += 1
    archived = root / f"attempt-{index}"
    destination.rename(archived)
    return str(archived.relative_to(run_dir))


def collect_with_retry(
    operation,
    *,
    destination,
    run_dir,
    phase,
    policy,
    reset,
    update,
    previous_failure=None,
):
    """Preserve failed evidence and clear cold KV/history before recollection."""
    marker = destination / "network-failure.json"
    if destination.exists():
        if phase != "estimated" or not (
            marker.exists() or is_network_failure(previous_failure)
        ):
            raise RuntimeError(
                f"Incomplete query is not a recoverable cold network failure: {destination}"
            )
        archive_attempt(destination, run_dir)
    retries = policy.get("max_retries", 2)
    delay = policy.get("backoff_seconds", 5)
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as error:
            if not is_network_failure(error):
                raise
            destination.mkdir(parents=True, exist_ok=True)
            failure = {
                "error": str(error),
                "attempt": attempt + 1,
                "at": datetime.now().astimezone().isoformat(),
                "phase": phase,
                "retryable_query": phase == "estimated",
            }
            save(marker, failure)
            if phase != "estimated" or attempt >= retries:
                raise
            archived = archive_attempt(destination, run_dir)
            wait = min(30, delay * 2**attempt)
            update(network_retry={**failure, "archive": archived, "wait_seconds": wait})
            print(
                f"Network failure: retry {attempt + 1}/{retries} in {wait}s; {destination.name}",
                flush=True,
            )
            time.sleep(wait)
            reset()
