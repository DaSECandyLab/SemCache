"""One explicit teacher transition after a completed estimated bootstrap.

The original bank is never moved or rewritten.  This module is stdlib-only so
the read-only monitors and training pipeline resolve exactly the same paths.
"""

import copy
import hashlib
import json
from pathlib import Path

from reference_answers.storage.files import file_digest

MARKER = "calibration/measured-epoch.json"
EPOCH = "measured-none-v1"
_HASHES = {
    "protocol_sha256": "protocol.json",
    "source_calibration_sha256": "calibration/global.json",
    "checkpoint_sha256": "estimated/round-1/model/value.pt",
    "estimated_experience_sha256": "estimated/round-1/experience.json",
}
_FIELDS = {
    "version",
    "id",
    "start_phase",
    "start_round",
    "source_teacher",
    "target_teacher",
    "reference_root",
    "reference_contract_sha256",
    *_HASHES,
}


def _read(path):
    return json.loads(Path(path).read_text())


def measured_epoch(run):
    """Read the opt-in marker; malformed markers fail closed everywhere."""
    path = Path(run) / MARKER
    if not path.exists():
        return None
    document = _read(path)
    if (
        not isinstance(document, dict)
        or set(document) != _FIELDS
        or type(document["version"]) is not int
        or document["version"] != 1
        or document["id"] != EPOCH
        or document["start_phase"] != "measured"
        or type(document["start_round"]) is not int
        or document["start_round"] != 1
        or not isinstance(document["source_teacher"], dict)
        or not isinstance(document["target_teacher"], dict)
    ):
        raise ValueError("Invalid measured teacher epoch marker")
    if not isinstance(document["source_teacher"].get("endpoint"), dict):
        raise ValueError("Invalid source teacher in measured epoch")
    expected = copy.deepcopy(document["source_teacher"])
    expected["endpoint"]["thinking"] = "disabled"
    expected["endpoint"]["reasoning_effort"] = "none"
    if document["target_teacher"] != expected or expected == document["source_teacher"]:
        raise ValueError(
            "Measured epoch permits only the explicit disabled/none teacher switch"
        )
    for key in (*_HASHES, "reference_contract_sha256"):
        value = document[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid measured epoch evidence hash")
    if (
        not isinstance(document["reference_root"], str)
        or not Path(document["reference_root"]).is_absolute()
    ):
        raise ValueError("Measured epoch reference root must be absolute")
    return document


def epoch_identity(document):
    if document is None:
        return None
    encoded = json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def calibration_paths(run, phase=None):
    """Estimated evidence stays original; current priors use the opted-in epoch."""
    run = Path(run)
    document = measured_epoch(run)
    root = run / "calibration"
    if document is not None and phase != "estimated":
        root /= "measured-none"
    return {
        "global": root / "global.json",
        "probes": root / "probes",
        "progress": root / "progress.json",
    }


def _completed_bootstrap(run):
    protocol = _read(run / "protocol.json")
    bank = _read(run / "calibration/global.json")
    workloads = _read(run / "estimated/round-1/workloads.json")["workloads"]
    report = _read(run / "estimated/round-1/model/training.json")
    expected = {f"estimated/1/{w['id']}" for w in workloads if w["split"] == "train"}
    if (
        protocol["pretrain_rounds"] != 1
        or report.get("source") != "estimated"
        or report.get("epochs", 0) < protocol["epochs"]
        or report.get("queries") != len(workloads)
        or not report.get("value_available")
        or set(bank["applied"]) != expected
        or not bank["profile"].get("relational_profiles")
        or not (run / _HASHES["checkpoint_sha256"]).is_file()
        or not (run / _HASHES["estimated_experience_sha256"]).is_file()
    ):
        raise ValueError(
            "Measured teacher switch requires the complete fitted estimated bootstrap"
        )
    return protocol, bank


def prepare_measured_epoch(run, target_teacher):
    """Build a reviewable marker without writing it or calling a model service.

    Activation is intentionally separate: publish this JSON atomically once the
    caller's required reference collection has finished and the teacher changed.
    """
    run = Path(run)
    if (
        (run / MARKER).exists()
        or (run / "measured").exists()
        or (run / "calibration/measured-none").exists()
    ):
        raise ValueError(
            "Enable the teacher epoch once, before any measured directory exists"
        )
    _, bank = _completed_bootstrap(run)
    expected = copy.deepcopy(bank["contract"]["teacher"])
    expected["endpoint"].update(thinking="disabled", reasoning_effort="none")
    if target_teacher != expected or target_teacher == bank["contract"]["teacher"]:
        raise ValueError(
            "Only the approved disabled/none teacher configuration may be activated"
        )
    root = Path(_read(run / "reference_answers_path.json")["root"]).resolve()
    return {
        "version": 1,
        "id": EPOCH,
        "start_phase": "measured",
        "start_round": 1,
        "source_teacher": bank["contract"]["teacher"],
        "target_teacher": target_teacher,
        "reference_root": str(root),
        "reference_contract_sha256": file_digest(root / "contract.json"),
        **{key: file_digest(run / relative) for key, relative in _HASHES.items()},
    }


def validate_measured_epoch(run, models, teacher):
    """Verify immutable parent evidence and target identity before any new calls."""
    run = Path(run)
    document = measured_epoch(run)
    if document is None:
        return None
    _, bank = _completed_bootstrap(run)
    if (
        bank["contract"]["models"] != models
        or bank["contract"]["teacher"] != document["source_teacher"]
        or teacher.configuration() != document["target_teacher"]
        or any(
            file_digest(run / relative) != document[key]
            for key, relative in _HASHES.items()
        )
        or _read(run / "reference_answers_path.json")["root"]
        != document["reference_root"]
        or file_digest(Path(document["reference_root"]) / "contract.json")
        != document["reference_contract_sha256"]
    ):
        raise ValueError(
            "Measured teacher epoch differs from its frozen teacher or bootstrap evidence"
        )
    identity = epoch_identity(document)
    measured = run / "measured"
    target_bank = calibration_paths(run)["global"]
    if target_bank.exists() and _read(target_bank).get("contract") != {
        "version": 2,
        "models": models,
        "teacher": document["target_teacher"],
    }:
        raise ValueError("Measured epoch calibration model/teacher contract changed")
    if measured.exists():
        if not target_bank.is_file():
            raise ValueError("Missing epoch calibration for existing measured evidence")
        rounds = list(measured.iterdir())
        if not rounds:
            raise ValueError(
                "Teacher epoch cannot adopt an existing empty measured directory"
            )
        for directory in rounds:
            manifest = directory / "round.json"
            if (
                not manifest.is_file()
                or _read(manifest).get("teacher_epoch") != identity
            ):
                raise ValueError(
                    "Measured artifacts predate or differ from the teacher epoch"
                )
    elif target_bank.exists() and _read(target_bank).get("applied"):
        raise ValueError(
            "New measured epoch has calibration commits without measured evidence"
        )
    return document
