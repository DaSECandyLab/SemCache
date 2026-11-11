"""Keep one dataset's workload, calibration, replay and model in one run."""

import copy
import hashlib
import shutil
from planning.settings.configuration import teacher_path
from training.artifacts import save


def scoped_configuration(settings, dataset, config_path):
    if dataset not in settings["runtime"]["source_databases"]:
        raise ValueError(f"Unknown training dataset: {dataset}")
    scoped = copy.deepcopy(settings)
    scoped["training"]["dataset"] = dataset
    # These statistics describe the training population, so they cannot be
    # inherited from a different dataset's global calibration or prefix bank.
    scoped["sampling"]["offline_statistics_dir"] = f".cache/prefix-statistics/{dataset}"
    scoped["calibration"]["cache_dir"] = f".cache/calibration/{dataset}"
    teacher = scoped["training"].get("teacher_config")
    if teacher:
        scoped["teacher_config"] = str(
            teacher_path({"teacher_config": teacher}, config_path)
        )
    return scoped


def select_manifest(manifest, dataset, expected_count):
    if dataset is None:
        return manifest
    scoped = copy.deepcopy(manifest)
    scoped["workloads"] = [w for w in scoped["workloads"] if w["dataset"] == dataset]
    if len(scoped["workloads"]) != expected_count:
        raise ValueError(
            f"Expected {expected_count} training queries for {dataset}; "
            f"found {len(scoped['workloads'])}"
        )
    return scoped


def validate_scope(manifest, dataset, expected_count):
    if dataset is None:
        return
    if len(manifest["workloads"]) != expected_count or any(
        w["dataset"] != dataset for w in manifest["workloads"]
    ):
        raise ValueError(f"Frozen training workload is not isolated to {dataset}")


def publish_model(run, checkpoint, dataset, workloads):
    """Expose the final round as the single inference model for this dataset."""
    if dataset is None or checkpoint is None:
        return
    destination = run / "value.pt"
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(checkpoint, temporary)
    temporary.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    save(
        run / "model.json",
        {
            "dataset": dataset,
            "checkpoint": str(destination),
            "source_checkpoint": str(checkpoint),
            "sha256": digest,
            "training_queries": [w["id"] for w in workloads],
            "training_scope": "one dataset; independent pretraining and feedback",
        },
    )
