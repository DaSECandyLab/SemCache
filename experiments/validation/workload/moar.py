"""Original MOAR workload metadata, operator contracts and native configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.core.paths import ROOT

MOAR_DATA = ROOT / "benchmarks/datasets/moar/test"

WORKLOADS: dict[str, dict[str, Any]] = {
    "biodex": {
        "source": MOAR_DATA / "biodex.json",
        "query": ROOT / "benchmarks/datasets/biodex/queries/moar/original/biodex.yaml",
        "required_fields": [
            "id",
            "fulltext_processed",
            "ground_truth_reactions",
            "possible_labels",
        ],
        "operators": ["map"],
        "output_fields": ["ranked_reaction_labels"],
        "text_fields": ["fulltext_processed", "possible_labels"],
    },
    "cuad": {
        "source": MOAR_DATA / "cuad.json",
        "query": ROOT / "benchmarks/datasets/cuad/queries/moar/original/cuad.yaml",
        "required_fields": ["document", "name", "id"],
        "operators": ["map"],
        "output_fields": ["clauses"],
        "text_fields": ["document"],
    },
    "blackvault": {
        "source": MOAR_DATA / "blackvault.json",
        "query": ROOT
        / "benchmarks/datasets/blackvault/queries/moar/original/blackvault.yaml",
        "required_fields": [
            "name",
            "content",
            "pdfs",
            "urlpath",
            "url",
            "uuid",
            "pdf_content",
            "all_content",
        ],
        "operators": ["map", "reduce"],
        "output_fields": ["event_type", "locations"],
        "text_fields": ["all_content"],
    },
    "medec": {
        "source": MOAR_DATA / "medec.json",
        "query": ROOT / "benchmarks/datasets/medec/queries/moar/original/medec.yaml",
        "required_fields": [
            "Text ID",
            "Text",
            "Sentences",
            "Error Flag",
            "Error Type",
            "Error Sentence ID",
            "Error Sentence",
            "Corrected Sentence",
            "Corrected Text",
        ],
        "operators": ["map"],
        "output_fields": ["is_error", "error_sentence", "corrected_sentence"],
        "text_fields": ["Text"],
    },
}


def port_contract(workload: str, system: str) -> dict[str, Any]:
    """Return whether a system port preserves every original operator."""
    if workload not in WORKLOADS:
        raise KeyError(workload)
    operators = WORKLOADS[workload]["operators"]
    if system == "moar":
        return {
            "supported": True,
            "operators": operators,
            "output_fields": WORKLOADS[workload]["output_fields"],
            "reason": None,
        }
    if system == "abacus":
        supported = workload in {"biodex", "cuad", "medec"}
        return {
            "supported": supported,
            "operators": operators,
            "output_fields": WORKLOADS[workload]["output_fields"],
            "reason": None
            if supported
            else "Abacus adapter does not replace BlackVault cross-row Reduce with a different aggregate",
        }
    if system in {"plop", "sema"}:
        return {
            "supported": False,
            "operators": operators,
            "output_fields": WORKLOADS[workload]["output_fields"],
            "reason": "native SQL port has no verified structured list/object semantic output and cannot express cross-row reduce",
        }
    if system == "nirvana":
        return {
            "supported": False,
            "operators": operators,
            "output_fields": WORKLOADS[workload]["output_fields"],
            "reason": "native long-context map/reduce contract has not been verified for this workload",
        }
    raise KeyError(system)


def native_moar_config(
    workload: str, input_path: Path, output_path: Path
) -> dict[str, Any]:
    """Load the original DocETL YAML and replace only its data/output paths."""
    import yaml

    if workload not in WORKLOADS:
        raise KeyError(workload)
    config = yaml.safe_load(WORKLOADS[workload]["query"].read_text())
    dataset_name = next(iter(config["datasets"]))
    config["datasets"][dataset_name]["path"] = str(Path(input_path).resolve())
    config["pipeline"]["output"]["path"] = str(Path(output_path).resolve())
    return config
