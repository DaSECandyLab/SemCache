"""Check explicit equivalence evidence when a public template has been rewritten."""

from pathlib import Path

from reference_answers.storage.files import file_digest

from training.artifacts import read


def load_template_rewrites(directory):
    """Authorize historical/current template hash pairs only from pinned evidence."""
    marker = Path(directory) / "template-rewrites.json"
    if not marker.is_file():
        return frozenset()
    proof = read(marker)
    certificate = Path(proof["certificate"])
    if file_digest(certificate) != proof["sha256"]:
        raise ValueError("Template rewrite certificate changed")
    installed = read(certificate)
    validation = Path(installed["validation"])
    if file_digest(validation) != installed["validation_sha256"]:
        raise ValueError("Template rewrite validation changed")
    verified = read(validation)
    cases = verified.get("cases", [])
    schemas = verified.get("schema_checks", [])
    if (
        verified.get("status") != "completed"
        or verified.get("model_calls") != 0
        or not cases
        or not schemas
        or len(cases) != verified.get("total_cases")
        or any(int(row["old_only"]) or int(row["new_only"]) for row in cases)
        or not all(row["matches"] for row in schemas)
    ):
        raise ValueError(
            "Template rewrite needs completed equivalence and schema checks"
        )
    hashes = verified["query_sql_sha256"]
    pairs = set()
    for record in installed["records"]:
        identity = record["query"].removeprefix("translated/semcache/")
        key = Path(identity).name
        expected = {
            "original": record["original_sql_sha256"],
            "rewritten": record["sql_sha256"],
        }
        if hashes[key] != expected or not any(row["query"] == key for row in cases):
            raise ValueError(f"Template rewrite hashes differ: {identity}")
        pairs.add((identity, expected["original"], expected["rewritten"]))
    return frozenset(pairs)
