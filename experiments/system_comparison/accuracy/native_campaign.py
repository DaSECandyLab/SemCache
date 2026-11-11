"""Score saved configurations against their own system's Teacher references."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save

from .aggregates import aggregate_deviation
from .matching import (
    SOURCE_GROUNDED_RUBRIC,
    TEXT_FIELDS,
    VERSION,
    finish_match,
    prepare_match,
)
from .native_summary import summarize_native
from .offline_campaign import load_native_campaign, write_report
from .teacher import judge_pairs, review_unresolved


def prepare_campaign(run, references, teacher_config):
    """Validate saved artifacts and prepare fixed-first, source-grounded row F1."""
    report, results = load_native_campaign(run, references)
    teacher_config = Path(teacher_config).resolve()
    scope = {"teacher_config_sha256": digest(teacher_config), "protocol": VERSION}
    cases, jobs = {}, {}
    for record in report["records"]:
        task_id, query = record["task_id"], record["query_id"]
        if task_id not in results:
            continue
        actual, expected = results[task_id]
        try:
            if record["aggregate"]:
                record.update(aggregate_deviation(actual, expected))
                continue
            case, pending = prepare_match(
                actual, expected, query,
                {**scope, "reference_sha256": record["reference_sha256"]},
            )
            cases[task_id] = case
            jobs.update(pending)
            record.update(finish_match(case, {}))
            if query in TEXT_FIELDS:
                record["text_evaluation_profile"] = SOURCE_GROUNDED_RUBRIC["version"]
        except (ValueError, KeyError, TypeError) as error:
            record.update(scoring_status="invalid_output", note=str(error))
    report.update(
        version="native-campaign-fixed-first-semantic-row-f1-v1",
        evaluation_mode="semantic",
        text_protocol=VERSION,
        evaluation_profiles=dict.fromkeys(TEXT_FIELDS, SOURCE_GROUNDED_RUBRIC),
        teacher_config=str(teacher_config),
        teacher_config_sha256=scope["teacher_config_sha256"],
        aggregate_metric="relative_deviation",
        teacher_pairs=len(jobs),
    )
    return report, cases, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-judgments", type=Path, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve() if args.output else args.run.resolve() / "accuracy"
    judgments_dir = args.judgments.resolve() if args.judgments else output / "judgments"
    report, cases, jobs = prepare_campaign(args.run, args.references, args.teacher_config)
    report["judgments"] = str(judgments_dir)
    judgments_dir.mkdir(parents=True, exist_ok=True)
    save(judgments_dir / "comparison-plan.json", {
        "campaign": report["run"],
        "campaign_sha256": report["campaign_sha256"],
        "reference_root": report["references"],
        "reference_contract_sha256": report["reference_contract_sha256"],
        "teacher_config_sha256": report["teacher_config_sha256"],
        "text_protocol": VERSION,
        "scoreable_tasks": len(cases),
        "unique_pairs": len(jobs),
        "pairs": jobs,
    })
    report["summary"] = summarize_native(report)
    write_report(report, output)
    print(f"Prepared {len(cases)} row-F1 tasks; {len(jobs)} scoped teacher items; "
          f"{dict(Counter(r['scoring_status'] for r in report['records']))}", flush=True)
    if args.prepare_only:
        return
    judgments, audit = judge_pairs(
        jobs, args.teacher_config, judgments_dir,
        workers=args.workers, batch_size=args.batch_size,
        reuse_dirs=args.reuse_judgments,
    )
    judgments, audit = review_unresolved(
        jobs, judgments, audit, args.teacher_config, judgments_dir,
    )
    for record in report["records"]:
        if record["task_id"] in cases:
            record.update(finish_match(cases[record["task_id"]], judgments))
    report.update(
        generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        teacher_audit=audit,
        http_requests=audit.get("cumulative", {}).get("http_requests", 0),
        http_requests_this_run=audit.get("http_requests", 0),
    )
    report["summary"] = summarize_native(report)
    write_report(report, output)
    print(f"Saved {output / 'accuracy.md'}; "
          f"{dict(Counter(r['scoring_status'] for r in report['records']))}", flush=True)


if __name__ == "__main__":
    main()
