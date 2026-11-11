"""Score saved system outputs using fixed fields followed by a WorkBuddy judge."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save
from experiments.system_comparison.accuracy.artifacts import (
    load_campaign,
    load_references,
)
from experiments.system_comparison.accuracy.matching import (
    SOURCE_GROUNDED_RUBRIC,
    TEXT_FIELDS,
    finish_match,
    prepare_match,
)
from experiments.system_comparison.accuracy.report import write_report
from experiments.system_comparison.accuracy.teacher import (
    judge_pairs,
    review_unresolved,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse-judgments", type=Path, action="append", default=[])
    parser.add_argument(
        "--aggregate-metric",
        choices=["relative_deviation", "input_rows"],
        default="relative_deviation",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    references, evidence, contract = load_references(args.references)
    campaign, records, results = load_campaign(
        args.run, references, evidence, contract, aggregate_metric=args.aggregate_metric
    )
    if set(campaign["query_ids"]) != set(references):
        raise ValueError("Campaign/reference query sets differ")
    jobs, cases = {}, {}
    for record in records:
        task_id = record["task_id"]
        if record["query_id"] in TEXT_FIELDS:
            record["text_evaluation_profile"] = SOURCE_GROUNDED_RUBRIC["version"]
        if task_id not in results:
            continue
        actual, expected = results[task_id]
        scope = {
            "reference_sha256": record["reference_sha256"],
            "teacher_config_sha256": digest(args.teacher_config),
            "protocol": "fixed-fields-then-workbuddy-v1",
        }
        try:
            text_fields = None
            if record["aggregate"]:
                # Judge the actual saved aggregate input, including any generated
                # text that survives in it, rather than only its final number.
                text_fields = [
                    name
                    for name, _ in expected["columns"]
                    if name in {"Amenities", "Features", "Genre", "Character"}
                ]
            case, pending = prepare_match(
                actual, expected, record["query_id"], scope, text_fields=text_fields
            )
            jobs.update(pending)
            cases[task_id] = case
        except (ValueError, KeyError, TypeError) as error:
            record.update(scoring_status="invalid_output", note=str(error))
    args.judgments.mkdir(parents=True, exist_ok=True)
    save(
        args.judgments / "comparison-plan.json",
        {
            "reference_root": str(args.references.resolve()),
            "campaign": str(args.run.resolve()),
            "unique_pairs": len(jobs),
            "pairs": jobs,
        },
    )
    print(
        f"Prepared {len(cases)} scoreable tasks; {len(jobs)} teacher review items",
        flush=True,
    )
    if args.prepare_only:
        return
    judgments, audit = judge_pairs(
        jobs,
        args.teacher_config,
        args.judgments,
        workers=args.workers,
        batch_size=args.batch_size,
        reuse_dirs=args.reuse_judgments,
    )
    judgments, audit = review_unresolved(
        jobs, judgments, audit, args.teacher_config, args.judgments
    )
    for record in records:
        if record["task_id"] in cases:
            record.update(finish_match(cases[record["task_id"]], judgments))
    report = {
        "version": "fixed-fields-then-workbuddy-v1",
        "aggregate_metric": args.aggregate_metric,
        "evaluation_profiles": dict.fromkeys(TEXT_FIELDS, SOURCE_GROUNDED_RUBRIC),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run": str(args.run.resolve()),
        "references": str(args.references.resolve()),
        "reference_contract_sha256": digest(args.references / "contract.json"),
        "campaign_sha256": digest(args.run / "campaign.json"),
        "query_ids": campaign["query_ids"],
        "variants": [v["name"] for v in campaign["variants"]],
        "reference_info": evidence,
        "judgments": str(args.judgments.resolve()),
        "teacher_pairs": len(jobs),
        "teacher_audit": audit,
        "records": records,
    }
    write_report(report, args.output)
    print(
        f"Saved report {args.output}; {dict(Counter(r['scoring_status'] for r in records))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
