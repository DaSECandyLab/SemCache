"""Reassess saved cardinalities under the current accepted dataset limits.

Reporting never starts queries or changes prompts, caches or stored decisions.
"""

import argparse
import json
import time
from pathlib import Path

from training.artifacts import save
from training.query_validation.errors import render_errors
from training.query_validation.targets import row_verdict


def render(root):
    state = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "workloads.json").read_text())
    errors = render_errors(root, state, manifest)
    results = []
    for query in manifest["workloads"]:
        r = state["queries"].get(query["id"])
        if (
            not r
            or r["status"] != "completed"
            or r["sql_sha256"] != query["sql_sha256"]
        ):
            continue
        verdict = row_verdict(r["rows"])
        results.append({"id": query["id"], "rows": r["rows"], **verdict})
    payload = {
        "status": state["status"],
        "target_minimum_result_rows": 50,
        "acceptable_minimum_result_rows": 30,
        "maximum_result_rows": 300,
        "assessed": len(results),
        "total": len(manifest["workloads"]),
        "passed": sum(r["status"] == "passed" for r in results),
        "results": results,
        "execution_error_operators": errors,
    }
    save(root / "acceptance.json", payload)
    lines = [
        "# 修复后的查询验收",
        "",
        f"状态：{state['status']}；已评估{len(results)}/{payload['total']}条，"
        f"通过{payload['passed']}条。",
        "",
        "目标结果约50～300行；按用户最新范围，只继续修复执行报错和少于30行的查询，30～49行不再为凑数调整。不设单个filter通过率上限。",
        "表中保留实测值，不裁剪查询结果、不复制记录凑数。",
        "未运行、报错或SQL已改动的查询均不算通过。",
        "",
        "| 查询 | 结果行数 | 验收 |",
        "|---|---:|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['rows']} | {r['classification']} |")
    lines += [
        "",
        "[出错算子：编号、过滤条件和模型输出](出错算子.md)",
        "",
        "[执行进度](progress.md)；本页按最新上限重新判定，无需重新发请求。",
    ]
    path = root / "验收结论.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)
    return state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        status = render(args.output)
        if not args.watch or status != "running":
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
