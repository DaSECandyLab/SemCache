"""Audit the selected translations against their explicit source contracts."""

from __future__ import annotations

import ast
import json
from collections import Counter

import sqlglot
from sqlglot import exp

from experiments.core.paths import TRANSLATED as OUT
from experiments.validation.audits import source as inspect
from experiments.validation.audits.sql_calls import sql_functions
from experiments.validation.runtime.filter_groups import merge_filter_stages
from experiments.validation.workload import source as original
from experiments.validation.workload.linear import inline_ctes
from experiments.validation.workload.numeric import CONTRACT, rewrite, stabilize_limits


def operation_signature(operations):
    return [
        {
            "method": op["method"],
            "task": op["kwargs"]["user_instruction"],
            "inputs": op["kwargs"].get(
                "input_columns", [op["kwargs"].get("input_column")]
            ),
            "outputs": op["kwargs"].get("output_columns", []),
        }
        for op in operations
    ]


def sql_signature(tree):
    return [
        {
            "method": "semantic_map"
            if op["kind"] == "semantic_project"
            else op["kind"],
            "task": op["task"].split("\nInputs:", 1)[0],
            "inputs": [
                c.name
                for value in op["inputs"]
                for c in sqlglot.parse_one(value, read="duckdb").find_all(exp.Column)
            ],
            "outputs": op["fields"],
        }
        for op in sql_functions(tree)
    ]


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text())
    selected = json.loads((original.QUERY_ROOT / "manifest.json").read_text())[
        "queries"
    ]
    records = []
    extensions = []
    for record in manifest["records"]:
        if record["contract"] != CONTRACT:
            extensions.append(record)
            continue
        query = next(
            q for q in selected if q["dataset"] + "/" + q["query"] == record["id"]
        )
        source = original.QUERY_ROOT / query["original"]
        path = original.QUERY_ROOT / record["query_file"]
        errors = []
        logical = None
        try:
            if inspect.digest(source) != query["original_sha256"] or record[
                "source_sha256"
            ] != inspect.digest(source):
                errors.append("Frozen original hash differs")
            if (
                record["source"] != query["original"]
                or inspect.digest(path) != record["sha256"]
            ):
                errors.append("Generated identity or hash differs")
            if (
                inspect.digest(original.QUERY_ROOT / record["metadata_file"])
                != record["metadata_sha256"]
            ):
                errors.append("Metadata hash differs")
            aggregate, numeric_columns = None, []
            if source.suffix == ".py":
                source_ops = [
                    {
                        "method": op["method"],
                        "kwargs": {
                            "user_instruction": op["task"],
                            **(
                                {"input_column": op["inputs"][0]}
                                if op["method"] == "semantic_reduce"
                                else {"input_columns": op["inputs"]}
                            ),
                            **(
                                {"output_columns": op["outputs"]}
                                if op["outputs"]
                                else {}
                            ),
                        },
                    }
                    for op in inspect.source_operations(source)
                ]
                operations, aggregate, numeric_columns = rewrite(
                    record["id"], source_ops
                )
                expected = operation_signature(operations)
                if path.suffix == ".py":
                    actual = inspect.python_operations(path, record["system"])
                    tree = ast.parse(path.read_text())
                    bindings = [
                        n
                        for n in ast.walk(tree)
                        if isinstance(n, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "AGGREGATE"
                            for t in n.targets
                        )
                    ]
                    actual_aggregate = (
                        ast.literal_eval(bindings[0].value) if bindings else None
                    )
                    if actual_aggregate != aggregate:
                        errors.append("Deterministic aggregate contract differs")
                    if aggregate:
                        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
                        used = any(
                            any(
                                k.arg == "aggregate"
                                and isinstance(k.value, ast.Name)
                                and k.value.id == "AGGREGATE"
                                for k in n.keywords
                            )
                            or isinstance(n.func, ast.Name)
                            and n.func.id == "aggregate_frame"
                            and any(
                                isinstance(a, ast.Name) and a.id == "AGGREGATE"
                                for a in n.args
                            )
                            for n in calls
                        )
                        if not used:
                            errors.append("Aggregate declared but never executed")
                else:
                    tree = inspect.native_tree(path.read_text(), record["system"])
                    actual = sql_signature(tree)
                    expected_tree = inline_ctes(
                        sqlglot.parse_one(
                            original.linear_sql(
                                query,
                                operations=operations,
                                aggregate=aggregate,
                                numeric_columns=numeric_columns,
                            )[0],
                            read="duckdb",
                        )
                    )
                    if record["system"] == "sema":
                        expected_tree = merge_filter_stages(expected_tree)
                    if inspect.canonical_sql(tree) != inspect.canonical_sql(
                        expected_tree
                    ):
                        errors.append(
                            "Relational structure, numeric schema or aggregate differs"
                        )
                if actual != expected:
                    errors.append("Semantic task, field or operator order differs")
                if any(op["method"] in {"semantic_reduce", "sem_agg"} for op in actual):
                    errors.append("LLM Reduce remains in the common workload")
                logical = actual + (
                    [{"method": "deterministic_aggregate", **aggregate}]
                    if aggregate
                    else []
                )
            else:
                if path.suffix == ".py":
                    module = ast.parse(path.read_text())
                    binding = next(
                        n
                        for n in ast.walk(module)
                        if isinstance(n, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "RELATIONAL_SQL"
                            for t in n.targets
                        )
                    )
                    actual_tree = sqlglot.parse_one(
                        ast.literal_eval(binding.value), read="duckdb"
                    )
                else:
                    actual_tree = inspect.native_tree(
                        path.read_text(), record["system"]
                    )
                actual_canonical = inspect.canonical_sql(
                    inspect.inline_appstore(actual_tree)
                )
                expected_canonical = inspect.canonical_sql(
                    stabilize_limits(inspect.source_tree(source))
                )
                if actual_canonical != expected_canonical:
                    errors.append(
                        "Original SQL relation or bound semantic predicate differs"
                    )
                logical = sql_functions(inspect.inline_appstore(actual_tree))
            if (
                record["aggregate"] != aggregate
                or record["numeric_projection_columns"] != numeric_columns
            ):
                errors.append("Manifest does not describe actual rewrite")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        records.append(
            {
                "id": record["id"],
                "system": record["system"],
                "passed": not errors,
                "aggregate": record["aggregate"],
                "query_file": record["query_file"],
                "execution_mode": record["execution_mode"],
                "logical_operations": logical,
                "errors": errors,
            }
        )
    if (
        Counter(r["system"] for r in records) != dict.fromkeys(original.SYSTEMS, 19)
        or len({(r["system"], r["id"]) for r in records}) != 114
    ):
        raise ValueError("Missing or duplicate system/query pairs")
    if extensions:
        from experiments.validation.audits.extended import CONTRACT as EXTENSION
        from experiments.validation.audits.extended import audit_records

        if any(r["contract"] != EXTENSION for r in extensions):
            raise ValueError("Unknown extension contract")
        records.extend(audit_records(extensions))
    selected_ids = {q["dataset"] + "/" + q["query"] for q in selected}
    expected_pairs = {(system, identity) for system in original.SYSTEMS for identity in selected_ids}
    if len(records) != len(expected_pairs) or {(r["system"], r["id"]) for r in records} != expected_pairs:
        raise ValueError("Selected sources and per-system translations differ")
    if Counter(r["system"] for r in records) != manifest["query_count_per_system"]:
        raise ValueError("Manifest query counts differ from actual translations")
    return {
        "contract": CONTRACT,
        "total": len(records),
        "passed": sum(r["passed"] for r in records),
        "rewritten_reduce_files": sum(r["aggregate"] is not None for r in records),
        "external_model_requests": 0,
        "actual_model_answer_equivalence_proven": False,
        "records": records,
    }


def main():
    result = audit()
    binding_path = OUT / "native-binding.json"
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        result["native_sql_binding"] = {
            "bound": binding["bound"],
            "total": binding["total"],
            "matches_current_manifest": binding["query_manifest_sha256"]
            == inspect.digest(OUT / "manifest.json"),
        }
        hashes = {
            (r["system"], r["id"]): r["sha256"]
            for r in json.loads((OUT / "manifest.json").read_text())["records"]
        }
        result["native_sql_binding"]["matches_current_queries"] = all(
            hashes.get((r["system"], r["id"])) == r["query_sha256"]
            for r in binding["records"]
        )
    (OUT / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 六系统共同数值工作负载检查",
        "",
        f"当前共 {result['total'] // len(original.SYSTEMS)} 条查询、{result['total']} 个系统程序。仅保留论文及技术报告中的查询；原有 LLM Reduce 继续沿用共同数值契约，不宣称等价于原始 Reduce。",
        "",
        "| 查询 | SemCache | Sema | PLOP | Abacus | Nirvana | Lotus | 聚合 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for query_id in dict.fromkeys(r["id"] for r in result["records"]):
        rows = [
            next(
                r for r in result["records"] if r["id"] == query_id and r["system"] == s
            )
            for s in original.SYSTEMS
        ]
        aggregate = rows[0]["aggregate"]
        cells = ["通过" if r["passed"] else "失败" for r in rows]
        lines.append(
            "| "
            + query_id
            + " | "
            + " | ".join(cells)
            + " | "
            + (
                aggregate["function"] + "(" + (aggregate["column"] or "*") + ")"
                if aggregate
                else "—"
            )
            + " |"
        )
    lines += [
        "",
        f"实际文件结构核验：{result['passed']}/{result['total']}。无模型调用，不能据此断言实际模型输出一致。",
        "",
        "Python 系统的数学聚合及 SQL 来源任务使用公开的 DuckDB 关系宿主，语义操作调用各自系统；原生全查询优化范围不同。不得把适配后的整条查询标为原生全 DAG 优化。",
        "",
        "单位、NULL、空输入、重复行及 COUNT(*) 规则见 README.md。原 TPC-H 七处无序 LIMIT 统一按稳定键排序后取样，避免跨引擎任意子集；原始条件与 LIMIT 数量保留。模型结果质量仍按独立的 COMMON_ACCURACY_PROTOCOL 评价。",
        "",
    ]
    (OUT / "SEMANTIC_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    for record in result["records"]:
        if record["errors"]:
            print(record["system"], record["id"], record["errors"])
    sql_count = sum(
        r["system"] in {"semcache", "sema", "plop"} for r in result["records"]
    )
    if (
        result["passed"] != result["total"]
        or binding_path.exists()
        and (
            binding["bound"] != sql_count
            or not result["native_sql_binding"]["matches_current_manifest"]
            or not result["native_sql_binding"]["matches_current_queries"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
