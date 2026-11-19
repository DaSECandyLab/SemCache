# Benchmarks: Data, Queries, and Databases

This directory maintains the active SQL workloads, source datasets, and per-system databases for cross-system evaluation. The active benchmark suite consists of **5 datasets and 20 paper-derived queries**, translated for six systems (120 programs total). All query and dataset definitions are centralized here to avoid drift across subdirectories.

## Active Query Suite

| Dataset | Queries | Count |
|---|---|---:|
| Estate | Q5, Q9, Q10, Q11, Q12 | 5 |
| Movies / Nirvana | Q9, Q10, Q11, Q12 | 4 |
| Movies / Sema | Q10 | 1 |
| Appstore | Q7, Q8, Q9 | 3 |
| TPC-H | Q16, Q17, Q25, Q26, Q27, Q28, Q30 | 7 |

Deprecated queries (`Estate Q7/Q8`, `Movies/Nirvana Q5/Q7/Q8`, `Movies/Sema Q13/Q14/Q15`) and CoinMarketCap queries are excluded. `Appstore Q7/Q8/Q9` and `Movies/Sema Q10` follow the exact conditions from the corresponding papers and technical report appendices rather than repository variants.

Original queries live in [queries/original/](queries/original/), and system translations live in:
- SQL: [SemCache](queries/translated/semcache/), [Sema](queries/translated/sema/), [PLOP](queries/translated/plop/)
- Python: [Abacus](queries/translated/abacus/), [Nirvana](queries/translated/nirvana/), [Lotus](queries/translated/lotus/)

TPC-H uses deterministic key ordering and the updated `Q17` order-customer join condition. Single-use CTEs are inlined, while `MATERIALIZED` and multi-reference CTEs are preserved. Numeric projections, deterministic aggregations, `NULL`s, duplicate rows, `ORDER BY`/`LIMIT`, and relational field dependencies follow a shared execution contract.

## Datasets

Download the [datasets](https://drive.google.com/file/d/1jjHYS8-VLoP8TDs1XTEnV7sz8oD7gKUi/view) and extract them into `benchmarks/datasets/` at the repository root. Neither `datasets/` nor generated `databases/` are tracked in Git.

| Dataset | Size and Location |
|---|---|
| Estate / Nirvana | 1,041 rows in `datasets/estate/data/nirvana/` (Parquet `image` column is not sent to LLMs) |
| Movies / Nirvana | 249 rows, 22 columns in `datasets/movies/data/nirvana/movie_data.csv` |
| Movies / Sema | 7 tables, 35,159 total rows in `datasets/movies/data/sema/` |
| Appstore / Sema | 2 tables, 75,126 total rows in `datasets/appstore/data/sema/` |
| TPC-H | SF=0.005, 8 tables in `datasets/tpch/data/` |

The two Movies datasets are stored separately. Estate preserves full text and all rows; Appstore preserves stable source row identifiers; TPC-H does not truncate inputs with `LIMIT`. `datasets/paper-workloads/` retains source provenance notes only.

## Per-System Databases

Each of the six systems (`semcache`, `sema`, `plop`, `abacus`, `nirvana`, `lotus`) has five databases under `databases/<system>/` with identical entity tables and compatible `raw_<table>` views (DuckDB for SQL systems, system-specific adapters for Python systems).

| Database | Entity Tables | Total Rows |
|---|---:|---:|
| Estate | 1 | 1,041 |
| Movies / Nirvana | 1 | 249 |
| Movies / Sema | 7 | 35,159 |
| Appstore | 2 | 75,126 |
| TPC-H | 8 | 43,531 |

Public SF=0.01 TPC-H Parquet files are kept only as provenance records; active comparisons use SF=0.005. Database preparation (`databases/prepare.py`) makes no model calls and validates schemas, `NULL`s, duplicates, row counts, and full result multisets.

## SemCache Usage

SemCache training queries live in `../experiments/semcache/queries/` and training source databases live in `../experiments/semcache/data/`. Heldout test execution reads directly from `queries/translated/semcache/`.

Run structural validation (no LLM calls):

```bash
PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.validation
```

For training, reference collection, and evaluation workflows, see [../experiments/README.md](../experiments/README.md). Experiment outputs are written to `../experiments/results/`.
