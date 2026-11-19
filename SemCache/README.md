# SemCache

SemCache is a DuckDB-based semantic query engine with prefix-cache awareness. Relational operators and semantic Filter, Projection, and Join operators execute within one engine: the Python optimizer manages plan states, sampling, feature encoding, and the value model, C++ extensions handle search and execution state, and SGLang serves the candidate LLMs.

## Components

| Path | Description |
|---|---|
| `semantic-duckdb/` | DuckDB semantic operators, prompt rendering, HTTP transport, and runtime |
| `beam-search-duckdb/` | Native Beam search, sampling, statistics, and cache history extension |
| `optimizer/` | Planning service, state representation, features, sampling, search, and training |
| `statistics/` | Coordinated bottom-k prefix sketches and cross-field summaries |
| `config/` | Modular system, search, execution, and Teacher configurations ([config/README.md](config/README.md)) |
| `../experiments/` | Training, reference collection, cross-system comparison, and results |
| `../benchmarks/` | Shared SQL workloads, datasets, and databases |
| `../sglang/` | Model serving launchers, SSH tunnel scripts, and prefix-cache patches |

## Build

```sh
git submodule update --init --recursive -- third_party/duckdb
python3.12 -m venv .venv-tools
python3.12 -m venv .venv-optimizer
.venv-optimizer/bin/python -m pip install -r optimizer/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make release
```

`make release` builds DuckDB, `semantic.duckdb_extension`, `beam_search.duckdb_extension`, `semcache_statistics_build`, and the native Python bindings. Normal queries, sampling, and training use the Release build.

Quality checks and tests:

```sh
make format-check
make optimizer-typecheck
PYTHONPATH=optimizer .venv-optimizer/bin/pytest -q tests/
git diff --check
```

## Usage

- Model serving and endpoints: [../sglang/README.md](../sglang/README.md)
- Configuration reference: [config/README.md](config/README.md)
- Training, reference collection, and evaluation: [../experiments/README.md](../experiments/README.md)
- Workloads and databases: [../benchmarks/README.md](../benchmarks/README.md)

Prefix statistics estimate reuse potential rather than guaranteeing online KV cache hits. SHA-256 hashing inside bottom-k sketches is part of the statistical ranking algorithm and is unrelated to file integrity verification.
