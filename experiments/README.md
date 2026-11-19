# Experiments

`experiments/` contains orchestration code, experiment manifests, validation tools, and result artifacts. The core optimizer lives in `SemCache/optimizer/` and shared SQL/data live in `benchmarks/`.

## Layout

| Directory | Contents |
|---|---|
| `semcache/` | SemCache training, checkpoint evaluation, reference answers, and monitoring |
| `system_comparison/` | Cross-system comparison runner for PLOP, Sema, Nirvana, LOTUS, Abacus, and MOAR |
| `validation/` | Structural SQL, binding, and runtime contract checks |
| `results/` | Generated reports, logs, and checkpoints (untracked) |
| `core/` | Shared path, atomic-write, and process utilities |

## Configuration

System, training, evaluation, and monitoring settings live in `SemCache/config/system/` and are composed via `SemCache/config/system/config.json`. Teacher templates live in `SemCache/config/teacher/teacher.example.json` (credentials are read from environment variables). See [SemCache/config/README.md](../SemCache/config/README.md) for details.

## SemCache Training

Training queries live in `semcache/queries/`, source databases in `semcache/data/`, and reference answers in `semcache/reference_answers/`. The training pipeline collects plan rollouts and execution labels, merges global calibration statistics, fits the graph value model, and evaluates checkpoints on heldout queries.

```sh
PYTHONPATH=SemCache/optimizer:experiments/semcache:. \
  SemCache/.venv-optimizer/bin/python \
  experiments/semcache/training/launcher.py --run experiments/results/semcache/model_training/train-run
```

## Cross-System Comparison

Cross-system comparisons evaluate 6 systems across 5 datasets and 20 queries. Scheduling, data loading, SQL execution, backend adapters, transport logging, reporting, and accuracy scoring live in `system_comparison/` (results are written to `results/system_comparison/`), while each system's native optimizer stays in `systems/`.

```sh
PYTHONPATH=. SemCache/.venv-optimizer/bin/python \
  -m experiments.system_comparison --help
```

| Submodule | Responsibility |
|---|---|
| `scheduling/` | Campaign lifecycle and task orchestration |
| `data/` | Dataset and input boundary management |
| `execution/` | SQL execution and result capture |
| `backends/` | Per-system execution adapters |
| `transport/` | Model request gateway and prompt caching |
| `reporting/` | Progress and timing reports |
| `accuracy/` | Reference scoring (multiset matching for fixed fields, Teacher judge for free text, relative error for aggregates) |
| `diagnostics/` | Model-free structural checks |

Errors, timeouts, and unsupported operators are reported explicitly and are never coerced to empty results.

## Validation

Run structural contract validation without invoking any LLM:

```sh
PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.validation
```
