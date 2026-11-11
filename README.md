# SemCache: Semantic Query Processing with Prefix Cache Awareness

SemCache is a semantic SQL engine built on DuckDB. It optimizes query plans, prompt layouts, and model selection with awareness of reusable prompt prefixes.

- **Semantic SQL:** execute semantic filters, projections, and joins alongside relational operators.
- **Cache-aware optimization:** estimate prefix reuse across operators and query history.
- **Learned planning:** predict execution time and accuracy with a graph-based value model.
- **Runtime adaptation:** update semantic operator implementations using execution statistics.

[Installation](#installation) · [Quick Start](#quick-start) · [Training](#training) · [Evaluation](#evaluation) · [Experiments](#experiments) · [Project Structure](#project-structure)

## Installation

### Requirements

- Linux, Python 3.12, and a C++17 compiler
- CMake, Ninja, and OpenSSL development libraries
- PyTorch with CPU or CUDA support
- OpenAI-compatible model endpoints supporting structured boolean output

Run the following commands from the repository root:

```bash
git submodule update --init --recursive -- SemCache/third_party/duckdb

python3.12 -m venv SemCache/.venv-tools
python3.12 -m venv SemCache/.venv-optimizer
SemCache/.venv-tools/bin/python -m pip install cmake ninja
SemCache/.venv-optimizer/bin/python -m pip install \
  -r SemCache/optimizer/requirements.txt \
  -r experiments/requirements.txt

make -C SemCache release
```

Set the Python module path in each terminal:

```bash
export PYTHONPATH=SemCache/optimizer:experiments/semcache:.
```

### Data

Download the [datasets](https://drive.google.com/file/d/1jjHYS8-VLoP8TDs1XTEnV7sz8oD7gKUi/view) and extract them into `benchmarks/datasets/`.

| Dataset | Test queries | Training queries |
|---|---:|---:|
| Estate | 5 | 15 |
| Movies / Nirvana | 4 | 12 |
| Movies / Sema | 1 | 3 |
| Appstore | 3 | 9 |
| TPC-H | 7 | 21 |
| **Total** | **20** | **60** |

Prepare the benchmark databases under `benchmarks/databases/semcache/` using the schemas in the [benchmark documentation](benchmarks/README.md#独立数据库). Configure training database paths in `SemCache/config/system/runtime.json`, then import the training sources:

```bash
SemCache/.venv-optimizer/bin/python -m training.source_data --rebuild
```

The importer reads the dataset manifest and source files. TPC-H uses a scale factor of `0.005`.

### Models

Deploy the E2B, E4B, and 12B models behind OpenAI-compatible endpoints. Set their addresses in `SemCache/config/system/models.json`:

```json
{
  "models": {
    "endpoints": [
      "http://127.0.0.1:30001",
      "http://127.0.0.1:30002",
      "http://127.0.0.1:30003"
    ],
    "max_output_tokens": 512
  },
  "teacher_config": "teacher.local.json"
}
```

Each endpoint should expose one distinct model ID through `/v1/models`.

Create a Teacher configuration for reference answers and accuracy calibration:

```bash
cp SemCache/config/teacher/teacher.example.json \
  SemCache/config/teacher/teacher.local.json
```

Set the endpoint, model, and `api_key_env` in this file. Export the named API key environment variable, and set `training.teacher_config` in `SemCache/config/system/training.json` to `teacher.local.json`.

## Quick Start

Run a query with a trained value model and its accompanying `models.json`. Set `TRAIN_RUN` to the training directory and choose a saved checkpoint.

**Terminal 1 — start the optimizer:**

```bash
export PYTHONPATH=SemCache/optimizer:experiments/semcache:.
TRAIN_RUN=experiments/results/semcache/model_training/your-training-run

SemCache/.venv-optimizer/bin/python SemCache/optimizer/optimizer_server.py \
  --config SemCache/config/system/config.json \
  --socket /tmp/semcache-optimizer.sock \
  --checkpoint "$TRAIN_RUN/measured/round-3/model/value.pt" \
  --plan-generation heuristic_value \
  --sampling-database benchmarks/databases/semcache/estate.duckdb \
  --sampling-models "$TRAIN_RUN/models.json" \
  --teacher-config SemCache/config/teacher/teacher.local.json \
  --parallelism 16
```

**Terminal 2 — execute a query:**

```bash
export PYTHONPATH=SemCache/optimizer:experiments/semcache:.
TRAIN_RUN=experiments/results/semcache/model_training/your-training-run
QUERY_RUN="experiments/results/semcache/execution/estate-Q5-$(date +%Y%m%d-%H%M%S)"

SemCache/.venv-optimizer/bin/python SemCache/optimizer/query_runner.py \
  --query benchmarks/queries/translated/semcache/estate/Q5.sql \
  --db benchmarks/databases/semcache/estate.duckdb \
  --models "$TRAIN_RUN/models.json" \
  --optimizer-socket /tmp/semcache-optimizer.sock \
  --parallelism 16 \
  --output-dir "$QUERY_RUN"
```

Use the same database and model registry for optimization and execution. Results, the selected plan, request telemetry, and profiling output are saved in `QUERY_RUN`.

Test SQL is available in `benchmarks/queries/translated/semcache/`. Replace `--query`, `--db`, and `--sampling-database` to run another query or dataset.

## Training

Training collects execution experience and fits the value model to latency and accuracy labels.

### 1. Collect reference answers

```bash
SemCache/.venv-optimizer/bin/python -m reference_answers.semcache \
  --workloads experiments/semcache/queries/workloads.json \
  --teacher-config SemCache/config/teacher/teacher.local.json \
  --output experiments/results/reference_answers/semcache/training \
  --parallelism 16
```

Set `training.reference_answers_dir` to this output directory. Teacher cache verification is configured through `reference_answers.semcache.verify_on_second_use`.

### 2. Configure training

Edit `SemCache/config/system/training.json`:

| Setting | Purpose |
|---|---|
| `training.dataset` | Set to `null` for all 60 queries, or select a dataset |
| `training.workload_manifest` | Training query manifest |
| `training.reference_answers_dir` | Reference answer directory |
| `training.pretrain_rounds` | Initial collection rounds |
| `training.rounds` | Feedback rounds |
| `training.epochs`, `training.round_update` | Fitting schedule and stopping criteria |

Network settings are in `value_model.json`; learning parameters are in `learning.json`.

### 3. Start training

```bash
TRAIN_RUN="experiments/results/semcache/model_training/train-$(date +%Y%m%d-%H%M%S)"

SemCache/.venv-optimizer/bin/python experiments/semcache/training/launcher.py \
  --run "$TRAIN_RUN"
```

Use a new directory for each run. The launcher runs in the background and writes progress reports to the training directory.

| Output | Contents |
|---|---|
| `预训练数据收集.md` | Initial collection progress |
| `真实训练数据收集.md` | Feedback collection and execution results |
| `模型训练.md` | Loss, elapsed time, and checkpoints |
| `models.json` | Resolved model registry |
| `calibration/global.json` | Operator calibration statistics |
| `measured/round-N/model/value.pt` | Value model checkpoint |

Follow the log, stop training, or resume a run:

```bash
tail -f "$TRAIN_RUN/train.log"
```

```bash
SemCache/.venv-optimizer/bin/python experiments/semcache/training/launcher.py stop
SemCache/.venv-optimizer/bin/python experiments/semcache/training/launcher.py \
  resume "$TRAIN_RUN"
```

Training collection disables the local prompt result cache.

## Evaluation

Evaluate a completed training run on the 20 test queries. Prepare matching test reference answers in `experiments/results/reference_answers/semcache/testing/`.

```bash
TRAIN_RUN=experiments/results/semcache/model_training/your-training-run
EVAL_RUN="experiments/results/semcache/evaluation/trained-20-$(date +%Y%m%d-%H%M%S)"

SemCache/.venv-optimizer/bin/python -m training.evaluation.checkpoint \
  --run "$TRAIN_RUN" \
  --output "$EVAL_RUN" \
  --heldout \
  --references experiments/results/reference_answers/semcache/testing \
  --cases min-50 min-80 \
  --global-http-limit 16
```

- `--heldout`: evaluate the test queries.
- `--cases`: select accuracy targets from `min-20`, `min-50`, and `min-80`.
- `--checkpoint-round N`: select a training round.
- `--global-http-limit`: set total HTTP concurrency.

Each accuracy target runs across the full query set. Evaluation enables the prompt result cache, disables second-use verification, and resets caches before each target group. Add `--preserve-cache` to retain caches across groups.

Results are written to `EVAL_RUN/测试结果.md`, including search time, predicted latency and accuracy, execution time, F1, and per-model request counts.

### Runtime adaptation

Train the cache predictor from collected execution telemetry:

```bash
SemCache/.venv-optimizer/bin/python -m training.cache_prediction \
  --run "$TRAIN_RUN" \
  --output "$TRAIN_RUN/cache_prediction" \
  --fit --once
```

Set `cache_predictor_checkpoint` in `SemCache/config/system/runtime_adaptation.json` to the absolute path of `cache_prediction/cache_tokens_mlp.pt`. Add `--runtime-adaptation-config SemCache/config/system/runtime_adaptation.json` to the evaluation command and use a new output directory.

Runtime adaptation searches semantic operator implementations in the background and applies updates at chunk boundaries while keeping the plan topology fixed.

## Experiments

Initialize comparison system sources:

```bash
git submodule update --init --recursive -- systems/
```

| System | Upstream repository |
|---|---|
| Abacus / Palimpzest | [mitdbg/palimpzest](https://github.com/mitdbg/palimpzest) |
| Nirvana | [JunHao-Zhu/nirvana](https://github.com/JunHao-Zhu/nirvana) |
| PLOP / Morrila | [DMoRanSky/Morrila](https://github.com/DMoRanSky/Morrila) |
| LOTUS | [lotus-data/lotus](https://github.com/lotus-data/lotus) |
| MOAR / DocETL | [ucbepic/docetl](https://github.com/ucbepic/docetl) |
| Sema | [BITQiKangK/Sema](https://github.com/BITQiKangK/Sema) |

Prepare each system's environment, adapters, benchmark databases, and input manifests. The Sema comparison uses the paper executable, deployed separately from the public source submodule.

Run the comparison suite:

```bash
COMPARE_RUN="experiments/results/system_comparison/compare-$(date +%Y%m%d-%H%M%S)"

PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.system_comparison \
  --run "$COMPARE_RUN" \
  --workload all20 \
  --preset three-model-quality-grid \
  --gateway-request-limit 16
```

Reference collection and accuracy scoring use `experiments.system_comparison.references` and `experiments.system_comparison.accuracy.native_campaign`. See [Experiments](experiments/README.md) for the experiment layout and [Benchmarks](benchmarks/README.md) for query definitions.

## Project Structure

```text
SemCache/
├── semantic-duckdb/       Semantic operators and execution runtime
├── beam-search-duckdb/    Native plan search and sampling
├── optimizer/            Planning service, features, and value model
├── statistics/           Prefix sketches and cache source directories
└── config/               System configuration and Teacher template
benchmarks/
├── queries/              Original and translated test queries
├── datasets/             Downloaded datasets
└── databases/            Benchmark databases
experiments/
├── config/               Experiment configuration and Teacher template
├── semcache/             Training, reference collection, and evaluation
├── system_comparison/    Cross-system experiments
└── results/              Reports, logs, and checkpoints
systems/                  Comparison system submodules
```

Core configuration is composed through `SemCache/config/system/config.json`. Experiment configuration is under `experiments/config/system/`. Runtime paths in the core configuration are relative to `SemCache/`; `include` paths are relative to the containing configuration file.

## Development

Rebuild after changing C++ code:

```bash
make -C SemCache release
```

Check query structure and bindings:

```bash
PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.validation
```

Command-line entry points provide their options through `--help`.

## License

See [LICENSE](SemCache/LICENSE). Third-party dependencies retain their respective licenses.
