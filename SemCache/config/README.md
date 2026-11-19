# SemCache Configuration

Core system settings are composed from `SemCache/config/system/config.json` via its `include` list. Paths inside `system/` modules are resolved relative to `SemCache/`, and `include` entries are resolved relative to `config.json`. Teacher credentials and endpoints are kept in separate files under `SemCache/config/teacher/`.

## System Modules (`system/`)

| File | Scope | Key settings |
|---|---|---|
| `runtime.json` | Python/C++ build paths, database locations, process CPU limit | `python`, `build`, `native_bindings`, `run_root`, `source_databases`, `max_cpu_threads`, `shutdown_timeout_seconds` |
| `models.json` | Candidate OpenAI-compatible model endpoints and default Teacher file | `models.endpoints`, `models.max_output_tokens`, `teacher_config` |
| `execution.json` | Execution threads, per-model concurrency, external Teacher isolation | `cpu_parallelism`, `max_concurrent_requests_per_model`, `max_concurrent_external_teacher_requests`, `enforce_model_operator_order` |
| `response_cache.json` | Persistent prompt result cache and second-use verification | `directory`, `verify_on_second_use`, `max_disk_bytes`, `max_memory_bytes`, `max_entries`, `max_entry_bytes`, `max_correction_retries` |
| `statistics.json` | Offline prefix-sketch statistics build parallelism | `build_parallelism` |
| `calibration.json` | Operator timing and accuracy calibration settings | `operator_time.{sample_rows,relational_repeats,seed}`, `sample_rows`, `cohort_seed`, `cache_dir` |
| `sampling.json` | Online selectivity & accuracy sampling budgets and adaptive sketches | `operator_sample_rows`, `max_operator_sample_rows`, `selectivity_parallel_models`, `relational_sample_rows`, `relational_cardinality_sample_rows`, `max_llm_requests`, `max_accuracy_batches_per_operator`, `accuracy_sample_units`, `cpu_parallelism`, `adaptive_sketch` |
| `search.json` | Beam search budgets and complexity-triggered grouped pruning | `beam_width`, `workers`, `fixed_model_id`, `inference.*`, `training.*` |
| `inference.json` | Planning mode, objective preference, GPU inference settings | `plan_generation`, `preference`, `device`, `gpu_batch_size`, `reference_evaluation_run`, `reference_aggregate_manifest` |
| `runtime_adaptation.json` | Runtime chunk-boundary re-optimization and cache predictor | `enabled`, `beam_width`, `device`, `selectivity_threshold`, `cache_threshold`, `cache_trigger_fraction`, `cache_predictor_checkpoint`, `max_switches` |
| `training.json` | Feedback collection rounds, timeouts, fitting schedule | `rounds`, `pretrain_rounds`, `timeout`, `minimum_timeout`, `feedback_cycle`, `round_update`, `teacher_config` |
| `value_model.json` | Graph value-network architecture | `width`, `heads`, `layers`, `use_cache_features` |
| `learning.json` | Value-model optimizer hyperparameters | `algorithm`, `device`, `threads`, `batch_size`, `learning_rate`, `warmup_fraction`, `patience`, `discount` |
| `reference_answers.json` | Teacher reference collection settings | `semcache.{verify_on_second_use,parallelism,cpu_parallelism,max_output_tokens,timeout_seconds}` |
| `monitoring.json` | Progress reporting and stall detection intervals | `refresh_seconds`, `health_interval`, `stalled_after`, `stop_query_after` |

## HTTP Concurrency & External Teacher Isolation (`execution.json`)

```json
{
  "execution": {
    "cpu_parallelism": 16,
    "max_concurrent_requests_per_model": 16,
    "max_concurrent_external_teacher_requests": 16,
    "enforce_model_operator_order": false
  }
}
```

- **Local model pool (`SEMCACHE_GLOBAL_HTTP_CONCURRENCY`)**: When `--global-http-limit` is passed to training/evaluation launchers, `SharedRequestAdmission` (`Scope::Local`) caps total concurrent HTTP requests across locally deployed candidate models (loopback hosts such as `http://127.0.0.1:30001..30003`, or endpoints with `"external": false`).
- **External Teacher pool (`max_concurrent_external_teacher_requests`)**: External Teacher endpoints (remote URLs or endpoints marked `"external": true` by `TeacherModel.transport_models`) use an independent `SharedRequestAdmission` queue (`Scope::External`) and never consume or wait on local concurrency slots. Setting `max_concurrent_external_teacher_requests` (default `16`, mapped to `SEMCACHE_EXTERNAL_HTTP_CONCURRENCY`; set `null` to disable the global external cap) bounds external Teacher requests independently.

## Beam Search & Grouped Pruning (`search.json`)

```json
{
  "search": {
    "beam_width": 90,
    "workers": 8,
    "fixed_model_id": "gemma4-e2b",
    "inference": {
      "max_expansions": 100000,
      "max_seconds": 60,
      "layer_pruning": "pareto",
      "grouped_operator_threshold": 10,
      "max_structure_groups": 15,
      "max_implementations_per_group": 6
    },
    "training": {
      "max_expansions": 1000000,
      "max_seconds": 600
    }
  }
}
```

- `layer_pruning`: Default layer pruning strategy during inference (`"pareto"`, `"accuracy_time"`, or `"grouped"`).
- `grouped_operator_threshold`: During value-guided inference (`heuristic_value`, `training=false`), when `layer_pruning` is `"pareto"` and a query has strictly more than `grouped_operator_threshold` operators (`len(problem.all_ops) > 10`), `SearchConfig.for_problem` automatically switches to `"grouped"` layer pruning. Set to `null` to disable automatic switching.
- `max_structure_groups` / `max_implementations_per_group`: Group and per-group quotas used when `"grouped"` pruning is active.

## Teacher Configurations (`teacher/`)

- `teacher.example.json`: Template for external OpenAI-compatible Teacher providers (`base_url`, `model`, `api_key_env`, `thinking`, `reasoning_effort`, `structured_output`, `max_output_tokens`, `connect_timeout_ms`).
- Non-loopback endpoints in `teacher.*.json` are automatically treated as external by `ModelRegistry::Parse` and `TeacherModel.transport_models`.
