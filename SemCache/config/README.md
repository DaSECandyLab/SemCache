# SemCache Configuration

Core system settings are composed from `SemCache/config/system/config.json` via its `include` list. Paths inside `system/` modules are resolved relative to `SemCache/`, and `include` entries are resolved relative to `config.json`. Teacher credentials and endpoints are kept separately under `SemCache/config/teacher/`.

## Module Index (`system/`)

| File | Scope |
|---|---|
| `config.json` | Root manifest listing the included configuration modules |
| `runtime.json` | Python/C++ paths, source databases, process CPU limit, shutdown timeout |
| `models.json` | Candidate model endpoints, token cap, and default Teacher config |
| `execution.json` | CPU parallelism, local per-model concurrency, external Teacher concurrency |
| `response_cache.json` | Persistent prompt result cache and second-use verification limits |
| `statistics.json` | Offline prefix-sketch statistics build concurrency |
| `calibration.json` | Operator timing and accuracy calibration settings |
| `sampling.json` | Online selectivity/accuracy sampling budgets and adaptive prefix sketches |
| `search.json` | Beam width, search budgets, and grouped layer pruning configuration |
| `inference.json` | Planning guidance mode, objective preference, GPU inference settings |
| `runtime_adaptation.json` | Runtime chunk-boundary re-optimization and cache token predictor |
| `training.json` | Feedback collection rounds, timeouts, fitting schedule, and checkpoints |
| `value_model.json` | Graph value-network architecture parameters |
| `learning.json` | Value-model optimizer and loss weighting hyperparameters |
| `reference_answers.json` | Reference Teacher answer collection settings |
| `monitoring.json` | Progress refresh, health check, and stall detection intervals |

---

## `runtime.json`

| Key | Default | Description |
|---|---|---|
| `runtime.python` | `.venv-optimizer/bin/python` | Python interpreter used by launchers and build scripts |
| `runtime.build` | `build/release` | DuckDB and native extension build directory |
| `runtime.native_bindings` | `build/release-bindings/overlay` | Native Python C++ binding overlay path |
| `runtime.run_root` | `../experiments/results/semcache/model_training` | Root output directory for training runs |
| `runtime.source_databases` | 5 dataset `.duckdb` paths | Training DuckDB paths for `movies_nirvana`, `movies_sema`, `estate`, `appstore`, `tpch` |
| `runtime.max_cpu_threads` | `16` | Process CPU affinity cap (also bounds `execution`/`sampling`/`learning`/`search` workers) |
| `runtime.shutdown_timeout_seconds` | `15` | Graceful shutdown timeout for managed child processes |

## `models.json`

| Key | Default | Description |
|---|---|---|
| `models.endpoints` | `127.0.0.1:30001..30003` | OpenAI-compatible base URLs for candidate LLMs (discovered via `/v1/models`) |
| `models.max_output_tokens` | `512` | Max output tokens requested per candidate LLM call |
| `teacher_config` | `teacher.magpie.json` | Default Teacher JSON filename (under `config/teacher/`) or path |

## `execution.json`

| Key | Default | Description |
|---|---|---|
| `execution.cpu_parallelism` | `16` | DuckDB execution worker threads |
| `execution.max_concurrent_requests_per_model` | `16` | In-flight HTTP request cap per physical model in `RequestScheduler` |
| `execution.max_concurrent_external_teacher_requests` | `16` | Independent global concurrency cap for external Teacher endpoints (`SEMCACHE_EXTERNAL_HTTP_CONCURRENCY`; `null` disables the cap) |
| `execution.enforce_model_operator_order` | `false` | Prioritize queued requests by plan operator order within each model |

- **Local vs. external concurrency**: `--global-http-limit` sets `SEMCACHE_GLOBAL_HTTP_CONCURRENCY` for `SharedRequestAdmission` (`Scope::Local`), bounding only locally deployed endpoints (loopback hosts or `"external": false`). External Teacher endpoints use an independent `Scope::External` queue controlled by `max_concurrent_external_teacher_requests` and never occupy or block local model slots.

## `response_cache.json`

| Key | Default | Description |
|---|---|---|
| `response_cache.directory` | `.cache/prompt-results` | Persistent prompt result cache directory (`null` disables cache storage) |
| `response_cache.verify_on_second_use` | `false` | Require an independent second-use confirmation before trusting cached results |
| `response_cache.max_disk_bytes` | `268435456` | On-disk cache size limit (bytes) |
| `response_cache.max_memory_bytes` | `16777216` | In-memory cache size limit (bytes) |
| `response_cache.max_entries` | `32768` | Maximum cached prompt entries |
| `response_cache.max_entry_bytes` | `262144` | Maximum size per cached entry (bytes, must be `>= 56`) |
| `response_cache.max_correction_retries` | `5` | Maximum majority-vote retries when verification mismatches occur |

## `statistics.json`

| Key | Default | Description |
|---|---|---|
| `statistics.build_parallelism` | `8` | Worker threads used when building offline prefix-sketch statistics |

## `calibration.json`

| Key | Default | Description |
|---|---|---|
| `calibration.operator_time.sample_rows` | `10` | Input sample rows per operator for runtime calibration probes |
| `calibration.operator_time.relational_repeats` | `3` | Repeat count for relational timing calibration |
| `calibration.operator_time.seed` | `42` | Random seed for timing probe selection |
| `calibration.sample_rows` | `16` | Sample rows per operator for global accuracy calibration |
| `calibration.cohort_seed` | `104729` | Cohort selection seed for calibration |
| `calibration.cache_dir` | `.cache/calibration` | Directory caching calibration probe artifacts (`null` disables caching) |

## `sampling.json`

| Key | Default | Description |
|---|---|---|
| `sampling.operator_sample_rows` | `20` | Initial input sample size per semantic operator |
| `sampling.max_operator_sample_rows` | `50` | Maximum sample size after zero-output refill waves |
| `sampling.selectivity_parallel_models` | `true` | Dispatch selectivity sample rows across candidate models in parallel |
| `sampling.relational_sample_rows` | `200` | Sample rows materialized for relational operators |
| `sampling.relational_cardinality_sample_rows` | `2000` | Sample rows used for population cardinality probes |
| `sampling.ensure_prefix_depth_rates` | `true` | Ensure per-depth selectivity coverage for cache-reuse estimation |
| `sampling.max_llm_requests` | `128` | Query-level LLM request budget for selectivity sampling |
| `sampling.max_accuracy_batches_per_operator` | `1` | Maximum accuracy comparison batches per logical operator |
| `sampling.accuracy_sample_units` | `4` | Units evaluated per operator against the Teacher during accuracy sampling |
| `sampling.cpu_parallelism` | `16` | CPU workers for native sampling SQL stages and key/sketch work |
| `sampling.seed` | `42` | Deterministic sampling seed |
| `sampling.collect_calibration` | `true` | Record sampling timing/accuracy observations into calibration state |
| `sampling.history_capacity` | `256` | Maximum sliding prefix-cache history entries retained per query |
| `sampling.trace_timings` | `false` | Record fine-grained C++ `PhaseTimer` spans |
| `sampling.offline_statistics_dir` | `.cache/prefix-statistics` | Directory containing prebuilt offline table prefix sketches |
| `sampling.adaptive_sketch` | `{...}` | Coordinated bottom-k adaptive sketch parameters (`sample_size: 4096`, `max_depth: 4096`, `max_candidates: 64`, `max_anchors: 24`, `budget_bytes: 67108864`, `seeds: [17, 43, 97]`, `fixed_grid: true`, etc.) |

## `search.json`

| Key | Default | Description |
|---|---|---|
| `search.beam_width` | `90` | Beam width retained at each search layer |
| `search.workers` | `8` | Native planner worker threads |
| `search.fixed_model_id` | `gemma4-e2b` | Single candidate model used under `heuristic_requests` mode |
| `search.inference.max_expansions` | `100000` | State expansion budget during online inference search |
| `search.inference.max_seconds` | `60` | Wall-clock time budget (seconds) for inference search |
| `search.inference.layer_pruning` | `pareto` | Default inference layer pruning (`pareto`, `accuracy_time`, or `grouped`) |
| `search.inference.grouped_operator_threshold` | `10` | In `heuristic_value` inference (`layer_pruning="pareto"`), automatically switches to `grouped` pruning when `len(problem.all_ops) > grouped_operator_threshold` (`null` disables) |
| `search.inference.max_structure_groups` | `15` | Maximum logical structure groups retained under `grouped` pruning |
| `search.inference.max_implementations_per_group` | `6` | Maximum physical implementations kept per structure group |
| `search.training.max_expansions` | `1000000` | Expansion budget during training collection search |
| `search.training.max_seconds` | `600` | Wall-clock time budget (seconds) for training search |

## `inference.json`

| Key | Default | Description |
|---|---|---|
| `inference.plan_generation` | `heuristic_value` | Default planning policy (`heuristic_requests` or `heuristic_value`) |
| `inference.preference` | `fastest` | Objective preference among constraint-satisfying plans (`fastest` or `accurate`) |
| `inference.reference_evaluation_run` | path | Reference evaluation run used for offline replay comparisons |
| `inference.reference_aggregate_manifest` | path | Offline replay aggregate accuracy manifest path |
| `inference.device` | `auto` | PyTorch device for value-model inference (`auto`, `cpu`, `cuda[:idx]`) |
| `inference.gpu_batch_size` | `512` | Batch size for batched value-model state scoring |

## `runtime_adaptation.json`

This standalone file is passed via `--runtime-adaptation-config` (not included in `config.json`).

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable progress-scheduled runtime adaptation at chunk boundaries |
| `beam_width` | `12` | Background runtime re-optimization beam width |
| `device` | `cuda` | Device for runtime value-model and cache-token MLP inference |
| `selectivity_threshold` | `0.3` | Relative selectivity drift threshold to trigger plan updates |
| `cache_threshold` | `0.2` | Relative cache-reuse drift threshold to trigger plan updates |
| `cache_trigger_fraction` | `0.3333...` | Minimum input progress fraction before checking adaptation triggers |
| `epsilon` | `1e-08` | Numerical floor for relative drift normalization |
| `cache_predictor_checkpoint` | path | Path to trained `cache_tokens_mlp.pt` checkpoint |
| `max_switches` | `2` | Maximum runtime plan switch checkpoints (`i / (max_switches + 1)`) per query |

## `training.json`

| Key | Default | Description |
|---|---|---|
| `training.dataset` | `null` | Restrict training to one dataset or `null` for all 5 datasets |
| `training.teacher_config` | `teacher.magpie.json` | Teacher configuration used by training/pipeline scripts |
| `training.workload_manifest` | `../experiments/semcache/queries/workloads.json` | Frozen training/testing workload manifest |
| `training.reference_answers_dir` | path | Directory of pre-collected Teacher reference answers |
| `training.queries_per_dataset` | `9/15/12/3/21` | Training query counts for `appstore`, `estate`, `movies_nirvana`, `movies_sema`, `tpch` |
| `training.train_families` / `max_relations` | `null` | Optional query-family filter and relation-count cap |
| `training.minimum_distinct_semantic_inputs` | `128` | Minimum distinct semantic inputs expected in training queries |
| `training.minimum_result_rows` / `maximum_result_rows` | `30` / `300` | Target result row bounds for training workload checks |
| `training.pretrain_rounds` / `rounds` | `1` / `3` | Pretraining and feedback collection round counts |
| `training.collect_only` / `evaluate_selected` | `false` / `false` | Collection-only mode and post-selection evaluation flags |
| `training.feedback_cycle` | list (3 entries) | Per-round `plan_generation` and `query_order` schedule |
| `training.feedback_rollouts_per_query` | `2` | Rollouts executed per query per round (must be `2`) |
| `training.repeats` | `1` | Execution repeats per rollout |
| `training.timeout` / `minimum_timeout` / `timeout_slack` | `300` / `60` / `2` | Adaptive per-execution timeout bounds (seconds) and slack multiplier |
| `training.estimated_timeout` | `1800` | Fallback query execution timeout (seconds) |
| `training.network_retry` | `max_retries: 2`, `backoff_seconds: 5` | Retry policy for transient network failures during collection |
| `training.epochs` / `round_update` / `checkpoint_epochs` | `64`, `{96,256,16,0.03}`, `[4,8,16,32,48,64]` | Value-model training epoch counts, early-stopping criteria, and saved checkpoints |
| `training.seed` / `exploration_fraction` | `17` / `0.2` | Training RNG seed and random rollout exploration rate |

## `value_model.json`

| Key | Default | Description |
|---|---|---|
| `value_model.width` | `64` | Hidden embedding dimension (`width` must be divisible by `heads`) |
| `value_model.heads` | `4` | Attention head count in the graph value network |
| `value_model.layers` | `3` | Message-passing / transformer layers in the value network |
| `value_model.use_cache_features` | `true` | Include prefix-cache reuse features as network inputs |

## `learning.json`

| Key | Default | Description |
|---|---|---|
| `learning.device` / `threads` | `cuda` / `16` | Training PyTorch device and CPU thread count |
| `learning.batch_size` | `64` | Minibatch size during value-model fitting |
| `learning.learning_rate` / `warmup_fraction` / `patience` | `0.0003` / `0.1` / `3` | Optimizer learning rate, linear warmup fraction, and scheduler patience |
| `learning.balance_datasets` | `true` | Weight training samples to balance across datasets |
| `learning.discount` / `time_scale` | `1` / `1` | Return discount factor and execution-time loss scale |
| `learning.accuracy_weight` | `1` | Loss weight on accuracy target predictions |
| `learning.censor_weight` / `estimated_censor_weight` | `1` / `0.1` | Loss weights for timed-out/censored complete and estimated observations |
| `learning.complete_state_weight` / `partial_state_weight` | `0.5` / `0.5` | Loss weights for complete plan states vs. intermediate partial states |

## `reference_answers.json`

| Key | Default | Description |
|---|---|---|
| `reference_answers.semcache.verify_on_second_use` | `true` | Enable second-use cache confirmation during reference collection |
| `reference_answers.semcache.parallelism` | `16` | Per-model HTTP concurrency used when collecting Teacher references |
| `reference_answers.semcache.cpu_parallelism` | `16` | DuckDB CPU threads during reference execution |
| `reference_answers.semcache.max_output_tokens` | `2048` | Max tokens per Teacher call during reference generation |
| `reference_answers.semcache.timeout_seconds` | `1800` | Per-query timeout (seconds) for reference execution |

## `monitoring.json`

| Key | Default | Description |
|---|---|---|
| `monitoring.refresh_seconds` | `10` | Interval (seconds) for refreshing progress Markdown/JSON reports |
| `monitoring.health_interval` | `120` | Health-check logging interval (seconds) |
| `monitoring.stalled_after` | `900` | Duration (seconds) without progress before marking a job stalled |
| `monitoring.stop_query_after` | `0` | Optional per-query watchdog limit (seconds; `0` disables) |

## Teacher Configurations (`teacher/`)

- `teacher.example.json`: Template for external OpenAI-compatible Teacher providers (`base_url`, `model`, `api_key_env`, `thinking`, `reasoning_effort`, `structured_output`, `max_output_tokens`, `connect_timeout_ms`, plus optional `headers`).
- `teacher.opencode.json`, `teacher.deepseek.json`, `teacher.magpie.json`: Provider-specific Teacher endpoint files selected via `teacher_config` or `--teacher-config`.
- Non-loopback endpoints and external Teacher configs marked by `TeacherModel.transport_models` are routed through the independent external concurrency scope (`Scope::External`), separate from the local `--global-http-limit` pool.
