"""Collect native feedback, train completion values and search legal plans (§4.1–4.4)."""

import argparse
import json
from pathlib import Path

from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.settings.configuration import (
    ROOT,
    config_from_argv,
    runtime_path,
    teacher_path,
)
from planning.settings.parallelism import parallelism_argument

from .state import SearchProblem


def read(path):
    return json.loads(Path(path).read_text())


def main():
    config_path, settings = config_from_argv()
    training = settings["training"]
    learning = settings["learning"]
    search = settings["search"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path)
    sub = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = sub.add_parser(
        "benchmark", help="Resumable native-SQL multi-dataset measured training"
    )
    benchmark_parser.add_argument("--run", type=Path, required=True)
    benchmark_parser.add_argument(
        "--build",
        type=Path,
        help="DuckDB build directory containing the matching semantic extension.",
    )
    benchmark_parser.add_argument("--sampling-config", type=Path)
    benchmark_parser.add_argument(
        "--model-config", type=Path, help="Shared candidate-model configuration JSON"
    )
    benchmark_parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Candidate-model output token limit per sample (default: %(default)s).",
    )
    benchmark_parser.add_argument(
        "--teacher-model",
        help="Reference-only teacher model (default: model in the supplied teacher JSON).",
    )
    benchmark_parser.add_argument(
        "--teacher-config",
        type=Path,
        help="Explicit JSON configuration for a reference-only teacher endpoint.",
    )
    benchmark_parser.add_argument(
        "--parallelism", type=parallelism_argument, default=None
    )
    benchmark_parser.add_argument(
        "--model-endpoints",
        "--endpoints",
        dest="endpoints",
        nargs="+",
    )
    benchmark_parser.add_argument(
        "--queries-per-dataset",
        type=int,
        help="Override all dataset counts with one number; otherwise use the per-dataset configuration.",
    )
    benchmark_parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Share of generated queries held out for validation; 0 keeps every query for training.",
    )
    benchmark_parser.add_argument(
        "--families",
        nargs="+",
        dest="train_families",
        help="Restrict generated training queries to these families (default: all training families).",
    )
    benchmark_parser.add_argument(
        "--max-relations",
        type=int,
        help="Cap the number of joined relations per generated query.",
    )
    benchmark_parser.add_argument("--repeats", type=int)
    benchmark_parser.add_argument("--epochs", type=int)
    benchmark_parser.add_argument("--pretrain-rounds", type=int)
    benchmark_parser.add_argument(
        "--collect-only",
        action=argparse.BooleanOptionalAction,
        help="Collect one estimated pretraining round without fitting a model or executing real feedback",
    )
    benchmark_parser.add_argument(
        "--rounds",
        type=int,
        help="Measured feedback rounds; 0 runs estimated pretraining only "
        "(without --evaluate-selected)",
    )
    benchmark_parser.add_argument("--candidates", type=int)
    benchmark_parser.add_argument("--timeout", type=float)
    benchmark_parser.add_argument("--minimum-timeout", type=float)
    benchmark_parser.add_argument("--timeout-slack", type=float)
    benchmark_parser.add_argument("--seed", type=int)
    benchmark_parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and bind all SQL without model inference",
    )
    benchmark_parser.add_argument("--search-seconds", type=float)
    benchmark_parser.add_argument("--max-expansions", type=int)
    benchmark_parser.add_argument(
        "--evaluate-selected",
        action="store_true",
        help="Test the frozen selected text suite after training",
    )
    search_parser = sub.add_parser("search")
    search_parser.add_argument("--query-ir", type=Path, required=True)
    search_parser.add_argument("--profile", type=Path)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--experience", type=Path, nargs="+", required=True)
    train_parser.add_argument(
        "--source", choices=("estimated", "measured"), required=True
    )
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument(
        "--initialize-remaining-attention", type=Path,
        help="Explicitly initialize the new encoder from feature18 v9 weights; fitting is required",
    )
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument(
        "--algorithm",
        choices=("balsa_monte_carlo", "supervised"),
        help="Value-learning target; balsa_monte_carlo propagates terminal returns to prefixes.",
    )
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--checkpoint-epochs", type=int, nargs="*")
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--accuracy-weight", type=float)
    train_parser.add_argument("--censor-weight", type=float)
    train_parser.add_argument("--estimated-censor-weight", type=float)
    train_parser.add_argument("--complete-state-weight", type=float)
    train_parser.add_argument("--partial-state-weight", type=float)
    train_parser.add_argument("--time-scale", type=float)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--patience", type=int)
    train_parser.add_argument("--no-dataset-balance", action="store_true")
    train_parser.add_argument("--threads", type=int)
    train_parser.add_argument("--width", type=int)
    train_parser.add_argument("--heads", type=int)
    train_parser.add_argument("--layers", type=int)
    train_parser.add_argument(
        "--feature-groups",
        nargs="*",
        choices=(
            "task",
            "implementation",
            "statistics",
            "cache",
            "history",
            "structure",
        ),
        help="Enabled feature additions; omit for all, pass without values for the control",
    )
    search_parser.add_argument(
        "--plan-generation",
        choices=("heuristic_requests", "heuristic_value"),
        default=settings["inference"].get("plan_generation", "heuristic_requests"),
    )
    search_parser.add_argument("--checkpoint", type=Path)
    search_parser.add_argument("--output", type=Path, required=True)
    search_parser.add_argument("--beam-width", type=int)
    search_parser.add_argument("--max-expansions", type=int)
    search_parser.add_argument("--max-seconds", type=float)
    for command in (benchmark_parser, search_parser, train_parser):
        command.add_argument("--config", type=Path, default=config_path)
    benchmark_parser.set_defaults(
        build=runtime_path(settings, "build"),
        teacher_config=teacher_path(settings, config_path),
        endpoints=settings["models"]["endpoints"],
        max_output_tokens=settings["models"]["max_output_tokens"],
        candidates=search["beam_width"],
        search_seconds=search["training"]["max_seconds"],
        max_expansions=search["training"]["max_expansions"],
        **{
            key: training[key]
            for key in (
                "queries_per_dataset",
                "validation_fraction",
                "train_families",
                "max_relations",
                "repeats",
                "epochs",
                "pretrain_rounds",
                "collect_only",
                "rounds",
                "seed",
                "timeout",
                "minimum_timeout",
                "timeout_slack",
                "evaluate_selected",
            )
        },
    )
    search_parser.set_defaults(
        beam_width=search["beam_width"],
        max_expansions=search["inference"]["max_expansions"],
        max_seconds=search["inference"]["max_seconds"],
    )
    train_parser.set_defaults(
        **{
            key: learning[key]
            for key in (
                "algorithm",
                "batch_size",
                "learning_rate",
                "accuracy_weight",
                "censor_weight",
                "estimated_censor_weight",
                "complete_state_weight",
                "partial_state_weight",
                "time_scale",
                "patience",
                "threads",
            )
        },
        **settings["value_model"],
        epochs=training["epochs"],
        checkpoint_epochs=training["checkpoint_epochs"],
        seed=training["seed"],
        no_dataset_balance=not learning["balance_datasets"],
    )
    args = parser.parse_args()
    if args.command == "benchmark":
        if min(args.repeats, args.epochs) < 1:
            parser.error("Positive training and timing budgets are required")
        import sys

        sys.path.insert(0, str(ROOT.parent / "experiments/semcache"))
        from training.pipeline import run

        run(args)
        return
    if args.command == "train":
        from .learning.cli import run

        run(args, parser)
        return
    from planning.learning.devices import load_inference_model
    from planning.settings.parallelism import configure_cpu_resources

    configure_cpu_resources(settings, configure_torch=True)

    config = SearchConfig.configured(
        settings,
        args.plan_generation,
        beam_width=args.beam_width,
        max_expansions=args.max_expansions,
        max_seconds=args.max_seconds,
    )
    from planning.settings.modes import profile_for_mode

    query = read(args.query_ir)
    profile = profile_for_mode(
        read(args.profile) if args.profile else {},
        query,
        args.plan_generation,
        fixed_model_id=config.fixed_model_id or "gemma4-e2b",
    )
    problem = SearchProblem(query, profile)
    ranker = (
        load_inference_model(args.checkpoint, settings)
        if not config.request_guided
        else None
    )
    result = ParetoBeamSearch(ranker, config).run(problem)
    result.pop("states")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"Generated {len(result['frontier'])} candidate plans; value_available={result['value_available']}"
    )


if __name__ == "__main__":
    main()
