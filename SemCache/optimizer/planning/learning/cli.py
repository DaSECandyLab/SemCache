"""Training command orchestration; algorithms live in training.py."""

import json

from .model import PlanValueModel


def read(path):
    return json.loads(path.read_text())


def run(args, parser):
    import torch

    from planning.learning.training.config import TrainingConfig
    from planning.learning.training.data import dataset, validate_dataset
    from planning.learning.training.engine import train

    if args.threads < 1:
        parser.error("--threads must be positive")
    from planning.settings.configuration import load_config
    from planning.settings.parallelism import configure_cpu_resources

    settings = load_config(args.config)
    settings["learning"]["threads"] = min(
        args.threads, settings["runtime"]["max_cpu_threads"]
    )
    configure_cpu_resources(settings, configure_torch=True)
    device = torch.device(settings["learning"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA value-model training requested, but CUDA is unavailable"
        )
    torch.manual_seed(args.seed)
    if args.initialize_remaining_attention and (args.checkpoint or args.feature_groups is not None):
        parser.error("Attention initialization cannot be combined with --checkpoint or --feature-groups")
    if args.checkpoint and args.feature_groups is not None:
        parser.error("A loaded checkpoint already fixes its feature groups")
    if args.checkpoint:
        model = PlanValueModel.load(args.checkpoint)
    elif args.initialize_remaining_attention:
        from .initialization import initialize_remaining_attention

        model = initialize_remaining_attention(args.initialize_remaining_attention)
    else:
        model = PlanValueModel(
            width=args.width,
            heads=args.heads,
            layers=args.layers,
            time_scale=args.time_scale,
            **(
                {"feature_groups": args.feature_groups}
                if args.feature_groups is not None
                else {}
            ),
        )
    initialization = model.training_info
    model.to(device)
    print(f"Value model training device: {device}", flush=True)
    config = TrainingConfig(
        algorithm=args.algorithm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_epochs=args.checkpoint_epochs,
        learning_rate=args.learning_rate,
        accuracy_weight=args.accuracy_weight,
        censor_weight=args.censor_weight,
        estimated_censor_weight=args.estimated_censor_weight,
        complete_state_weight=args.complete_state_weight,
        partial_state_weight=args.partial_state_weight,
        time_scale=args.time_scale,
        seed=args.seed,
        patience=args.patience,
        balance_datasets=not args.no_dataset_balance,
        warmup_fraction=settings["learning"]["warmup_fraction"],
        discount=settings["learning"]["discount"],
        estimated_timeout=settings["training"]["estimated_timeout"],
    )
    inputs = [read(path) for path in args.experience]
    for payload in inputs:
        validate_dataset(payload)
    data = (
        inputs[0]
        if len(inputs) == 1
        else dataset([r for payload in inputs for r in payload["records"]])
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = train(
        model,
        data,
        source=args.source,
        config=config,
        progress=lambda message: print(message, flush=True),
        checkpoint_dir=args.output / "checkpoints",
    )
    report["device"] = str(device)
    if initialization and initialization.get("requires_fitting"):
        report["initialization"] = initialization
        model.training_info["initialization"] = initialization
    report["experience_files"] = [str(path) for path in args.experience]
    model.save(args.output / "value.pt")
    args.output.joinpath("training.json").write_text(json.dumps(report, indent=2))
    print(f"Trained {report['queries']} queries; checkpoint={args.output / 'value.pt'}")
