#!/usr/bin/env python3
"""Local QueryIR -> PlanSpec service backed by the rebuilt section-4 planner."""

import argparse
import json
import socketserver
import sqlite3
import struct
from pathlib import Path
from typing import Any, cast

from planning.search.protocol import validate_spec

__all__ = ["OptimizerServer", "validate_spec"]

MAX_FRAME = 16 * 1024 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_frame(connection):
    """Read one bounded network-order length-prefixed JSON object."""

    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = connection.recv(size - len(result))
            if not part:
                raise ValueError("truncated optimizer frame")
            result.extend(part)
        return result

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError("optimizer frame must be 1..16 MiB")
    return json.loads(exact(size), object_pairs_hook=unique_object)


def write_frame(connection, value):
    body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(body) > MAX_FRAME:
        raise ValueError("optimizer response exceeds 16 MiB")
    connection.sendall(struct.pack("!I", len(body)) + body)


class OptimizerServer(socketserver.UnixStreamServer):
    def __init__(self, path, policy, log_path=None, print_plans=False):
        self.policy: Any = policy
        self.print_plans = print_plans
        self.log_path = Path(log_path) if log_path else None
        super().__init__(str(path), OptimizerHandler)

    def server_close(self):
        try:
            close = getattr(self.policy, "close", None)
            if close is not None:
                close()
        finally:
            super().server_close()

    def record(self, query, response):
        event = {"query_ir": query, "plan_spec": response}
        if self.print_plans:
            print(json.dumps(response, ensure_ascii=False, indent=2), flush=True)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                )


class OptimizerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Always return a framed error when possible. C++ treats it as a planning
        # failure and therefore sends no model requests.
        self.request.settimeout(15)
        try:
            query = read_frame(self.request)
            server = cast(OptimizerServer, self.server)
            response = server.policy.optimize(query)
            server.record(query, response)
            write_frame(self.request, response)
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
            ArithmeticError,
            sqlite3.Error,
        ) as error:
            try:
                write_frame(self.request, {"error": str(error)})
            except OSError:
                pass


def main():
    from planning.settings.configuration import (
        config_from_argv,
        load_sampling_config,
        runtime_path,
        teacher_path,
    )

    config_path, settings = config_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trained checkpoint for heuristic_value or optional runtime adaptation",
    )
    parser.add_argument(
        "--profiles", type=Path, help="JSON profiles keyed by query_key"
    )
    parser.add_argument(
        "--search-log",
        type=Path,
        help="Full Pareto frontier and pruning/feasibility audit",
    )
    from planning.settings.parallelism import parallelism_argument, sampling_parallelism

    parser.add_argument("--parallelism", type=parallelism_argument, default=None)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--preference", choices=["fastest", "accurate"])
    parser.add_argument(
        "--plan-generation",
        choices=("heuristic_requests", "heuristic_value"),
        default=settings["inference"].get("plan_generation", "heuristic_requests"),
    )
    parser.add_argument(
        "--sampling-database",
        type=Path,
        help="Read-only DuckDB snapshot matching the query tables",
    )
    parser.add_argument(
        "--sampling-setup",
        type=Path,
        help="SQL recreating query tables/views in the sampler connection",
    )
    parser.add_argument(
        "--sampling-models",
        type=Path,
        help="semantic_models JSON for online beam sampling",
    )
    parser.add_argument(
        "--build",
        type=Path,
    )
    parser.add_argument(
        "--sampling-config", type=Path, help="SamplingCoordinator budget/config JSON"
    )
    parser.add_argument(
        "--teacher-model",
        help="Reference-only teacher model used for quality calibration.",
    )
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument(
        "--runtime-adaptation-config",
        type=Path,
        help="Optional, default-off progress-scheduled runtime monitoring and implementation selection",
    )
    parser.add_argument(
        "--model-config", type=Path, help="Shared candidate-model configuration JSON"
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=settings["search"]["inference"]["max_expansions"],
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=settings["search"]["inference"]["max_seconds"],
    )
    parser.set_defaults(
        build=runtime_path(settings, "build"),
        teacher_config=teacher_path(settings, config_path),
        beam_width=settings["search"]["beam_width"],
        preference=settings["inference"]["preference"],
    )
    args = parser.parse_args()
    from planning.execution.adaptation_monitor import AdaptationSettings

    adaptation = (
        AdaptationSettings(**json.loads(args.runtime_adaptation_config.read_text()))
        if args.runtime_adaptation_config
        else AdaptationSettings()
    )
    if args.model_config and args.sampling_models:
        parser.error("Use only one of --model-config and --sampling-models")
    from planning.search.beam import SearchConfig
    from planning.search.policy import PlanningPolicy
    from planning.settings.parallelism import configure_cpu_resources

    configure_cpu_resources(settings, configure_torch=True)
    profiles = json.loads(args.profiles.read_text()) if args.profiles else None
    from planning.settings.modes import models_for_mode, sampling_for_mode

    sampler_factory = None
    if (
        args.sampling_models
        or args.model_config
        or args.sampling_database
        or args.sampling_setup
        or not args.profiles
    ):
        from planning.backends.sampling import NativeSamplingFactory
        from planning.settings.teacher import default_teacher_model, load_teacher

        catalog = None
        if args.model_config or not args.sampling_models:
            from planning.settings.model_config import discover_configured_models

            candidate_models, catalog = discover_configured_models(
                args.model_config or config_path
            )
        else:
            candidate_models = json.loads(args.sampling_models.read_text())
        teacher = load_teacher(
            args.teacher_model or default_teacher_model(args.teacher_config),
            candidate_models,
            args.teacher_config,
        )
        candidate_models, catalog = models_for_mode(
            candidate_models, catalog or {}, settings, args.plan_generation
        )
        sampler_factory = NativeSamplingFactory(
            args.build,
            candidate_models,
            database=args.sampling_database,
            setup=args.sampling_setup.read_text() if args.sampling_setup else "",
            config=sampling_for_mode(
                sampling_parallelism(
                    load_sampling_config(args.sampling_config, settings),
                    override=args.parallelism,
                ),
                args.plan_generation,
                fixed_model_id=settings["search"].get("fixed_model_id", "gemma4-e2b"),
            ),
            teacher=teacher,
            catalog=catalog,
        )
    elif args.sampling_database or args.sampling_setup or args.sampling_config:
        parser.error("online sampling requires --model-config or --sampling-models")
    policy = PlanningPolicy(
        args.checkpoint,
        profiles,
        SearchConfig.configured(
            settings,
            args.plan_generation,
            beam_width=args.beam_width,
            max_expansions=args.max_expansions,
            max_seconds=args.max_seconds,
        ),
        args.preference,
        args.search_log,
        sampler_factory=sampler_factory,
        inference_settings=settings,
        runtime_adaptation=adaptation,
    )
    # bind fails if a path already exists. Never unlink another process's socket.
    with OptimizerServer(args.socket, policy, args.log, print_plans=True) as server:
        identity = args.socket.stat()
        try:
            print(f"Optimizer listening on {args.socket}", flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if args.socket.exists() and args.socket.stat().st_ino == identity.st_ino:
                args.socket.unlink()


if __name__ == "__main__":
    main()
