"""Prepare GAT inputs from one native layer without executing planner hooks.

The native engine exports ``states=[{state, statistics}]`` and one live
``context={profile, history, version}``. Statistics use string logical node IDs;
state handles retain topology and the owning field registry. Operator evidence is frozen before search; C++ propagates estimates and cache
relations for each candidate without sampling.
"""

import json
from types import SimpleNamespace

from planning.cost.physical_work import PhysicalWorkContext
from planning.cost.row_time import plan_time, time_predictions
from planning.learning.features.cache import (
    bind_prompt_features,
    build_history_feature_index,
    cache_relations,
    history_features,
    static_prompt_features,
)
from planning.learning.features.remaining import (
    remaining_models,
    remaining_numeric,
    remaining_statistics,
)
from planning.learning.features.schema import (
    KINDS,
    STATISTICS,
    description,
    field_identity,
    fields,
    first_fields,
    implementation,
    kind,
    known,
    log_value,
)
from planning.learning.features.statistics import (
    add_input_time,
    estimated_feature,
    statistics,
    timing_kind,
)
from planning.observations.accuracy import evidence as accuracy_evidence
from planning.observations.statistics import prefix_length
from planning.state.partial_plans import descendants

from ..state import canonical


class RawFeatureContext:
    """Static query descriptors prepared once, independently of sample values."""

    def __init__(self, problem):
        problem.native_problem
        self._native_domain = problem.domain
        self.field_buckets = tuple(
            field_identity(value) for value in problem._native_owner.fields
        )
        self.logical = problem.logical
        self.operators = problem.all_ops
        self.semantic = problem.ops
        self.physical_work_context = PhysicalWorkContext(
            self.operators, self.logical, problem._native_owner.fields
        )
        self.by_node = {op["node_id"]: op for op in problem.ops.values()}
        self.profile = problem.profile
        self.history = problem.cache_history
        self._parsed_actions = {}
        self._action_variant_keys = {}
        self._empty_statistics = {}
        self._numeric_statistics = {}
        self._work_features = {}
        self._descendant_paths = {}
        self.descriptors = {}
        self.variants = {}
        for nid, logical in self.logical.items():
            op = self.by_node.get(nid)
            label = kind(op or logical)
            self.descriptors[nid] = {
                "fields": fields(logical, op),
                "description": description(logical, op),
                "kind": [float(label == value) for value in KINDS],
            }
        for oid, domain in problem.domain.items():
            op = problem.ops.get(oid)
            for action in domain:
                self.variants[self.variant_key(action)] = (
                    implementation(action),
                    first_fields(op, action) if op else [[], []],
                    static_prompt_features(op, action, self.profile)
                    if op
                    else (None, None),
                )

    @staticmethod
    def variant_key(action):
        return canonical(
            {
                key: value
                for key, value in action.items()
                if key not in ("inputs", "pair_for")
            }
        )

    def actions(self, row):
        if "actions" not in row:
            actions = []
            for value in row["state"].history:
                if value not in self._parsed_actions:
                    self._parsed_actions[value] = json.loads(value)
                actions.append(self._parsed_actions[value])
            row["actions"] = actions
        return row["actions"]

    def action_variant_key(self, action):
        identity = id(action)
        if identity not in self._action_variant_keys:
            # Parsed actions remain owned for this search, so their identities
            # cannot be recycled while a cached variant key is in use.
            self._action_variant_keys[identity] = self.variant_key(action)
        return self._action_variant_keys[identity]

    @staticmethod
    def nodes(row):
        if "nodes" not in row:
            row["nodes"] = row["state"].nodes()
        return row["nodes"]

    def context(self, batch):
        context = batch.get("context", {})
        return {**self.profile, **context.get("profile", {})}, context.get(
            "history", self.history
        )

    @staticmethod
    def copy_description(value):
        return value[0], list(value[1])

    def numeric_statistics(self, stat, semantic):
        # Most model/layout candidates share the same small scalar estimate.
        # Include every input consumed by estimated_feature/feature_reliability;
        # field sketches and plan identities do not affect these numbers.
        validity = stat.get("valid", {})
        reliability = stat.get("reliability", {})
        intervals = stat.get("intervals", {})

        def typed(value):
            # bool and int compare equal in Python but known() distinguishes
            # them. Keep those types distinct in the memo as well.
            if isinstance(value, (list, tuple)):
                return type(value), tuple(typed(item) for item in value)
            return type(value), value

        key = (
            semantic,
            typed(stat.get("effective_samples")),
            bool(stat.get("uncertain")),
            bool(stat.get("low_survival_uncertain")),
            bool(stat.get("shared_selectivity")),
            typed(stat.get("estimated_selectivity")),
            tuple(
                (
                    typed(stat.get(name)),
                    typed(validity.get(name, True)),
                    typed(reliability.get(name)),
                    typed(intervals.get(name)),
                )
                for name in (*STATISTICS, "l_self")
            ),
        )
        try:
            cached = self._numeric_statistics.get(key)
        except TypeError:
            return statistics(stat, semantic) + estimated_feature(
                stat, "l_self", applicable=semantic
            )
        if cached is None:
            cached = statistics(stat, semantic) + estimated_feature(
                stat, "l_self", applicable=semantic
            )
            self._numeric_statistics[key] = cached
        return cached

    def features(self, batch):
        profile, history = self.context(batch)
        history_context = SimpleNamespace(
            cache_history=history,
            history_feature_index=build_history_feature_index(history),
        )
        node_features = {}
        return [
            self.graph(row, profile, history_context, node_features)
            for row in batch["states"]
        ]

    def graph(self, raw, profile, history_context, node_features=None):
        if node_features is None:
            node_features = {}
        handle = raw["state"]
        actions = {action["operator_id"]: action for action in self.actions(raw)}
        estimates = raw["statistics"]
        nodes = self.nodes(raw)
        rows, by_operator, semantic_estimates = {}, {}, {}
        for node in nodes:
            nid, oid = node.node_id, node.operator_id
            op = self.by_node.get(nid)
            action = actions.get(oid)
            descriptor = self.descriptors[nid]
            source_stat = estimates.get(
                str(nid), estimates.get(nid, self._empty_statistics)
            )
            cache_key = (node.handle_id, id(source_stat))
            cached = node_features.get(cache_key)
            if cached is not None:
                template, stat = cached
                rows[nid] = {**template, "l_previous": {}}
                if op and action:
                    by_operator[oid] = nid
                    semantic_estimates[oid] = stat
                continue
            stat = dict(source_stat)
            variant, first, static_prompt = (
                self.variants[self.action_variant_key(action)]
                if action
                else (implementation(None), [[], []], (None, None))
            )
            if not action:
                stat = {
                    "input_rows": self.logical[nid].get("estimated_rows"),
                    "output_rows": self.logical[nid].get("estimated_rows"),
                    **profile.get("node_statistics", {}).get(str(nid), {}),
                    **stat,
                }
                if stat.get("input_width") is None:
                    stat["input_width"] = stat.get("output_width")
            if op and action:
                bind_prompt_features(stat, *static_prompt)
                quality = accuracy_evidence(
                    profile, action["model_id"], op["kind"], action["layout"]
                )
                if isinstance(quality, dict):
                    stat["accuracy"] = quality.get("value")
                elif known(quality):
                    stat["accuracy"] = quality
                else:
                    stat.pop("accuracy", None)
                stat["reliability"] = dict(stat.get("reliability", {}))
                count = (
                    quality.get("effective_samples")
                    if isinstance(quality, dict)
                    else None
                )
                if known(count) and count > 0:
                    stat["reliability"]["accuracy"] = count / (count + 1)
                by_operator[oid] = nid
                semantic_estimates[oid] = stat
            add_input_time(
                stat,
                op or self.logical[nid],
                profile,
                action.get("model_id") if action else None,
            )
            if action and "model_id" in action:
                variant = list(variant)
                length = stat.get("pre_length")
                variant[-1] = log_value(
                    length if length is not None else prefix_length(stat.get("pre"))
                )
            rows[nid] = {
                "fields": (
                    list(descriptor["fields"][0]),
                    sorted({self.field_buckets[index] for index in node.field_ids}),
                ),
                "description": self.copy_description(descriptor["description"]),
                "first_fields": [list(side) for side in first],
                "numeric": descriptor["kind"]
                + variant
                + self.numeric_statistics(stat, bool(op)),
                "l_existing": history_features(history_context, action, stat)
                if op and action
                else None,
                "children": list(node.children),
                "l_previous": {},
            }
            node_features[cache_key] = (
                {key: value for key, value in rows[nid].items() if key != "l_previous"},
                stat,
            )
        edges = tuple((node.node_id, tuple(node.children)) for node in nodes)
        if edges not in self._descendant_paths:
            self._descendant_paths[edges] = descendants(edges)
        paths = self._descendant_paths[edges]
        cache_relations(rows, by_operator, actions, semantic_estimates, paths)
        remaining = tuple(handle.remaining)
        work_key = (
            remaining,
            tuple(
                tuple(
                    remaining_numeric(self.by_node.get(nid), self.logical[nid], profile)
                )
                for nid in remaining
            ),
        )
        if work_key not in self._work_features:
            work = {}
            for nid in remaining:
                descriptor = self.descriptors[nid]
                work[nid] = {
                    "fields": tuple(list(side) for side in descriptor["fields"]),
                    "description": self.copy_description(descriptor["description"]),
                    "numeric": remaining_numeric(
                        self.by_node.get(nid), self.logical[nid], profile
                    ),
                }
            self._work_features[work_key] = work
        work = self._work_features[work_key]
        return {"nodes": rows, "work": work}

    def native_features(self, model, *, workers=1):
        """Build immutable descriptor tables for the built-in CUDA encoder."""
        from planning.learning.features.schema import FEATURE_GROUPS
        from planning.learning.model import PlanValueModel

        from . import _search_core

        if (
            type(model) is not PlanValueModel
            or model.encoder.words.weight.device.type != "cuda"
            or set(model.encoder.feature_groups)
            not in (set(FEATURE_GROUPS), set(FEATURE_GROUPS) - {"cache"})
            or "cache" not in model.encoder.feature_groups
            and not getattr(_search_core, "SUPPORTS_CACHE_FEATURE_ABLATION", False)
            or not hasattr(_search_core, "NativeFeatureContext")
        ):
            return None
        descriptors = {}
        for nid, descriptor in self.descriptors.items():
            descriptors[str(nid)] = {
                **descriptor,
                "semantic": nid in self.by_node,
                "timing_kind": timing_kind(self.by_node.get(nid) or self.logical[nid]),
                "remaining_models": [
                    {"model_id": model_id, "slot": slot}
                    for model_id, slot in remaining_models(
                        self.by_node.get(nid), self.profile
                    ).items()
                ],
                "estimated_rows": self.logical[nid].get("estimated_rows"),
                "remaining_statistics": remaining_statistics(
                    self.by_node.get(nid), self.logical[nid], self.profile
                ),
            }
        variants = {}
        for oid, domain in self._native_domain.items():
            op = self.semantic.get(oid)
            rows = []
            for index, action in enumerate(domain):
                key = self.variant_key(action)
                impl, first, prompt = self.variants[key]
                rows.append(
                    {
                        "index": index,
                        "implementation": impl,
                        "first_fields": first,
                        "pre": prompt[0],
                        "pre_length": prompt[1],
                        "has_model": "model_id" in action,
                        "model": action.get("model_id", ""),
                        "kind": op["kind"] if op else "",
                        "layout_key": ":".join(
                            action.get("layout", {}).get("block_order", ())
                        ),
                    }
                )
            variants[str(oid)] = rows
        return _search_core.NativeFeatureContext(
            {
                "chunk_size": model.inference_batch_size,
                "workers": workers,
                "use_cache_features": "cache" in model.encoder.feature_groups,
                "descriptors": descriptors,
                "variants": variants,
                "field_buckets": self.field_buckets,
                "profile": {
                    key: self.profile[key]
                    for key in (
                        "node_statistics",
                        "request_profiles",
                        "accuracy",
                        "operator_time_calibration",
                        "relational_profiles",
                    )
                    if key in self.profile
                },
                "history": [
                    {key: value for key, value in row.items() if key != "summary"}
                    for row in self.history
                ],
            }
        )

    def predict(self, model, batch):
        if not isinstance(batch, dict):
            return model.completion_native_prediction(batch)
        if hasattr(model, "completion_prediction"):
            return model.completion_prediction(self.features(batch))
        return [
            (-seconds, accuracy)
            for seconds, accuracy in model.values_features(self.features(batch))
        ]

    def request_prediction(self, batch):
        """Rank using shared per-input calibration without sampling in the loop."""
        from planning.cost.requests import request_count

        profile, _ = self.context(batch)

        def estimate(raw):
            estimates = raw["statistics"]
            return plan_time(
                nodes=self.nodes(raw),
                actions=self.actions(raw),
                operators=self.operators,
                semantic=self.semantic,
                logical=self.logical,
                profile=profile,
                physical_context=self.physical_work_context,
                statistics=lambda node: estimates.get(
                    str(node.node_id), estimates.get(node.node_id, {})
                ),
            )

        def requests(raw):
            estimates = raw["statistics"]
            return request_count(
                nodes=self.nodes(raw),
                operators=self.operators,
                semantic=self.semantic,
                logical=self.logical,
                profile=profile,
                statistics=lambda node: estimates.get(
                    str(node.node_id), estimates.get(node.node_id, {})
                ),
            )

        return time_predictions(
            batch["states"],
            estimate,
            request_fallback=requests
            if profile.get("cardinality_only_sources")
            else None,
        )
