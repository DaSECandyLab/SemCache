"""The two shared plan-generation policies and their sampling contracts."""

from planning.search.heuristics import fixed_layout

REQUESTS = "heuristic_requests"
VALUE = "heuristic_value"
MODES = (REQUESTS, VALUE, "value")
DEFAULT_FIXED_MODEL = "gemma4-e2b"


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(
            "plan_generation must be heuristic_requests or heuristic_value"
        )
    return mode


def configured_fixed_model(settings):
    name = settings.get("search", {}).get("fixed_model_id", DEFAULT_FIXED_MODEL)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("search.fixed_model_id must be a nonempty model identifier")
    return name


def models_for_mode(models, catalog, settings, mode, *, fixed_model_id=None):
    """Restrict transport candidates before EXPLAIN as well as Beam expansion."""
    validate_mode(mode)
    catalog = catalog or {}
    if mode != REQUESTS:
        return dict(models), dict(catalog)
    name = fixed_model_id or configured_fixed_model(settings)
    if name not in models:
        raise ValueError(f"Fixed planning model is unavailable: {name}")
    return {name: models[name]}, {name: catalog[name]} if name in catalog else {}


def profile_for_mode(profile, query, mode, *, fixed_model_id=DEFAULT_FIXED_MODEL):
    """Freeze implementation domains without weakening bound-field dependencies."""
    validate_mode(mode)
    from planning.state.copying import copy_profile

    result = copy_profile(profile)
    if result.get("sampling_strategy") != "independent_base":
        result.pop("statistics", None)
        result.pop("operator_statistics", None)
    result.update(plan_generation=mode, sampling_strategy="independent_base")
    result.pop("conditional_inputs", None)
    result.pop("prefix_statistics", None)
    if mode == REQUESTS:
        if not isinstance(fixed_model_id, str) or not fixed_model_id.strip():
            raise ValueError("The request-count policy requires a fixed model")
        if result.get("constraints", {}).get("a_min") is not None:
            raise ValueError(
                "heuristic_requests does not support an accuracy constraint; "
                "use heuristic_value"
            )
        result["fixed_model_id"] = fixed_model_id
        result.setdefault("constraints", {})["allowed_models"] = [fixed_model_id]
        result["constraints"].pop("a_min", None)
        result["layouts"] = {
            str(op["operator_id"]): [fixed_layout(op)]
            for op in query.get("semantic_operators", [])
        }
    else:
        result.pop("fixed_model_id", None)
        result.pop("layouts", None)
    return result


def sampling_for_mode(config, mode, *, fixed_model_id=DEFAULT_FIXED_MODEL):
    """Internal flags follow the selected policy rather than adding more modes."""
    validate_mode(mode)
    result = dict(config or {})
    result.update(
        plan_generation=mode,
        operator_sample_rows=result.get("operator_sample_rows", 20),
        relational_sample_rows=result.get("relational_sample_rows", 200),
        sample_depth_selectivity=mode != REQUESTS,
        ensure_prefix_depth_rates=mode != REQUESTS
        and result.get("ensure_prefix_depth_rates", True),
        sample_accuracy=mode != REQUESTS,
    )
    if mode == REQUESTS:
        result["fixed_model_id"] = fixed_model_id
    else:
        result.pop("fixed_model_id", None)
    return result
