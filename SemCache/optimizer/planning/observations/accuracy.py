"""Layout-aware semantic accuracy profile lookup."""

from __future__ import annotations


def layout_key(layout):
    """Stable identifier for an allowed O/T/D block order."""
    return ":".join(layout.get("block_order", ()))


def sampling_layouts(op):
    """Compare the three block orders using each operator's bound field order."""
    from planning.prompt_layouts import BLOCK_ORDERS

    return [
        {**op["default_layout"], "block_order": list(order)} for order in BLOCK_ORDERS
    ]


def evidence(profile, model, kind, layout=None):
    """Return the best accuracy evidence without merging different layouts.

    New profiles store ``accuracy[model][kind].layouts[layout_key]``. Old
    profiles retain a scalar/object value and are accepted as a legacy global
    fallback, because their data cannot be reassigned to a layout afterwards.
    """
    value = profile.get("accuracy", {}).get(model, {}).get(kind)
    if not isinstance(value, dict):
        return value
    layouts = value.get("layouts")
    if isinstance(layouts, dict) and layout is not None:
        selected = layouts.get(layout_key(layout))
        if selected is not None:
            return selected
        return None
    if "value" in value:
        return value
    return None


def value(profile, model, kind, layout=None):
    item = evidence(profile, model, kind, layout)
    if isinstance(item, dict):
        return item.get("value")
    return item
