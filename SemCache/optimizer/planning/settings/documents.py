"""Read JSON documents and explicit lists of configuration modules."""

import json
import re
from pathlib import Path


def read_json(path):
    """Read JSON with // comments without altering quoted URLs or escapes."""
    text = Path(path).read_text()
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\r\n]*', lambda match: match[1] or "", text)
    return json.loads(text)


def read_configuration(path):
    """Merge named modules once, rejecting ambiguous or duplicate sections.

    Complete configuration snapshots remain ordinary JSON objects, so saved
    training runs can still pass their frozen settings through --config.
    """
    source = Path(path)
    settings = read_json(source)
    if not isinstance(settings, dict):
        raise ValueError("Optimizer configuration must be a JSON object")
    if "include" not in settings:
        return settings
    modules = settings["include"]
    if (
        settings.keys() != {"include"}
        or not isinstance(modules, list)
        or not modules
        or any(not isinstance(name, str) or not name.strip() for name in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError(
            "Configuration index requires only a distinct, nonempty include list"
        )
    merged = {}
    sources = set()
    for name in modules:
        module_path = (source.resolve().parent / name).resolve()
        if module_path == source.resolve() or module_path in sources:
            raise ValueError(f"Repeated configuration module: {name}")
        sources.add(module_path)
        module = read_json(module_path)
        if not isinstance(module, dict) or not module or "include" in module:
            raise ValueError(
                f"Configuration module must contain sections: {module_path}"
            )
        duplicates = merged.keys() & module.keys()
        if duplicates:
            raise ValueError(
                "Duplicate configuration sections: " + ", ".join(sorted(duplicates))
            )
        merged.update(module)
    return merged
