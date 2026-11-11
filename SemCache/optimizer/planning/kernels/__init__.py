"""Python shims with an optional compiled resident-kernel overlay."""

from pkgutil import extend_path

# Keep the source package importable while allowing the configured native
# overlay (``build/*-bindings/overlay/planning/kernels``) to provide the
# compiled ``_search_core`` module without requiring an import-order side
# effect from the training entry point.
__path__ = extend_path(__path__, __name__)
