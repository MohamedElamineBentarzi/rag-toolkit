"""Every name in every ``__all__`` must actually resolve.

This guards the class of bug where a symbol is listed in ``__all__`` but never
imported into the package namespace — which makes ``from rag_blocks import *``
crash and ``rag_blocks.Name`` raise ``AttributeError`` (regression: the
``EnrichmentError`` export). Parametrized over every *package* ``__init__`` so a
new subsystem is covered automatically.
"""

import importlib
import pkgutil

import pytest

import rag_blocks


def _packages_with_all() -> list:
    """Every package (dir with __init__) under rag_blocks, plus the root.

    Only packages are imported — leaf modules (vendor adapters) are skipped so
    this test never forces an optional SDK import.
    """
    modules = [rag_blocks]
    for info in pkgutil.walk_packages(rag_blocks.__path__, rag_blocks.__name__ + "."):
        if info.ispkg:
            modules.append(importlib.import_module(info.name))
    return [m for m in modules if hasattr(m, "__all__")]


@pytest.mark.parametrize(
    "module", _packages_with_all(), ids=lambda m: m.__name__
)
def test_all_exports_resolve(module) -> None:
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"{module.__name__}.__all__ lists unresolved names: {missing}"
