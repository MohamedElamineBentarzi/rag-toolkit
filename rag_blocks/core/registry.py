"""Registry: the single extension point of the whole toolkit.

Pattern: Registry + Factory Method. A global mapping of
(kind, name) → Component class, filled by a class decorator, consumed by
`registry.create(...)`.

Why this matters (Open/Closed Principle in practice):
    - Adding a new OCR engine or parser requires ZERO changes to core code.
      You write a class, decorate it, done. The library is open for
      extension, closed for modification.
    - Pipelines become *data*: `{"parser": "docling", "ocr": "mistral"}` is a
      complete, serializable spec. That's exactly what the auto-tuner needs —
      it enumerates configs, not code.
    - Third-party packages can ship components via Python entry points
      (`rag_blocks.components` group) and they appear here automatically —
      a plugin ecosystem with no plugin framework.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Any, Optional, Type, TypeVar

from .component import Component
from .errors import ComponentNotFoundError, DuplicateComponentError

__all__ = ["Registry", "registry"]

_log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "rag_blocks.components"

#: Preserves the decorated class's exact type through @register, so type
#: checkers see `HashingEmbedder`, not `Component` — otherwise every registered
#: class collapses to `type[Component]` and can't be passed where its own base
#: (Embedder, VectorStore, ...) is expected.
_C = TypeVar("_C", bound=Component)


class Registry:
    """Maps (kind, name) → Component class and builds instances on demand."""

    def __init__(self) -> None:
        self._components: dict[tuple[str, str], Type[Component]] = {}
        self._entry_points_loaded = False

    # -- registration --------------------------------------------------------

    def register(self, cls: Type[_C]) -> Type[_C]:
        """Class decorator. Reads identity from the class itself:

            @registry.register
            class MistralOcrEngine(OcrEngine):
                name = "mistral"
                ...

        No decorator arguments: identity lives on the class (one source of
        truth), the decorator only files it away.
        """
        kind = getattr(cls, "kind", None)
        name = getattr(cls, "name", None)
        if not kind or not name:
            raise DuplicateComponentError(
                f"{cls.__name__} must define class attributes 'kind' and 'name' "
                "before it can be registered"
            )
        key = (kind, name)
        existing = self._components.get(key)
        if existing is not None and existing is not cls:
            raise DuplicateComponentError(
                f"A component is already registered under {key}: "
                f"{existing.__name__}"
            )
        self._components[key] = cls
        return cls

    # -- lookup / factory ----------------------------------------------------

    def get(self, kind: str, name: str) -> Type[Component]:
        self._ensure_entry_points()
        try:
            return self._components[(kind, name)]
        except KeyError:
            available = ", ".join(self.available(kind)) or "<none>"
            raise ComponentNotFoundError(
                f"No {kind!r} component named {name!r}. "
                f"Available {kind}s: {available}"
            ) from None

    def create(self, kind: str, name: str, config: Any = None,
               **overrides: Any) -> Component:
        """Factory Method: build a configured instance from its string name."""
        return self.get(kind, name)(config, **overrides)

    def available(self, kind: Optional[str] = None) -> list[str]:
        self._ensure_entry_points()
        if kind is None:
            return sorted(f"{k}:{n}" for k, n in self._components)
        return sorted(n for k, n in self._components if k == kind)

    # -- plugin discovery ----------------------------------------------------

    def _ensure_entry_points(self) -> None:
        """Lazily import third-party components declared as entry points.

        A plugin package only needs, in its pyproject.toml:

            [project.entry-points."rag_blocks.components"]
            my_ocr = "my_pkg.ocr"          # importing the module registers it

        Lazy + fault-tolerant: a broken plugin must never take down the core.
        """
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        try:
            eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception:  # pragma: no cover - stdlib API drift safety net
            return
        for ep in eps:
            try:
                ep.load()  # side effect: module-level @registry.register runs
            except Exception as exc:  # noqa: BLE001 - isolate faulty plugins
                # Never crash core over a third-party plugin, but the failure
                # must be discoverable — a silent drop is an hour of a plugin
                # author's life. (AGENTS.md forbids swallowing without context.)
                _log.warning(
                    "rag_blocks: entry-point plugin %r failed to load: %s",
                    ep.name, exc,
                )


#: The default process-wide registry. A module-level singleton is a pragmatic
#: choice here (like logging's root logger); tests can still instantiate
#: private `Registry()` objects for isolation.
registry = Registry()
