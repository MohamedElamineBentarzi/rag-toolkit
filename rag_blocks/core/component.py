"""Component: the common base of every swappable building block.

Why one base class?
-------------------
Every stage implementation (a parser, an OCR engine, a chunker, a reranker…)
needs the same plumbing: a (kind, name) identity for the registry, a typed
config, and a stable *fingerprint*. Centralizing that here keeps each concrete
component focused purely on its algorithm (Single Responsibility), and it
gives the evaluation suite a uniform way to describe/cache/compare components
without knowing anything about them (Dependency Inversion: the tuner depends
on `Component`, never on `DoclingParser`).

The fingerprint is the quiet superpower: sha256(kind | name | version |
canonical-config). Two pipeline variants that share the same parser config
share the same fingerprint ⇒ the tuner can reuse the cached parse output
instead of re-parsing 5 GB of PDFs per combination. Bump `version` whenever
you change a component's behavior so stale caches invalidate themselves.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from abc import ABC
from enum import Enum
from typing import Any, ClassVar, Optional, Type

from .errors import ConfigError


def _plain(value: Any) -> Any:
    """Normalize config values to log/JSON-friendly primitives.

    Enums become their `.value` (so `OcrPolicy.AUTO` logs as "auto"), and
    containers are walked recursively. Keeps describe() output — and thus
    trial logs — clean and diffable.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value

__all__ = ["Component"]

# Config field names containing these substrings are redacted from
# describe()/fingerprint(): secrets must never leak into logs or cache keys,
# and rotating an API key must not invalidate caches.
_SECRET_MARKERS = ("key", "token", "secret", "password", "credential")


class Component(ABC):
    """Base class for every pluggable building block.

    Subclass contract:
        kind:    the stage slot this fills ("parser", "ocr", "chunker", ...)
        name:    unique name within the kind ("docling", "mistral", ...)
        version: bump on behavior changes (cache invalidation)
        Config:  optional dataclass type describing the configuration
    """

    kind: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    Config: ClassVar[Optional[Type[Any]]] = None

    #: The resolved config instance (or None when Config is undeclared).
    #: Declared as Any so subclasses read `self.config.<field>` without mypy
    #: narrowing it to `Any | None` at every access — the config's real shape
    #: is the nested `Config` dataclass, checked where it is constructed.
    config: Any

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        if self.Config is None:
            if config is not None or overrides:
                raise ConfigError(
                    f"{type(self).__name__} takes no configuration"
                )
            self.config = None
            return

        if config is not None and not isinstance(config, self.Config):
            raise ConfigError(
                f"{type(self).__name__} expected config of type "
                f"{self.Config.__name__}, got {type(config).__name__}"
            )

        # Ergonomics: accept a ready Config, keyword overrides, or both.
        # `MyParser(page_batch_size=4)` beats building a config object by hand.
        base = config if config is not None else self._default_config()
        try:
            self.config = (
                dataclasses.replace(base, **overrides) if overrides else base
            )
        except TypeError as exc:  # unknown field name → clear error, fail fast
            raise ConfigError(f"{type(self).__name__}: {exc}") from exc

    def _default_config(self) -> Any:
        try:
            return self.Config()  # type: ignore[misc]
        except TypeError as exc:
            raise ConfigError(
                f"{type(self).__name__}.Config has required fields; "
                f"pass them explicitly: {exc}"
            ) from exc

    # -- identity ------------------------------------------------------------

    def describe(self) -> dict:
        """Loggable, secret-free description of this exact component setup.

        This dict is what lands in every evaluation Trial record, so a result
        is always reproducible from its log line alone.
        """
        cfg: dict[str, Any] = {}
        if self.config is not None:
            for k, v in dataclasses.asdict(self.config).items():
                lowered = k.lower()
                if any(m in lowered for m in _SECRET_MARKERS):
                    cfg[k] = "<redacted>"
                else:
                    cfg[k] = _plain(v)
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "config": cfg,
        }

    def fingerprint(self) -> str:
        """Stable hash of (kind, name, version, config) — the cache key."""
        canonical = json.dumps(self.describe(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def __repr__(self) -> str:  # helpful in logs and notebooks
        return f"<{type(self).__name__} {self.kind}:{self.name} v{self.version}>"
