"""The Studio manifest: the registry introspected into UI-ready data.

Studio (the optional React app under `studio/app`) is a *static* site — it holds
no Python. Everything it knows about the library (which blocks exist, each one's
ports, its params, its docs) comes from the dict this module builds. It is the
one bridge between the components and the canvas, and it is *generated*, never
hand-written: the whole library is Open/Closed via the registry (AGENTS.md §2.4),
so a newly registered component must appear in the UI with zero edits. The code
*is* the source, read the same way `PipelineBuilder` reads it.

Two callers:
- `rag_blocks.studio.server` serves `build_manifest()` fresh at launch, so
  `rag-blocks studio` reflects whatever components *this* install has (including
  third-party plugins) — better than any checked-in file.
- `studio/tools/build_manifest.py` writes it to `studio/app/public/blocks.json`
  for local `npm run dev`.

Stdlib + rag_blocks only; adds no third-party dependency to the core.
The emitted shape is documented in `studio/app/src/manifest/types.ts`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import inspect
import typing
from typing import Any

import rag_blocks  # noqa: F401  (import side effect: registers all built-ins)
from rag_blocks.core.component import Component
from rag_blocks.core.registry import registry
from rag_blocks.evaluation.space import CHAIN_STAGES, SPEC_KINDS
from rag_blocks.storage.base import BlobStore
from rag_blocks.storage.vector_store import VectorStore

#: Storage backends are *optional injected infra*, not composition: a component
#: that takes one (BM25Index's optional BlobStore for persistence) runs fine
#: without it — the flat spec just omits it. So, unlike a Retriever/Callable
#: dependency, a storage-backend param must NOT mark a component non-exportable.
_STORAGE_BASES = (BlobStore, VectorStore)

#: Constructor parameters that are wiring, not settable spec params: `self` and
#: the two the builder supplies itself (`config`, the live `index`).
_WIRING_PARAMS = frozenset({"self", "config", "index"})

#: Composition the builder wires from *nested sub-specs* (`inner`, `retrievers`)
#: or the generator's LLM seam (`complete`) — so these params don't block export
#: and aren't flat form fields; the Studio inspector renders them specially.
_HANDLED_COMPOSITION = frozenset({"inner", "retrievers", "complete"})

#: Credential markers — a field whose value is a secret that must never enter a
#: spec (§7.4): rendered as a password field, dropped on export.
#:
#: Deliberately NARROWER than core's `Component._is_secret_key`, which also
#: treats any "token" substring as secret. That over-broad match redacts
#: `max_tokens`/`n_tokens` — ordinary tunables, not credentials — and would drop
#: them from the exported spec here. (It also means the library's own
#: fingerprint ignores `max_tokens`; flagged to the owner separately.)
_SECRET_MARKERS = (
    "api_key", "apikey", "secret", "password", "credential",
    "access_key", "access_token", "refresh_token", "auth_token",
    "private_key", "authorization",
)


def _is_secret_param(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)

# -- the contract → port model --------------------------------------------
#
# Each spec stage has a fixed input/output data contract, read straight from the
# stage ABCs (parser: Source->Document, chunker: Document->Iterator[Chunk], ...).
# The ABC method signatures *are* this map; we restate it explicitly because a
# stable hand-map is more robust than parsing signatures, and it is the single
# thing to update if a stage's contract ever changes. Ports connect iff their
# type strings match — that one rule powers Studio's live connection validation.
CHUNKS = "Chunk[]"
SCORED = "ScoredChunk[]"
STAGE_IO: dict[str, dict[str, Any]] = {
    # The blob store backs parse-caching + raw capture, which happen at parse
    # time — so it is an (optional) dependency of the parser.
    "parser":    {"in": ["Source", "BlobStore"], "out": "Document"},
    "chunker":   {"in": ["Document"],          "out": CHUNKS},
    "enrich":    {"in": [CHUNKS],              "out": CHUNKS},
    "embedder":  {"in": [CHUNKS],              "out": "Representation"},
    "sparse":    {"in": [CHUNKS],              "out": "Representation"},
    "lexical":   {"in": [CHUNKS],              "out": "Representation"},
    "retriever": {"in": ["Query", "Index"],    "out": SCORED},
    "refine":    {"in": [SCORED],              "out": SCORED},
    "generator": {"in": [SCORED],              "out": "Answer"},
    # Infrastructure: no data inputs; each is a dependency wired into a node
    # (Store -> ChunkIndex, BlobStore -> parser).
    "vector_store": {"in": [],                 "out": "Store"},
    "blob_store":   {"in": [],                 "out": "BlobStore"},
}

# One synthetic node, not a registry stage: representation blocks
# (embedder/sparse/lexical) fan into it, a vector Store backs it, and it feeds
# retrievers — mirroring a live ChunkIndex being wired from those backends
# (DR-0001 v2). It is why the spec keys embedder/sparse/lexical/store are
# separate rather than one "index" entry.
INDEX_NODE = {"kind": "index", "in": ["Representation", "Store"], "out": "Index",
              "synthetic": True}

#: A color per contract type, so a port's type is legible at a glance and an
#: edge inherits its source type's color. Tuned for the dark n8n-ish canvas.
TYPE_COLORS: dict[str, str] = {
    "Source":        "#8b8b9e",
    "Document":      "#4f9dde",
    CHUNKS:          "#43b581",
    "Representation": "#c586f0",
    "Index":         "#e0a458",
    "Store":         "#8f7dff",
    "BlobStore":     "#c08a52",
    "Query":         "#5ec8c8",
    SCORED:          "#e06c9f",
    "Answer":        "#d4d44a",
}


def build_manifest() -> dict:
    """Introspect the registry into the Studio manifest dict."""
    stages = _stages()
    components = []
    for stage, reg_kind in SPEC_KINDS.items():
        for name in registry.available(reg_kind):
            components.append(_component(stage, name))
    return {
        "types": {t: {"color": c} for t, c in TYPE_COLORS.items()},
        "stages": stages,
        "components": components,
    }


def _stages() -> list[dict]:
    """Every spec key in pipeline order (SPEC_KINDS order is load-bearing for
    stages; infra follows), plus the synthetic Index node."""
    out = []
    for stage in SPEC_KINDS:  # dict preserves order: stages, then infra
        io = STAGE_IO[stage]
        out.append({
            "kind": stage,
            "in": io["in"],
            "out": io["out"],
            "chain": stage in CHAIN_STAGES,
            "single": stage not in CHAIN_STAGES,
        })
    out.append(INDEX_NODE)
    return out


def _component(stage: str, name: str) -> dict:
    cls = registry.get(SPEC_KINDS[stage], name)
    exportable, blocker = _exportability(cls)
    slot, needs_llm = _composite_shape(cls)
    entry: dict[str, Any] = {
        "kind": stage,
        "name": name,
        "version": getattr(cls, "version", "0.1.0"),
        "doc": inspect.getdoc(cls) or "",
        "takes_index": _takes_index(cls),
        "exportable": exportable,
        "params": _params(cls),
    }
    if slot is not None:
        # A composite retriever: its sub-retrievers nest under this key.
        entry["composite"] = slot
    if needs_llm:
        # Shapes the query with an LLM — wired from the pipeline's generator.
        entry["needs_llm"] = True
    if not exportable:
        # Surfaced as a tooltip so the palette can explain *why* a block is
        # greyed out (it needs another component / a callable a flat spec can't
        # carry — the FusionRetriever/HydeRetriever limitation from builder.py).
        entry["not_exportable_reason"] = (
            f"needs {blocker!r}, which a flat spec can't express"
        )
    return entry


def _takes_index(cls: type) -> bool:
    """Mirror of PipelineBuilder._takes_index: does the constructor accept a
    live index? Asked of the signature, so it stays true for any future
    index-backed component."""
    try:
        return "index" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False


def _exportability(cls: type) -> tuple[bool, str | None]:
    """Can this component be built from a flat `{name, params}` spec?

    It can, unless a constructor parameter is a *composition* dependency a flat
    spec can't carry: another pipeline component or a callable —
    FusionRetriever's `retrievers: Sequence[Retriever]`, HydeRetriever's
    `inner: Retriever` / `complete: Callable`. Those default to None (so they can
    be built by name and raise later), so the tell is the type, not
    required-ness: JSON holds a chunk size, never a live retriever or a function.

    A storage-backend param (BM25Index's optional `store: BlobStore`) is the
    exception: it's optional infra, not composition, so it doesn't block — the
    component just runs unpersisted. Same limit `PipelineBuilder` hits, detected
    once here so the palette greys out only the blocks that truly can't export.
    """
    for pname, hint in _ctor_params(cls):
        if pname in _HANDLED_COMPOSITION:
            continue  # builder wires these (nested sub-specs / generator.complete)
        if _blocks_export(hint):
            return False, pname
    return True, None


def _composite_shape(cls: type) -> tuple[str | None, bool]:
    """How this component nests others: `retrievers` (fusion) or `inner` (hyde/
    multi-query), and whether it shapes the query with an LLM (`complete`)."""
    names = {pname for pname, _ in _ctor_params(cls)}
    slot = "retrievers" if "retrievers" in names else "inner" if "inner" in names else None
    return slot, "complete" in names


def _params(cls: type) -> list[dict]:
    """Every settable param as a UI descriptor: constructor-level params (like
    IndexRetriever's `representation`) *and* Config dataclass fields.

    Both are how a spec configures a component — `cls(index, representation=...)`
    and `**overrides` onto the Config — so the panel must offer both.
    """
    out: list[dict] = []
    seen: set[str] = set()

    # 1. Constructor-level params that aren't wiring and aren't components/
    #    callables. Composition slots (inner/retrievers/complete) are wired
    #    specially, not offered as flat form fields.
    for pname, hint in _ctor_params(cls):
        if pname in _HANDLED_COMPOSITION or _is_component_or_callable(hint):
            continue
        kind, choices = _param_type(hint)
        default = _ctor_default(cls, pname)
        out.append(_param(pname, kind, choices, default, required=default is _NO_DEFAULT))
        seen.add(pname)

    # 2. Config dataclass fields.
    config_cls = getattr(cls, "Config", None)
    if config_cls is not None and dataclasses.is_dataclass(config_cls):
        try:
            hints = typing.get_type_hints(config_cls)
        except Exception:
            hints = {}
        for f in dataclasses.fields(config_cls):
            if f.name in seen:
                continue
            kind, choices = _param_type(hints.get(f.name, f.type))
            default, required = _param_default(f)
            out.append(_param(f.name, kind, choices, default, required))
    return out


_NO_DEFAULT = object()  # sentinel: a constructor param with no default


def _param(name: str, kind: str, choices: list | None, default: Any,
           required: bool) -> dict:
    param: dict[str, Any] = {
        "name": name,
        "type": kind,
        "default": None if default is _NO_DEFAULT else default,
        "required": required,
    }
    if choices is not None:
        param["choices"] = choices
    if _is_secret_param(name):
        # Secrets never travel in a spec (§7.4): rendered as a password field
        # and dropped on export — the environment supplies them.
        param["secret"] = True
    return param


def _ctor_params(cls: type) -> list[tuple[str, Any]]:
    """(name, resolved type hint) for each non-wiring, non-var constructor
    parameter."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    try:
        init: Any = cls.__init__  # type: ignore[misc]  # a type's ctor, not instance access
        hints = typing.get_type_hints(init)
    except Exception:
        hints = {}
    out = []
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) or pname in _WIRING_PARAMS:
            continue
        out.append((pname, hints.get(pname, p.annotation)))
    return out


def _ctor_default(cls: type, pname: str) -> Any:
    p = inspect.signature(cls).parameters[pname]
    if p.default is inspect.Parameter.empty:
        return _NO_DEFAULT
    return _plain(p.default)


def _is_component_or_callable(hint: Any) -> bool:
    """Does this annotation reference a Component subclass or a Callable — i.e.
    something a flat JSON spec can't carry as a settable param? Walks
    Optional/Sequence/Union args. Used to drop such params from the config form
    (a store or a sub-retriever isn't a text field)."""
    if hint is inspect.Parameter.empty or hint is None:
        return False
    origin = typing.get_origin(hint)
    if origin is collections.abc.Callable:
        return True
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if args:
        return any(_is_component_or_callable(a) for a in args)
    return isinstance(hint, type) and issubclass(hint, Component)


def _blocks_export(hint: Any) -> bool:
    """Like `_is_component_or_callable`, but a storage backend
    (BlobStore/VectorStore) does NOT count — it's optional infra a flat spec
    omits, not a composition dependency. This is the line between "can't set this
    param" (both) and "can't build this component at all" (only this)."""
    if hint is inspect.Parameter.empty or hint is None:
        return False
    origin = typing.get_origin(hint)
    if origin is collections.abc.Callable:
        return True
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if args:
        return any(_blocks_export(a) for a in args)
    return (
        isinstance(hint, type)
        and issubclass(hint, Component)
        and not issubclass(hint, _STORAGE_BASES)
    )


def _param_type(hint: Any) -> tuple[str, list | None]:
    """Map a resolved type hint to a UI widget kind (+ enum choices)."""
    hint = _unwrap_optional(hint)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return "enum", [_plain(e.value) for e in hint]
    if hint is bool:
        return "bool", None
    if hint is int:
        return "int", None
    if hint is float:
        return "float", None
    if hint is str:
        return "str", None
    # Everything else (list/tuple/dict/callable/complex) → a JSON editor. Better
    # a raw-but-honest field than a wrong widget.
    return "json", None


def _unwrap_optional(hint: Any) -> Any:
    """Optional[X] / X | None -> X (the widget is the same; the default carries
    the None-ness)."""
    if typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _param_default(f: dataclasses.Field) -> tuple[Any, bool]:
    """(json-safe default, required?). Required means no default at all."""
    if f.default is not dataclasses.MISSING:
        return _plain(f.default), False
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return _plain(f.default_factory()), False
        except Exception:
            return None, False
    return None, True


def _plain(value: Any) -> Any:
    """Coerce a default into something JSON can hold (enums -> value, tuples ->
    lists, unknowns -> their repr)."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
