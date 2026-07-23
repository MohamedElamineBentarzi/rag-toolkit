"""PipelineBuilder: a spec dict → a live RagPipeline.

Why this exists at all — it is the one piece of v0.8 with no precedent, so the
argument had better be good:

`registry.create(kind, name, **params)` turns data into components, and that is
enough for a chunker or a generator. It is **not** enough for the pipeline as a
whole, because a `ChunkIndex` is *"wired from live, stateful backends — never
built by `registry.create` alone"* (DR-0001 v2), and the components that read it
say so out loud: build an `IndexRetriever` by name and it raises *"must be built
with index=, not by name alone"*. Something has to hold the live store, assemble
the index, and hand it to whatever needs it. In hand-written code that something
is you. For a tuner enumerating 24 combinations, it is this.

**The rule is stage-agnostic: a component gets the index iff its constructor
takes one.** Asked of the signature, not tracked in a list — `IndexRetriever`
and `HybridRetriever` need one, and so does `NeighborExpander`, which is a
*refiner*. An earlier version special-cased the retriever stage and thereby made
every index-backed refiner unbuildable; the benchmark reported fourteen failed
trials instead of a search.

**It is wiring, not a Strategy** — a plain class, no `kind`, no registry slot,
no fingerprint, exactly like the pipelines it builds. What identifies a trial
is the components' fingerprints, not the glue that assembled them.

**A fresh backend per trial, by default.** Two trials with different chunkers
must not share a `MemoryVectorStore`, or trial 2 retrieves trial 1's chunks and
every number after that is fiction. So `store_factory` is a *factory*, called
per build. The blob store is the deliberate exception: it is a content-addressed
cache keyed by (content hash × fingerprint), so sharing it is not contamination
— it is the entire reason a 24-combination grid parses once (ARCHITECTURE §6.2).

**Its limits, stated rather than discovered.** It builds the pipeline shapes
`SearchSpace` can describe: one index over dense/sparse/lexical representations,
an index-backed retriever, chains of enrichers and refiners. It cannot build
components composed of *other components* — `FusionRetriever` takes retrievers,
`HydeRetriever` takes an inner retriever and a `complete` callable, and there is
no sensible generic spelling for either in a flat spec. Those raise their own
errors, which say what they wanted. If you need a shape this doesn't cover, pass
your own `Callable[[dict], RagPipeline]` to the tuner: the seam is the callable,
and this class is only its default.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, cast

from ..core.component import Component
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.corpus import Corpus
from ..indexing.representation import Representation
from ..pipeline import RagPipeline, TraceHook, _noop_trace
from ..storage.base import BlobStore
from ..storage.memory_store import MemoryVectorStore
from ..storage.vector_store import VectorStore
from .space import CHAIN_STAGES, SPEC_KINDS

__all__ = ["PipelineBuilder", "PipelineFactory", "validate_spec"]

#: What the tuner actually depends on: spec → pipeline. `PipelineBuilder` is
#: the default implementation, never a requirement (Dependency Inversion — the
#: tuner knows this signature, not this class).
PipelineFactory = Callable[[dict], RagPipeline]


class PipelineBuilder:
    """Builds one `RagPipeline` per spec, with fresh state and shared caches.

        builder = PipelineBuilder(blob_store=LocalBlobStore(), trace=collector)
        rag = builder.build({"chunker": {"name": "fixed",
                                         "params": {"chunk_chars": 512}}})
    """

    def __init__(
        self,
        *,
        store_factory: Callable[[], VectorStore] = MemoryVectorStore,
        blob_store: Optional[BlobStore] = None,
        trace: TraceHook = _noop_trace,
        fetch_k: int = 50,
    ) -> None:
        self.store_factory = store_factory
        self.blob_store = blob_store
        self.trace = trace
        self.fetch_k = fetch_k

    def build(self, spec: dict) -> RagPipeline:
        """Assemble a live pipeline from one combination of the search space.

        Anything the spec omits keeps `RagPipeline`'s own default — including
        the retriever, which the pipeline derives from the index's
        representations (one ⇒ IndexRetriever, several ⇒ HybridRetriever). The
        tuner therefore searches over what you asked it to and nothing else.
        """
        # Structure first, one shared gate (`validate_spec`, also what save/load
        # leans on) — known stages, well-formed entries. Semantics (does this
        # component exist, does it accept these params) stay below, where the
        # registry and the Config can say precisely what they wanted.
        validate_spec(spec)

        # The corpus and its representations. A fresh store per trial: sharing
        # one would let an earlier trial's chunks answer a later trial's query.
        # The vector store a spec-named one (Qdrant, in-memory) if given, else
        # the builder's own factory — a fresh instance per build either way, so
        # trials never share a store (the isolation invariant holds for both).
        store = (
            self._create("vector_store", spec["vector_store"], None)
            if "vector_store" in spec else self.store_factory()
        )
        # Representations are a generic list keyed by kind (dense/lexical/… or a
        # plugin) — no hardcoded encoder keys (DR-0004). Each is built here, its
        # nested encoder sub-spec resolved first, then handed to one Corpus.
        reps: list[Representation] = [
            self._build_representation(entry)
            for entry in spec.get("representations", [])
        ]
        corpus: Optional[Corpus] = Corpus(store, reps) if reps else None

        # The truth/parse-cache store: spec-named (MinIO, local) or the builder's.
        # Its credentials never live in the spec (§7.4) — the adapter reads them
        # from the environment, so a spec names *which* store, not its secrets.
        blob_store = (
            self._create("blob_store", spec["blob_store"], None)
            if "blob_store" in spec else self.blob_store
        )
        kwargs: dict[str, Any] = {
            "corpus": corpus,
            "blob_store": blob_store,
            "trace": self.trace,
            "fetch_k": self.fetch_k,
        }
        for stage in ("parser", "chunker", "generator"):
            if stage in spec:
                kwargs[stage] = self._create(stage, spec[stage], corpus)
        if "retriever" in spec:
            # The retriever may be composite (fusion wraps retrievers; hyde /
            # multi-query wrap one inner + shape the query with an LLM). The one
            # thing a spec can't carry — that LLM — is the pipeline's own
            # generator (§7.6's `generator.complete` seam), not a spec field.
            kwargs["retriever"] = self._build_retriever(
                spec["retriever"], corpus, _completion_seam(kwargs.get("generator"))
            )
        for stage in CHAIN_STAGES:
            if stage == "representations":
                continue  # already built into the corpus above (not a RagPipeline arg)
            if stage in spec:
                kwargs[stage] = self._chain(stage, spec[stage], corpus)

        return RagPipeline(**kwargs)

    # -- construction --------------------------------------------------------

    def _create(self, stage: str, entry: dict, corpus: Optional[Corpus]) -> Any:
        """One component from `{"name": ..., "params": {...}}`.

        A component whose constructor takes a `corpus` gets the live one. That
        rule is stage-agnostic on purpose: `IndexRetriever` needs a corpus, and
        so does `NeighborExpander` — a *refiner*. Special-casing the retriever
        stage (as this first did) silently made every corpus-backed refiner
        unbuildable from a spec, and the tuner reported fourteen failed trials
        instead of a search.
        """
        name, params = _unpack(stage, entry)
        cls = registry.get(SPEC_KINDS[stage], name)
        if not _takes_corpus(cls):
            try:
                return registry.create(SPEC_KINDS[stage], name, **params)
            except ConfigError as exc:
                # Name the stage: "unknown field 'sze'" is a lot less useful
                # than knowing which of nine stages spelled it.
                raise ConfigError(
                    f"PipelineBuilder: {stage}={name!r}: {exc}"
                ) from exc
        if corpus is None:
            raise ConfigError(
                f"PipelineBuilder: {stage}={name!r} needs a corpus; add an "
                f"embedder/sparse/lexical stage to the space, or drop this "
                f"stage and let RagPipeline derive one"
            )
        try:
            return cls(corpus, **params)  # type: ignore[call-arg]
        except ConfigError as exc:
            raise ConfigError(f"PipelineBuilder: {stage}={name!r}: {exc}") from exc

    def _build_retriever(
        self,
        entry: dict,
        corpus: Optional[Corpus],
        complete: Optional[Callable[[str], str]],
    ) -> Any:
        """A retriever, recursively — composites wrap other retrievers *as data*
        (DR-0001 v2: retrievers wrapping retrievers, never new pipeline slots).

        `fusion` carries a `retrievers: [<spec>, ...]` list; `hyde` / `multi-query`
        carry an `inner: <spec>` and shape the query with an LLM. A base retriever
        (`index` / `hybrid`) has neither and goes through `_create` (corpus-backed).
        """
        name, _ = _unpack("retriever", entry)
        if "retrievers" in entry:  # fusion: fuse a list of sub-retrievers
            subs = [
                self._build_retriever(e, corpus, complete)
                for e in _entry_list(entry["retrievers"])
            ]
            return self._compose(name, entry, retrievers=subs)
        if "inner" in entry:  # hyde / multi-query: wrap one inner + an LLM
            inner = self._build_retriever(entry["inner"], corpus, complete)
            if complete is None:
                raise ConfigError(
                    f"PipelineBuilder: retriever={name!r} shapes the query with an "
                    f"LLM, but the pipeline has none — add an LLM generator "
                    f"(e.g. {{'generator': {{'name': 'anthropic'}}}})."
                )
            return self._compose(name, entry, inner=inner, complete=complete)
        return self._create("retriever", entry, corpus)  # base: index / hybrid

    def _build_representation(self, entry: dict) -> Representation:
        """One `Representation` from its spec entry, resolving any nested encoder
        sub-spec into a live component first.

        A representation wraps an encoder (an `Embedder`/`SparseEncoder`/
        `LexicalIndex`), which a flat spec can't carry inline — so it arrives as
        a nested `{"name", "params"}` sub-spec under the constructor param that
        is Component-typed (`embedder`/`encoder`/`index`). The builder finds that
        param *by its type* (not a hardcoded name), builds the encoder, and
        injects it — the same nested-sub-spec seam a composite retriever uses for
        `inner`, so a new representation kind needs no builder change (DR-0004).
        """
        name, params = _unpack("representations", entry)
        cls = registry.get("representation", name)
        wiring: dict[str, Any] = {}
        flat: dict[str, Any] = {}
        for pname, value in params.items():
            kind = _encoder_kind(cls, pname)
            if kind is None:
                flat[pname] = value       # a plain settable param (e.g. `space`)
                continue
            if not isinstance(value, Mapping) or "name" not in value:
                raise ConfigError(
                    f"PipelineBuilder: representation={name!r}: {pname!r} must be "
                    f"a {kind} sub-spec {{'name': ...}}, got {value!r}"
                )
            sub_params = dict(value.get("params") or {})
            try:
                wiring[pname] = registry.create(kind, value["name"], **sub_params)
            except ConfigError as exc:
                raise ConfigError(
                    f"PipelineBuilder: representation={name!r}.{pname}: {exc}"
                ) from exc
        try:
            return cast(Representation, cls(**wiring, **flat))
        except ConfigError as exc:
            raise ConfigError(
                f"PipelineBuilder: representation={name!r}: {exc}"
            ) from exc

    def _compose(self, name: str, entry: dict, **wiring: Any) -> Any:
        """Build a composite retriever from its already-built parts + its params."""
        _, params = _unpack("retriever", entry)
        cls = registry.get("retriever", name)
        try:
            return cls(**wiring, **params)
        except (ConfigError, TypeError) as exc:
            raise ConfigError(f"PipelineBuilder: retriever={name!r}: {exc}") from exc

    def _chain(
        self, stage: str, entries: Sequence[dict], corpus: Optional[Corpus]
    ) -> list:
        if not isinstance(entries, (list, tuple)):
            raise ConfigError(
                f"PipelineBuilder: {stage}= must be a chain (a list), got "
                f"{type(entries).__name__}"
            )
        return [self._create(stage, entry, corpus) for entry in entries]


def _completion_seam(generator: Any) -> Optional[Callable[[str], str]]:
    """The bare LLM completion a query-shaping retriever needs, taken from the
    pipeline's generator (§7.6's `generator.complete` seam). `None` when the
    generator has no LLM (the extractive default) — HyDE/MultiQuery then fail
    with a clear message, since they cannot shape a query without one."""
    complete = getattr(generator, "complete", None)
    return complete if callable(complete) else None


def _entry_list(value: Any) -> list:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(
            "PipelineBuilder: retriever.retrievers must be a list of retriever specs"
        )
    return list(value)


def _encoder_kind(cls: type, pname: str) -> Optional[str]:
    """The registry kind of a Component-typed constructor param, else None.

    How the builder tells a representation's *encoder* param (`embedder: Embedder`
    → kind ``"embedder"``) — which arrives as a nested sub-spec to resolve — from
    a plain settable param (`space: str`). Read from the type, so a new
    representation with a new encoder type resolves with no builder change."""
    try:
        hint = typing.get_type_hints(cls.__init__).get(pname)  # type: ignore[misc]
    except Exception:
        return None
    if typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if len(args) == 1 else hint
    if isinstance(hint, type) and issubclass(hint, Component):
        return getattr(hint, "kind", None)
    return None


def _takes_corpus(cls: type) -> bool:
    """Does this component's constructor accept a live `Corpus`?

    Asked of the signature rather than tracked in a list, so a new corpus-backed
    component works here the day it is written — the Open/Closed rule the
    registry exists to keep. Components composed of *other components*
    (`FusionRetriever`, `HydeRetriever`) take no corpus and are left to raise
    their own, better-worded errors.
    """
    try:
        # signature(cls), not cls.__init__: it reports the constructor's
        # parameters without `self`, and doesn't reach through the instance.
        return "corpus" in inspect.signature(cls).parameters
    except (TypeError, ValueError):  # no introspectable signature
        return False


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Structural check that `spec` is a pipeline recipe, *without building it*.

    The cheap, dependency-free gate `save_spec`/`load_spec` and `build` all
    share: every key names a known stage, and every entry is shaped
    `{"name": str, "params": {...}}` — a chain stage (`refine`, `enrich`)
    holding a *list* of those, the empty list included. It stops at structure
    on purpose: an unknown component *name* or a bad *param* is not caught here
    but at `build`, which actually instantiates and lets the registry and the
    component's Config report exactly what they wanted. Structure here (no
    imports, no instantiation); semantics there.

    Raises `ConfigError` at the first problem — the house rule is fail fast at
    the place the mistake was made, so a spec that could never name a pipeline
    never silently reaches (or leaves) disk.
    """
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"spec must be a mapping of stage -> entry, got {type(spec).__name__}"
        )
    unknown = set(spec) - set(SPEC_KINDS)
    if unknown:
        raise ConfigError(
            f"unknown stage(s) {sorted(unknown)}; known: {sorted(SPEC_KINDS)}"
        )
    for stage, value in spec.items():
        if stage in CHAIN_STAGES:
            # A chain stage's value is a list of entries (`[]` = no stage).
            if not isinstance(value, (list, tuple)):
                raise ConfigError(
                    f"{stage}= must be a chain (a list of entries, [] for none), "
                    f"got {type(value).__name__}"
                )
            for entry in value:
                _validate_entry(stage, entry)
        else:
            _validate_entry(stage, value)


def _validate_entry(stage: str, entry: Any) -> None:
    """One `{"name": ..., "params": {...}}` entry, checked for shape only."""
    if not isinstance(entry, Mapping) or "name" not in entry:
        raise ConfigError(
            f'{stage} entry must be {{"name": ..., "params": {{...}}}}, '
            f"got {entry!r}"
        )
    params = entry.get("params")
    # A present-but-non-mapping params would otherwise explode later as a
    # cryptic `dict()` error; name it where the mistake is.
    if params is not None and not isinstance(params, Mapping):
        raise ConfigError(
            f"{stage}={entry['name']!r}: params must be a mapping, "
            f"got {type(params).__name__}"
        )
    # Composite retrievers nest other retriever specs (fusion's `retrievers`,
    # hyde/multi-query's `inner`) — validate them recursively, same shape.
    inner = entry.get("inner")
    if inner is not None:
        _validate_entry(stage, inner)
    subs = entry.get("retrievers")
    if subs is not None:
        if not isinstance(subs, (list, tuple)):
            raise ConfigError(
                f"{stage}={entry['name']!r}: retrievers must be a list of specs"
            )
        for sub in subs:
            _validate_entry(stage, sub)


def _unpack(stage: str, entry: dict) -> tuple[str, dict]:
    _validate_entry(stage, entry)
    return entry["name"], dict(entry.get("params") or {})
