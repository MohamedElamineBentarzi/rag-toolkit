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
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence

from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from ..pipeline import RagPipeline, TraceHook, _noop_trace
from ..storage.base import BlobStore
from ..storage.memory_store import MemoryVectorStore
from ..storage.vector_store import VectorStore
from .space import CHAIN_STAGES, STAGE_KINDS

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

        # The index and its representations. A fresh store per trial: sharing
        # one would let an earlier trial's chunks answer a later trial's query.
        # Spelled out rather than **kwargs so each representation keeps the
        # type ChunkIndex declares for it.
        # index=None: these are the index's ingredients, built before it exists.
        # None of them takes an index — they are what an index is made of.
        dense = self._create("embedder", spec["embedder"], None) if "embedder" in spec else None
        sparse = self._create("sparse", spec["sparse"], None) if "sparse" in spec else None
        lexical = self._create("lexical", spec["lexical"], None) if "lexical" in spec else None
        index: Optional[ChunkIndex] = None
        if dense is not None or sparse is not None or lexical is not None:
            index = ChunkIndex(
                self.store_factory(), dense=dense, sparse=sparse, lexical=lexical
            )

        kwargs: dict[str, Any] = {
            "chunk_index": index,
            "blob_store": self.blob_store,
            "trace": self.trace,
            "fetch_k": self.fetch_k,
        }
        for stage in ("parser", "chunker", "generator", "retriever"):
            if stage in spec:
                kwargs[stage] = self._create(stage, spec[stage], index)
        for stage in CHAIN_STAGES:
            if stage in spec:
                kwargs[stage] = self._chain(stage, spec[stage], index)

        return RagPipeline(**kwargs)

    # -- construction --------------------------------------------------------

    def _create(self, stage: str, entry: dict, index: Optional[ChunkIndex]) -> Any:
        """One component from `{"name": ..., "params": {...}}`.

        A component whose constructor takes an `index` gets the live one. That
        rule is stage-agnostic on purpose: `IndexRetriever` needs an index, and
        so does `NeighborExpander` — a *refiner*. Special-casing the retriever
        stage (as this first did) silently made every index-backed refiner
        unbuildable from a spec, and the tuner reported fourteen failed trials
        instead of a search.
        """
        name, params = _unpack(stage, entry)
        cls = registry.get(STAGE_KINDS[stage], name)
        if not _takes_index(cls):
            try:
                return registry.create(STAGE_KINDS[stage], name, **params)
            except ConfigError as exc:
                # Name the stage: "unknown field 'sze'" is a lot less useful
                # than knowing which of nine stages spelled it.
                raise ConfigError(
                    f"PipelineBuilder: {stage}={name!r}: {exc}"
                ) from exc
        if index is None:
            raise ConfigError(
                f"PipelineBuilder: {stage}={name!r} needs an index; add an "
                f"embedder/sparse/lexical stage to the space, or drop this "
                f"stage and let RagPipeline derive one"
            )
        try:
            return cls(index, **params)  # type: ignore[call-arg]
        except ConfigError as exc:
            raise ConfigError(f"PipelineBuilder: {stage}={name!r}: {exc}") from exc

    def _chain(
        self, stage: str, entries: Sequence[dict], index: Optional[ChunkIndex]
    ) -> list:
        if not isinstance(entries, (list, tuple)):
            raise ConfigError(
                f"PipelineBuilder: {stage}= must be a chain (a list), got "
                f"{type(entries).__name__}"
            )
        return [self._create(stage, entry, index) for entry in entries]


def _takes_index(cls: type) -> bool:
    """Does this component's constructor accept a live `ChunkIndex`?

    Asked of the signature rather than tracked in a list, so a new index-backed
    component works here the day it is written — the Open/Closed rule the
    registry exists to keep. Components composed of *other components*
    (`FusionRetriever`, `HydeRetriever`) take no index and are left to raise
    their own, better-worded errors.
    """
    try:
        # signature(cls), not cls.__init__: it reports the constructor's
        # parameters without `self`, and doesn't reach through the instance.
        return "index" in inspect.signature(cls).parameters
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
    unknown = set(spec) - set(STAGE_KINDS)
    if unknown:
        raise ConfigError(
            f"unknown stage(s) {sorted(unknown)}; known: {sorted(STAGE_KINDS)}"
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


def _unpack(stage: str, entry: dict) -> tuple[str, dict]:
    _validate_entry(stage, entry)
    return entry["name"], dict(entry.get("params") or {})
