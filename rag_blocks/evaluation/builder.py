"""PipelineBuilder: a spec dict → a live RagPipeline.

Why this exists at all — it is the one piece of v0.8 with no precedent, so the
argument had better be good:

`registry.create(kind, name, **params)` turns data into components, and that is
enough for a chunker or a generator. It is **not** enough for the pipeline as a
whole, because a `ChunkIndex` is *"wired from live, stateful backends — never
built by `registry.create` alone"* (DR-0001 v2) and an `IndexRetriever` says so
out loud: build it by name and it raises *"must be built with index=, not by
name alone"*. Something has to hold the live store, assemble the index, and
hand it to the retriever. In hand-written code that something is you. For a
tuner enumerating 24 combinations, it is this.

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
an index-backed retriever, chains of enrichers and refiners. It cannot build a
`FusionRetriever` (it composes *other retrievers*, not an index — there is no
sensible generic spelling for that in a flat spec). If you need a shape this
doesn't cover, pass your own `Callable[[dict], RagPipeline]` to the tuner:
the seam is the callable, and this class is only its default.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from ..pipeline import RagPipeline, TraceHook, _noop_trace
from ..storage.base import BlobStore
from ..storage.memory_store import MemoryVectorStore
from ..storage.vector_store import VectorStore
from .space import CHAIN_STAGES, STAGE_KINDS

__all__ = ["PipelineBuilder", "PipelineFactory"]

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
        unknown = set(spec) - set(STAGE_KINDS)
        if unknown:
            raise ConfigError(
                f"PipelineBuilder: unknown stage(s) {sorted(unknown)}; "
                f"known: {sorted(STAGE_KINDS)}"
            )

        # The index and its representations. A fresh store per trial: sharing
        # one would let an earlier trial's chunks answer a later trial's query.
        # Spelled out rather than **kwargs so each representation keeps the
        # type ChunkIndex declares for it.
        dense = self._create("embedder", spec["embedder"]) if "embedder" in spec else None
        sparse = self._create("sparse", spec["sparse"]) if "sparse" in spec else None
        lexical = self._create("lexical", spec["lexical"]) if "lexical" in spec else None
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
        for stage in ("parser", "chunker", "generator"):
            if stage in spec:
                kwargs[stage] = self._create(stage, spec[stage])
        for stage in CHAIN_STAGES:
            if stage in spec:
                kwargs[stage] = self._chain(stage, spec[stage])
        if "retriever" in spec:
            kwargs["retriever"] = self._retriever(spec["retriever"], index)

        return RagPipeline(**kwargs)

    # -- construction --------------------------------------------------------

    def _create(self, stage: str, entry: dict) -> Any:
        """One component from `{"name": ..., "params": {...}}`."""
        name, params = _unpack(stage, entry)
        try:
            return registry.create(STAGE_KINDS[stage], name, **params)
        except ConfigError as exc:
            # Name the stage: "unknown field 'sze'" is a lot less useful than
            # knowing which of nine stages spelled it.
            raise ConfigError(f"PipelineBuilder: {stage}={name!r}: {exc}") from exc

    def _chain(self, stage: str, entries: Sequence[dict]) -> list:
        if not isinstance(entries, (list, tuple)):
            raise ConfigError(
                f"PipelineBuilder: {stage}= must be a chain (a list), got "
                f"{type(entries).__name__}"
            )
        return [self._create(stage, entry) for entry in entries]

    def _retriever(self, entry: dict, index: Optional[ChunkIndex]) -> Any:
        """The one stage that cannot be built from data alone.

        A retriever is a *view* over a live index (DR-0001 v2), so the index
        has to be injected — which is the concrete reason this class exists
        rather than a `registry.create` loop.
        """
        name, params = _unpack("retriever", entry)
        if index is None:
            raise ConfigError(
                f"PipelineBuilder: retriever={name!r} needs an index; add an "
                f"embedder/sparse/lexical stage to the space, or drop the "
                f"retriever stage and let RagPipeline derive one"
            )
        cls = registry.get("retriever", name)
        try:
            return cls(index, **params)  # type: ignore[call-arg]
        except TypeError as exc:
            # A retriever composed of other retrievers (fusion) has no index
            # parameter. Say so, instead of leaking a constructor TypeError.
            raise ConfigError(
                f"PipelineBuilder: retriever={name!r} does not take an index "
                f"(retrievers composed of other retrievers can't be built from "
                f"a flat spec) — pass your own pipeline factory to the tuner"
            ) from exc


def _unpack(stage: str, entry: dict) -> tuple[str, dict]:
    if not isinstance(entry, dict) or "name" not in entry:
        raise ConfigError(
            f"PipelineBuilder: {stage} entry must be "
            f'{{"name": ..., "params": {{...}}}}, got {entry!r}'
        )
    return entry["name"], dict(entry.get("params") or {})
