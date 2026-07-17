"""PipelineBuilder: a spec dict -> a live RagPipeline. Hermetic."""
from __future__ import annotations

import pytest

from rag_blocks.core.errors import ConfigError
from rag_blocks.evaluation import PipelineBuilder
from rag_blocks.pipeline import RagPipeline
from rag_blocks.storage.local import LocalBlobStore
from rag_blocks.storage.memory_store import MemoryVectorStore

DENSE = {"embedder": {"name": "hashing", "params": {"dimensions": 64}}}


def test_builds_a_live_pipeline_from_a_spec():
    rag = PipelineBuilder().build(
        {
            "chunker": {"name": "fixed", "params": {"chunk_chars": 512}},
            "generator": {"name": "extractive", "params": {}},
            **DENSE,
        }
    )
    assert isinstance(rag, RagPipeline)
    assert rag.indexing.chunker.name == "fixed"
    assert rag.indexing.chunker.config.chunk_chars == 512
    assert rag.generator.name == "extractive"


def test_params_may_be_omitted():
    rag = PipelineBuilder().build({"chunker": {"name": "markdown-aware"}})
    assert rag.indexing.chunker.name == "markdown-aware"


def test_an_omitted_stage_keeps_the_pipelines_own_default():
    # The tuner searches over what you asked for and nothing else.
    rag = PipelineBuilder().build(DENSE)
    assert rag.generator.name == "extractive"  # RagPipeline's default


def test_representations_become_the_index():
    rag = PipelineBuilder().build(
        {**DENSE, "lexical": {"name": "bm25", "params": {}}}
    )
    assert set(rag.chunk_index.representations()) == {"dense", "lexical"}


def test_a_chain_stage_builds_in_order():
    rag = PipelineBuilder().build(
        {
            **DENSE,
            "refine": [
                {"name": "score-threshold", "params": {"min_score": 0.1}},
                {"name": "keyword", "params": {}},
            ],
        }
    )
    assert [r.name for r in rag.query_pipeline.refine] == ["score-threshold", "keyword"]
    assert rag.query_pipeline.refine[0].config.min_score == 0.1


def test_an_empty_chain_is_the_null_object():
    rag = PipelineBuilder().build({**DENSE, "refine": []})
    assert rag.query_pipeline.refine == []


# -- state isolation: the invariant that keeps numbers real ---------------


def test_each_build_gets_a_fresh_store():
    # If two trials shared a store, trial 2 would retrieve trial 1's chunks
    # and every number after that is fiction.
    builder = PipelineBuilder()
    first = builder.build(DENSE)
    second = builder.build(DENSE)
    assert first.chunk_index is not second.chunk_index
    assert first.chunk_index._store is not second.chunk_index._store


def test_the_store_factory_is_injectable():
    made: list = []

    def factory():
        store = MemoryVectorStore()
        made.append(store)
        return store

    builder = PipelineBuilder(store_factory=factory)
    builder.build(DENSE)
    builder.build(DENSE)
    assert len(made) == 2  # called per build, not once


def test_the_blob_store_is_shared_across_builds(tmp_path):
    # The deliberate exception to isolation: a content-addressed cache keyed by
    # (content hash x fingerprint) can't contaminate, and sharing it is the
    # entire reason a 24-combination grid parses once.
    blobs = LocalBlobStore(root=str(tmp_path))
    builder = PipelineBuilder(blob_store=blobs)
    first = builder.build(DENSE)
    second = builder.build(DENSE)
    assert first.indexing.blob_store is second.indexing.blob_store is blobs


def test_the_trace_hook_reaches_both_sub_pipelines():
    events: list = []
    hook = events.append  # bound once: `x.append is x.append` is False
    rag = PipelineBuilder(trace=hook).build(DENSE)
    assert rag.indexing.trace is hook
    assert rag.query_pipeline.trace is hook


# -- the retriever: the stage that can't be built from data alone ---------


def test_an_index_backed_retriever_gets_the_live_index():
    rag = PipelineBuilder().build(
        {**DENSE, "retriever": {"name": "index", "params": {"representation": "dense"}}}
    )
    assert rag.retriever.name == "index"
    assert rag.retriever.index is rag.chunk_index  # the reason this class exists


def test_a_retriever_without_an_index_says_what_to_do():
    with pytest.raises(ConfigError, match="needs an index"):
        PipelineBuilder().build({"retriever": {"name": "index"}})


def test_an_index_backed_refiner_also_gets_the_index():
    # Regression: the index was injected only into RETRIEVERS, which made
    # NeighborExpander — a refiner that reads the index — unbuildable from a
    # spec. The benchmark reported 14 failed trials instead of a search.
    rag = PipelineBuilder().build(
        {**DENSE, "refine": [{"name": "neighbor-expander", "params": {"window": 2}}]}
    )
    expander = rag.query_pipeline.refine[0]
    assert expander.index is rag.chunk_index
    assert expander.config.window == 2


def test_a_refiner_that_needs_no_index_is_built_plainly():
    rag = PipelineBuilder().build(
        {**DENSE, "refine": [{"name": "score-threshold", "params": {"min_score": 0.2}}]}
    )
    assert rag.query_pipeline.refine[0].config.min_score == 0.2


def test_an_index_backed_refiner_without_an_index_says_what_to_do():
    with pytest.raises(ConfigError, match="needs an index"):
        PipelineBuilder().build({"refine": [{"name": "neighbor-expander"}]})


def test_a_component_composed_of_other_components_raises_its_own_error():
    # FusionRetriever takes retrievers, not an index; there's no flat spelling
    # for that, so it says what it wanted rather than leaking a TypeError.
    with pytest.raises(ConfigError, match="retrievers"):
        PipelineBuilder().build({**DENSE, "retriever": {"name": "fusion"}})


def test_an_unknown_representation_fails_fast():
    with pytest.raises(ConfigError):
        PipelineBuilder().build(
            {**DENSE, "retriever": {"name": "index", "params": {"representation": "nope"}}}
        )


# -- fail fast ------------------------------------------------------------


def test_an_unknown_stage_lists_the_known_ones():
    with pytest.raises(ConfigError, match="unknown stage"):
        PipelineBuilder().build({"chunkerr": {"name": "fixed"}})


def test_an_unknown_component_name_fails_fast():
    with pytest.raises(Exception, match="nonexistent|not found|no component"):
        PipelineBuilder().build({"chunker": {"name": "nonexistent"}})


def test_a_bad_param_names_the_stage_that_owns_it():
    # "unknown field 'sze'" is much less useful without knowing which of nine
    # stages spelled it.
    with pytest.raises(ConfigError, match="chunker"):
        PipelineBuilder().build({"chunker": {"name": "fixed", "params": {"sze": 1}}})


def test_a_malformed_entry_says_the_shape_it_wanted():
    with pytest.raises(ConfigError, match='"name"'):
        PipelineBuilder().build({"chunker": "fixed"})


def test_a_chain_stage_given_a_bare_entry_fails_fast():
    with pytest.raises(ConfigError, match="must be a chain"):
        PipelineBuilder().build({**DENSE, "refine": {"name": "keyword"}})


# -- the seam is the callable, not the class ------------------------------


def test_a_custom_factory_can_replace_the_builder_entirely():
    # PipelineFactory is Callable[[dict], RagPipeline]; the tuner depends on
    # that signature, never on this class.
    def my_factory(spec: dict) -> RagPipeline:
        return RagPipeline(chunker=None)

    rag = my_factory({"anything": "at all"})
    assert isinstance(rag, RagPipeline)
