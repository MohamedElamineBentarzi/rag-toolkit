"""save_spec / load_spec: a pipeline recipe round-trips through JSON. Hermetic.

The split under test: the *recipe* (components + params) persists here; the
*state* (vectors, blobs) does not — it lives in the stores. So the guarantee is
"save then load then build gives the same pipeline shape", never "the indexed
data came back".
"""
from __future__ import annotations

import json

import pytest

import rag_blocks as rk
from rag_blocks.core.errors import ConfigError
from rag_blocks.evaluation import (
    PipelineBuilder,
    load_spec,
    save_spec,
    validate_spec,
)

# A complete, buildable recipe: one dense representation, a real chunker, a
# refine chain, a generator. Hashing embedder keeps it dependency-free.
SPEC = {
    "chunker": {"name": "fixed", "params": {"chunk_chars": 512, "overlap_chars": 64}},
    "embedder": {"name": "hashing", "params": {"dimensions": 64}},
    "retriever": {"name": "index", "params": {"representation": "dense"}},
    "refine": [{"name": "score-threshold", "params": {"min_score": 0.1}}],
    "generator": {"name": "extractive", "params": {}},
}


# -- the round trip -------------------------------------------------------


def test_save_then_load_is_the_same_spec(tmp_path):
    path = tmp_path / "pipeline.json"
    save_spec(SPEC, path)
    assert load_spec(path) == SPEC


def test_a_loaded_spec_builds_the_pipeline_it_described(tmp_path):
    path = tmp_path / "pipeline.json"
    save_spec(SPEC, path)

    rag = PipelineBuilder().build(load_spec(path))

    assert rag.indexing.chunker.name == "fixed"
    assert rag.indexing.chunker.config.chunk_chars == 512
    assert rag.retriever.name == "index"
    assert [r.name for r in rag.query_pipeline.refine] == ["score-threshold"]
    assert rag.generator.name == "extractive"


def test_the_loaded_pipeline_actually_answers(tmp_path):
    # End to end on the recipe: rebuild, re-index (state is NOT restored), ask.
    path = tmp_path / "pipeline.json"
    save_spec(SPEC, path)
    rag = PipelineBuilder().build(load_spec(path))

    rag.index(rk.Source.from_bytes(b"The capital of France is Paris.", name="f.txt"))
    answer = rag.ask("What is the capital of France?")

    assert answer.text  # a non-empty cited answer came back


def test_str_and_path_are_both_accepted(tmp_path):
    path = tmp_path / "pipeline.json"
    save_spec(SPEC, str(path))       # str in
    assert load_spec(path) == SPEC   # Path out


def test_the_empty_chain_survives_the_round_trip(tmp_path):
    # "no reranker" is a real, distinct recipe — it must not be dropped as falsy.
    path = tmp_path / "pipeline.json"
    spec = {"embedder": {"name": "hashing"}, "refine": []}
    save_spec(spec, path)
    assert load_spec(path)["refine"] == []


# -- what lands on disk ---------------------------------------------------


def test_the_file_is_human_readable_json(tmp_path):
    path = tmp_path / "pipeline.json"
    save_spec(SPEC, path)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text                     # indented, not one line
    assert json.loads(text) == SPEC         # and valid JSON


def test_the_output_is_stable_across_key_order(tmp_path):
    # sort_keys: two specs that differ only in insertion order save identically,
    # so a re-save diffs cleanly in review.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_spec({"chunker": {"name": "fixed"}, "embedder": {"name": "hashing"}}, a)
    save_spec({"embedder": {"name": "hashing"}, "chunker": {"name": "fixed"}}, b)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_indent_none_writes_the_compact_form(tmp_path):
    path = tmp_path / "pipeline.json"
    save_spec({"chunker": {"name": "fixed"}}, path, indent=None)
    assert "\n" not in path.read_text(encoding="utf-8")


# -- fail fast, at the call that made the mistake -------------------------


def test_save_validates_before_writing(tmp_path):
    # A malformed recipe must not leave a half-written file to fail at load.
    path = tmp_path / "pipeline.json"
    with pytest.raises(ConfigError, match="unknown stage"):
        save_spec({"chunkerr": {"name": "fixed"}}, path)
    assert not path.exists()


def test_load_rejects_a_drifted_file(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"nope": {"name": "fixed"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown stage"):
        load_spec(path)


def test_load_rejects_a_non_object_json(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(["fixed"]), encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_spec(path)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_spec(tmp_path / "does-not-exist.json")


# -- validate_spec directly: structure only, never semantics --------------


def test_validate_accepts_a_well_formed_spec():
    validate_spec(SPEC)  # does not raise


def test_validate_rejects_a_malformed_entry():
    with pytest.raises(ConfigError, match='"name"'):
        validate_spec({"chunker": "fixed"})


def test_validate_rejects_a_bare_chain_entry():
    with pytest.raises(ConfigError, match="must be a chain"):
        validate_spec({"refine": {"name": "keyword"}})


def test_validate_rejects_non_mapping_params():
    with pytest.raises(ConfigError, match="params must be a mapping"):
        validate_spec({"chunker": {"name": "fixed", "params": [1, 2]}})


def test_validate_stops_at_structure_not_names():
    # An unknown component name is the builder's to catch, not this gate — so a
    # structurally sound spec with a nonexistent chunker validates fine here.
    validate_spec({"chunker": {"name": "nonexistent"}})


def test_the_pair_is_exposed_at_the_top_level():
    assert rk.save_spec is save_spec
    assert rk.load_spec is load_spec
