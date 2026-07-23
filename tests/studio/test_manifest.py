"""build_manifest: the registry introspected into Studio's blocks.json. Hermetic.

Studio is a static app; this manifest is its only knowledge of the library, so
the contract that matters is "every registered component shows up, correctly
typed, and round-trips back to a real class". These tests guard that.
"""
from __future__ import annotations

import pytest

from rag_blocks.core.registry import registry
from rag_blocks.evaluation.space import CHAIN_STAGES, SPEC_KINDS
from rag_blocks.studio.manifest import build_manifest


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


def _by_name(manifest, name):
    return next(c for c in manifest["components"] if c["name"] == name)


def test_top_level_shape(manifest):
    assert set(manifest) == {"types", "stages", "components"}


def test_every_registered_component_appears(manifest):
    # The whole point: nothing registered is missing from the palette.
    expected = {
        (stage, name)
        for stage, reg_kind in SPEC_KINDS.items()
        for name in registry.available(reg_kind)
    }
    got = {(c["kind"], c["name"]) for c in manifest["components"]}
    assert got == expected


def test_every_component_round_trips_to_a_real_class(manifest):
    # A manifest entry must name a component the registry can actually build.
    for c in manifest["components"]:
        cls = registry.get(SPEC_KINDS[c["kind"]], c["name"])
        assert cls.name == c["name"]


def test_stages_are_in_pipeline_order_plus_the_synthetic_index(manifest):
    kinds = [s["kind"] for s in manifest["stages"]]
    assert kinds == list(SPEC_KINDS) + ["index"]
    chain = {s["kind"] for s in manifest["stages"] if s.get("chain")}
    assert chain == set(CHAIN_STAGES)


def test_param_types_are_all_known_widgets(manifest):
    allowed = {"str", "int", "float", "bool", "enum", "json"}
    for c in manifest["components"]:
        for p in c["params"]:
            assert p["type"] in allowed, (c["name"], p)


def test_enum_params_carry_choices(manifest):
    # ocr_policy is the canonical enum -> dropdown case.
    policy = next(p for p in _by_name(manifest, "docling")["params"]
                  if p["name"] == "ocr_policy")
    assert policy["type"] == "enum"
    assert set(policy["choices"]) == {"auto", "force", "never"}


def test_index_backed_components_are_flagged(manifest):
    assert _by_name(manifest, "index")["takes_index"] is True
    assert _by_name(manifest, "neighbor-expander")["takes_index"] is True
    assert _by_name(manifest, "fixed")["takes_index"] is False


def test_index_retriever_exposes_its_constructor_param(manifest):
    # `representation` is a constructor arg, not a Config field -- it must still
    # be a settable param or you couldn't pick dense/sparse from the UI.
    names = [p["name"] for p in _by_name(manifest, "index")["params"]]
    assert "representation" in names


def test_composites_are_exportable_with_their_nesting_shape(manifest):
    # fusion wraps a list of retrievers; hyde/multi-query wrap one + need an LLM.
    fusion = _by_name(manifest, "fusion")
    assert fusion["exportable"] is True
    assert fusion["composite"] == "retrievers"
    for name in ("hyde", "multi-query"):
        c = _by_name(manifest, name)
        assert c["exportable"] is True
        assert c["composite"] == "inner"
        assert c["needs_llm"] is True
    assert "composite" not in _by_name(manifest, "index")  # base retriever


def test_optional_storage_backend_does_not_block_export(manifest):
    # BM25Index takes an optional BlobStore for persistence — it runs in-memory
    # without one, so it's fully usable from a flat spec (unlike the composites,
    # whose component/callable deps are essential). The backend isn't a settable
    # param either.
    bm25 = _by_name(manifest, "bm25")
    assert bm25["exportable"] is True
    assert "store" not in [p["name"] for p in bm25["params"]]


def test_store_and_blob_store_are_blocks(manifest):
    # The infrastructure the ChunkIndex/pipeline is built on, now spec-expressible.
    names = {(c["kind"], c["name"]) for c in manifest["components"]}
    assert ("vector_store", "qdrant") in names
    assert ("vector_store", "memory") in names
    assert ("blob_store", "minio") in names
    assert ("blob_store", "local") in names


def test_index_gains_a_store_port_and_parser_a_blobstore_port(manifest):
    stage = {s["kind"]: s for s in manifest["stages"]}
    assert "Store" in stage["index"]["in"]        # Store -> ChunkIndex
    assert "BlobStore" in stage["parser"]["in"]   # BlobStore -> parser
    assert stage["vector_store"]["out"] == "Store"
    assert stage["blob_store"]["out"] == "BlobStore"


def test_minio_credentials_are_secret(manifest):
    secret = {p["name"] for p in _by_name(manifest, "minio")["params"] if p.get("secret")}
    assert {"access_key", "secret_key"} <= secret


def test_only_real_credentials_are_secret(manifest):
    # api_key is secret; max_tokens (contains "token") must NOT be, or it'd be
    # dropped from the exported spec.
    anthropic = {p["name"]: p for p in _by_name(manifest, "anthropic")["params"]}
    assert anthropic["api_key"].get("secret") is True
    assert anthropic["max_tokens"].get("secret") is not True
