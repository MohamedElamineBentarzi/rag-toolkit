"""build_manifest: the registry introspected into Studio's blocks.json. Hermetic.

Studio is a static app; this manifest is its only knowledge of the library, so
the contract that matters is "every registered component shows up, correctly
typed, and round-trips back to a real class". These tests guard that.
"""
from __future__ import annotations

import pytest

from rag_blocks.core.registry import registry
from rag_blocks.evaluation.space import CHAIN_STAGES, STAGE_KINDS
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
        for stage, reg_kind in STAGE_KINDS.items()
        for name in registry.available(reg_kind)
    }
    got = {(c["kind"], c["name"]) for c in manifest["components"]}
    assert got == expected


def test_every_component_round_trips_to_a_real_class(manifest):
    # A manifest entry must name a component the registry can actually build.
    for c in manifest["components"]:
        cls = registry.get(STAGE_KINDS[c["kind"]], c["name"])
        assert cls.name == c["name"]


def test_stages_are_in_pipeline_order_plus_the_synthetic_index(manifest):
    kinds = [s["kind"] for s in manifest["stages"]]
    assert kinds == list(STAGE_KINDS) + ["index"]
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


def test_composites_are_marked_not_exportable(manifest):
    for name in ("fusion", "hyde", "multi-query"):
        c = _by_name(manifest, name)
        assert c["exportable"] is False
        assert "not_exportable_reason" in c
    assert _by_name(manifest, "fixed")["exportable"] is True


def test_only_real_credentials_are_secret(manifest):
    # api_key is secret; max_tokens (contains "token") must NOT be, or it'd be
    # dropped from the exported spec.
    anthropic = {p["name"]: p for p in _by_name(manifest, "anthropic")["params"]}
    assert anthropic["api_key"].get("secret") is True
    assert anthropic["max_tokens"].get("secret") is not True
