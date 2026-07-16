"""Semantics of the shared metadata filter (rag_blocks.storage.filters.matches).

This is the single reference definition used by every Python-side store; Qdrant
reproduces it natively. Pin each rule so a change to one store can't silently
diverge from the others.
"""

from rag_blocks.core.contracts import Chunk
from rag_blocks.storage.filters import matches


def _chunk(**meta) -> Chunk:
    return Chunk(
        id="d:0", doc_id="d", text="body", index=0,
        char_start=0, char_end=1, page_start=1, page_end=1, metadata=meta,
    )


def test_scalar_value_is_equality():
    c = _chunk()
    assert matches(c, {"index": 0})
    assert not matches(c, {"index": 1})
    assert matches(c, {"doc_id": "d"})
    assert not matches(c, {"doc_id": "other"})


def test_list_value_is_membership():
    c = _chunk()
    assert matches(c, {"index": [0, 2, 4]})
    assert not matches(c, {"index": [1, 3]})
    # tuples and sets behave the same as lists
    assert matches(c, {"doc_id": ("d", "e")})
    assert matches(c, {"doc_id": {"d", "e"}})


def test_field_then_metadata_resolution():
    c = _chunk(source="wiki", lang="en")
    # a Chunk field resolves first...
    assert matches(c, {"index": 0})
    # ...then metadata for keys that aren't fields.
    assert matches(c, {"source": "wiki"})
    assert matches(c, {"source": ["wiki", "news"]})
    assert not matches(c, {"source": "news"})
    assert not matches(c, {"missing": "x"})


def test_all_keys_are_anded():
    c = _chunk(source="wiki")
    assert matches(c, {"doc_id": "d", "source": "wiki"})
    assert not matches(c, {"doc_id": "d", "source": "news"})


def test_empty_or_none_filters_match_everything():
    c = _chunk()
    assert matches(c, None)
    assert matches(c, {})
