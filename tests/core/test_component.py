"""Component base: config ergonomics, identity, and abstract enforcement."""
from dataclasses import dataclass
from typing import Optional

import pytest

from rag_blocks.core.component import Component
from rag_blocks.core.errors import ConfigError
from rag_blocks.ingestion.parsers.base import Parser


class Toy(Component):
    kind = "toy"
    name = "toy"

    @dataclass
    class Config:
        size: int = 3
        api_key: Optional[str] = None


def test_defaults_and_overrides():
    assert Toy().config.size == 3
    assert Toy(size=7).config.size == 7
    assert Toy(Toy.Config(size=5), size=9).config.size == 9  # overrides win


def test_unknown_config_key_fails_fast():
    with pytest.raises(ConfigError):
        Toy(sizzle=1)


def test_component_without_config_rejects_kwargs():
    class Bare(Component):
        kind = "toy"
        name = "bare"

    with pytest.raises(ConfigError):
        Bare(anything=1)


def test_fingerprint_is_config_identity():
    assert Toy(size=4).fingerprint() == Toy(size=4).fingerprint()
    assert Toy(size=4).fingerprint() != Toy(size=5).fingerprint()


def test_secrets_are_redacted_and_do_not_affect_identity():
    a, b = Toy(api_key="s3cret-A"), Toy(api_key="s3cret-B")
    assert a.describe()["config"]["api_key"] == "<redacted>"
    # Rotating an API key must NOT invalidate caches:
    assert a.fingerprint() == b.fingerprint()


class Nested(Component):
    kind = "toy"
    name = "nested"

    @dataclass
    class Config:
        headers: Optional[dict] = None


def test_nested_dict_secrets_are_redacted_at_every_depth():
    # A token tucked inside a non-secret-named field must still be redacted (A6).
    c = Nested(headers={"authorization": "Bearer tok", "accept": "json"})
    cfg = c.describe()["config"]["headers"]
    assert cfg["authorization"] == "<redacted>"
    assert cfg["accept"] == "json"
    # And it must not leak into the fingerprint either.
    other = Nested(headers={"authorization": "Bearer different", "accept": "json"})
    assert c.fingerprint() == other.fingerprint()


def test_enums_serialize_as_plain_values():
    from rag_blocks.ingestion.parsers.docling_parser import DoclingParser

    assert DoclingParser().describe()["config"]["ocr_policy"] == "auto"


def test_abstract_methods_are_enforced_at_instantiation():
    """The Python equivalent of Java's `abstract`: you CAN define a subclass
    that forgets the contract, but you cannot INSTANTIATE it."""

    class Incomplete(Parser):  # forgets iter_pages
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()
