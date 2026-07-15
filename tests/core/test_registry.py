"""Registry: uses fresh Registry() instances — never the global one — so
these tests stay isolated from components other test modules register."""
from dataclasses import dataclass

import pytest

from rag_blocks.core.component import Component
from rag_blocks.core.errors import ComponentNotFoundError, DuplicateComponentError
from rag_blocks.core.registry import Registry


def make_component(kind_: str, name_: str):
    class C(Component):
        kind = kind_
        name = name_

        @dataclass
        class Config:
            x: int = 0

    return C


def test_register_create_available():
    reg = Registry()
    alpha_cls = reg.register(make_component("stage", "alpha"))
    reg.register(make_component("stage", "beta"))
    assert reg.available("stage") == ["alpha", "beta"]
    inst = reg.create("stage", "alpha", x=42)
    assert isinstance(inst, alpha_cls)
    assert inst.config.x == 42


def test_same_class_reregistration_is_idempotent():
    reg = Registry()
    cls = make_component("stage", "alpha")
    reg.register(cls)
    reg.register(cls)
    assert reg.available("stage") == ["alpha"]


def test_conflicting_registration_is_rejected():
    reg = Registry()
    reg.register(make_component("stage", "alpha"))
    with pytest.raises(DuplicateComponentError):
        reg.register(make_component("stage", "alpha"))


def test_unknown_component_error_lists_alternatives():
    reg = Registry()
    reg.register(make_component("stage", "alpha"))
    with pytest.raises(ComponentNotFoundError, match="alpha"):
        reg.create("stage", "nope")


def test_class_must_declare_identity():
    reg = Registry()

    class Nameless(Component):
        kind = "stage"

    with pytest.raises(DuplicateComponentError):
        reg.register(Nameless)
