"""`save_spec` / `load_spec`: a pipeline's *recipe*, persisted as JSON.

A pipeline splits into two halves that live in different places, and this module
exists to keep them from being confused for each other:

- The **recipe** — which components, which params — is plain data by design
  (principle #5, config-as-data): the very same `{stage: {"name", "params"}}`
  spec that `SearchSpace` emits and `PipelineBuilder` consumes. Small, diffable,
  reviewable, safe to commit.
- The **state** — the embedded vectors, the raw/parsed blobs — is large and
  backend-owned, and already persists where it belongs: the `VectorStore` and
  the `BlobStore` (§7.2, "blob store = truth; Qdrant = derived and rebuildable").

So this saves and loads the recipe, and *only* the recipe. Rebuilding is the
recipe plus a re-`index()` (or a store you reopen) — never a pickled object
graph of live backends, which the architecture deliberately has no way to make.

    save_spec(spec, "pipeline.json")
    rag = PipelineBuilder().build(load_spec("pipeline.json"))

Both ends run `validate_spec` — the same structural gate `build` uses — so a
malformed recipe fails at the call that made it, not weeks later at load. It is
structure only: a component *name* or a *param* is still the builder's to check,
when it actually instantiates. Round-tripping is exact; there is nothing here
but JSON I/O and that shared check, on purpose.

Secrets (§7.4): a spec names *which* component, never its credentials, so a spec
file is safe to commit — the environment supplies the keys. This module writes
exactly the dict it is given; keeping secrets out of the spec is the caller's
job, as it is everywhere else in the library.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .builder import validate_spec

__all__ = ["save_spec", "load_spec"]


def save_spec(
    spec: Mapping[str, Any], path: str | Path, *, indent: int | None = 2
) -> None:
    """Write a validated pipeline spec to `path` as UTF-8 JSON.

    Validated *before* the write, so a malformed recipe never leaves a
    half-written file behind and fails at this call rather than at some later
    `load_spec`. `sort_keys` makes the output stable across equivalent specs so
    two saves diff cleanly in review; `indent` defaults to a human-readable 2
    (pass `None` for the compact single-line form).
    """
    validate_spec(spec)
    text = json.dumps(spec, indent=indent, sort_keys=True)
    Path(path).write_text(text, encoding="utf-8")


def load_spec(path: str | Path) -> dict:
    """Read a pipeline spec from `path`, validate its structure, return the dict.

    The result is exactly what `PipelineBuilder.build` wants, so the whole
    load-and-rebuild is one line:

        rag = PipelineBuilder().build(load_spec("pipeline.json"))

    A hand-edited or drifted file is caught here (`validate_spec`) rather than as
    a cryptic failure deep inside the build.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec
