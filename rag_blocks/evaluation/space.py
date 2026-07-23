"""SearchSpace: the tuner's input, as declarative data (ARCHITECTURE.md §6.1).

    space = SearchSpace(
        chunker=[choice("fixed", chunk_chars=[512, 1024], overlap_chars=[0, 64]),
                 choice("markdown-aware")],
        retriever=[choice("index", representation="dense"), choice("hybrid")],
        refine=[[], [choice("cross-encoder", top_k=[5, 10])]],
    )

Plain data, not components. A search space names *what to try*; it has no
behavior to swap, so it gets no `kind`, no registry slot, and no fingerprint —
inventing a component for it would be a taxonomy entry wrapping a dict
(DR-0002 §7 made the same call about `EvalDataset`).

**A list is a grid axis; a tuple is one value.** `chunk_chars=[512, 1024]` means
two trials; `k_values=(1, 5, 10)` means one trial configured with that tuple.
The distinction has to live somewhere — a parameter whose value is genuinely a
sequence is common (`k_values`, `weights`) and "expand every list" would make
those untunable and, worse, silently wrong. This convention matches §6.1's
notation and the codebase's existing habit of tuples for literal sequences
(`RetrievalEvaluator.Config.k_values` is a tuple).

Chain stages (`refine`, `enrich`) take a list of *chains*, each chain a list of
choices — including the empty chain `[]`, which is a legitimate and important
candidate: "no reranker" is exactly what a cross-encoder must beat to earn its
180 ms (the Null Object as an empty chain, DR-0001 v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterator, Mapping

from ..core.errors import ConfigError

__all__ = [
    "Choice", "choice", "SearchSpace",
    "STAGE_KINDS", "INFRA_KINDS", "SPEC_KINDS", "CHAIN_STAGES",
]

#: Pipeline stage → registry kind. The keys are the vocabulary a SearchSpace
#: accepts; the values are what `registry.create` wants. Two names for one
#: thing, because a pipeline's keyword ("refine") and a component's slot
#: ("refiner") legitimately differ.
#:
#: **Declaration order is pipeline order, and it is load-bearing** — `expand()`
#: enumerates with the earliest stage varying slowest, so trials that share an
#: expensive prefix (same parser, same chunker) run back to back and inherit a
#: warm cache. Sorting these alphabetically would put `parser` after
#: `generator` and re-parse the corpus on nearly every trial.
STAGE_KINDS: Mapping[str, str] = {
    "parser": "parser",
    "chunker": "chunker",
    "enrich": "enricher",
    "embedder": "embedder",
    "sparse": "sparse_encoder",
    "lexical": "lexical_index",
    "retriever": "retriever",
    "refine": "refiner",
    "generator": "generator",
}

#: Infrastructure a pipeline is *given*, not a tunable stage: the vector store
#: the ChunkIndex persists into, and the blob store that backs parse-caching +
#: raw capture. Kept OUT of STAGE_KINDS on purpose — they carry no pipeline
#: order and a SearchSpace does not tune them (a live Qdrant/MinIO is deployment,
#: not a knob). A single built spec MAY carry them; `PipelineBuilder` wires them
#: in, else falls back to its own `store_factory` / `blob_store` defaults.
INFRA_KINDS: Mapping[str, str] = {
    "vector_store": "vector_store",
    "blob_store": "blob_store",
}

#: Everything a concrete pipeline spec may name: tunable stages + infrastructure.
#: `validate_spec` and `PipelineBuilder` accept these keys; `SearchSpace` accepts
#: only `STAGE_KINDS`.
SPEC_KINDS: Mapping[str, str] = {**STAGE_KINDS, **INFRA_KINDS}

#: Stages that are a *chain* of components rather than one.
CHAIN_STAGES = frozenset({"refine", "enrich"})


@dataclass(frozen=True)
class Choice:
    """One component to try, with the parameter grid to try it over."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def expand(self) -> Iterator[dict]:
        """Every concrete config this choice stands for, in a stable order.

        Cartesian product over the list-valued params; tuple- and scalar-valued
        params are carried through untouched.
        """
        axes = {k: v for k, v in self.params.items() if isinstance(v, list)}
        fixed = {k: v for k, v in self.params.items() if not isinstance(v, list)}
        if not axes:
            yield {"name": self.name, "params": dict(fixed)}
            return
        # sorted(): the product's order must not depend on kwargs order, or a
        # grid would enumerate differently between two equivalent spellings of
        # the same space — and trial ids would follow.
        keys = sorted(axes)
        for combo in product(*(axes[k] for k in keys)):
            yield {"name": self.name, "params": {**fixed, **dict(zip(keys, combo))}}


def choice(name: str, **params: Any) -> Choice:
    """Name a component to try: `choice("fixed", chunk_chars=[512, 1024])`.

    List values expand into separate trials; everything else (including
    tuples) is one value. See the module docstring — that rule is the one
    thing to know about this API.
    """
    if not name or not isinstance(name, str):
        raise ConfigError(f"choice() needs a component name, got {name!r}")
    for key, value in params.items():
        if isinstance(value, list) and not value:
            # An empty axis silently annihilates the whole grid (product with
            # an empty sequence is empty), and the user would see zero trials
            # with no explanation. Fail where the mistake is.
            raise ConfigError(
                f"choice({name!r}): {key}=[] has no values to try; drop the "
                f"parameter, or pass a value"
            )
    return Choice(name=name, params=dict(params))


class SearchSpace:
    """The stages to vary and the options per stage.

    Unknown stage names fail at construction rather than at trial 30 of an
    overnight run — the house rule, and the cheapest place to catch a typo.
    """

    def __init__(self, **stages: Any) -> None:
        unknown = set(stages) - set(STAGE_KINDS)
        if unknown:
            raise ConfigError(
                f"SearchSpace: unknown stage(s) {sorted(unknown)}; "
                f"known: {sorted(STAGE_KINDS)}"
            )
        self.stages: dict[str, list] = {}
        for stage, options in stages.items():
            if not isinstance(options, (list, tuple)):
                raise ConfigError(
                    f"SearchSpace: {stage}= must be a list of options, "
                    f"got {type(options).__name__}"
                )
            if not options:
                raise ConfigError(
                    f"SearchSpace: {stage}=[] has nothing to try; drop the "
                    f"stage to leave it at the pipeline's default"
                )
            self.stages[stage] = list(options)
        self._validate()

    def _validate(self) -> None:
        for stage, options in self.stages.items():
            for option in options:
                if stage in CHAIN_STAGES:
                    # A chain stage's option is a chain: a (possibly empty)
                    # sequence of Choices.
                    if not isinstance(option, (list, tuple)):
                        raise ConfigError(
                            f"SearchSpace: {stage}= takes chains (lists of "
                            f"choices, `[]` for none), not a bare "
                            f"{type(option).__name__} — did you mean "
                            f"[{stage}=[[choice(...)]]]?"
                        )
                    bad = [c for c in option if not isinstance(c, Choice)]
                    if bad:
                        raise ConfigError(
                            f"SearchSpace: {stage} chain holds a "
                            f"{type(bad[0]).__name__}, expected choice(...)"
                        )
                elif not isinstance(option, Choice):
                    raise ConfigError(
                        f"SearchSpace: {stage}= takes choice(...) options, "
                        f"got {type(option).__name__}"
                    )

    def dimensions(self) -> list[str]:
        """The stages being varied, in **pipeline order** (see `STAGE_KINDS`).

        Not alphabetical: the order decides which stage `expand()` varies
        slowest, and therefore whether a grid re-parses the corpus every trial
        or once. The leaderboard's marginal analysis groups by exactly these
        (v0.8 PR 4).
        """
        return [stage for stage in STAGE_KINDS if stage in self.stages]

    def _options_for(self, stage: str) -> list:
        """Every concrete option for one stage: configs for a plain stage,
        chains-of-configs for a chain stage."""
        expanded: list = []
        for option in self.stages[stage]:
            if stage in CHAIN_STAGES:
                # Each link in a chain has its own grid, so a chain of two
                # choices with 2 params each is 4 chains, not 2.
                links = [list(link.expand()) for link in option]
                if not links:
                    expanded.append([])  # the empty chain: a real candidate
                    continue
                expanded.extend([list(combo) for combo in product(*links)])
            else:
                expanded.extend(option.expand())
        return expanded

    def expand(self) -> Iterator[dict]:
        """Every combination in the space, as concrete specs, in a stable order.

        Order is deterministic and prefix-major: stages go in pipeline order
        and the product varies the LAST one fastest, so the earliest (most
        expensive) stage changes least often. That is not cosmetic — it is
        what keeps trials sharing a parse/embed prefix adjacent, so the caches
        that already exist stay warm across neighbours instead of thrashing.
        It is also the whole reason this milestone needs no cache of its own
        (DR-0003).
        """
        stages = self.dimensions()
        if not stages:
            yield {}
            return
        per_stage = [self._options_for(stage) for stage in stages]
        for combo in product(*per_stage):
            yield dict(zip(stages, combo))

    def __len__(self) -> int:
        """How many trials this space is worth — before you start the run."""
        total = 1
        for stage in self.dimensions():
            total *= len(self._options_for(stage))
        return total

    def __repr__(self) -> str:
        # ASCII only: a repr gets printed, and Windows consoles default to
        # cp1252, where a stray "→" raises UnicodeEncodeError and takes the
        # caller's program down with it. Prose may use arrows; output may not.
        return f"<SearchSpace {self.dimensions()} -> {len(self)} combinations>"
