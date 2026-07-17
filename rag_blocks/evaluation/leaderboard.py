"""Leaderboard: a view over trials, not a store of them.

The trial log holds the facts; this ranks and prints them. Keeping it a view
(it takes a list of `Trial` and owns nothing) means it can rank a log, a
filtered SQL query, or trials still in memory mid-run, without any of them
knowing about it.

Per-stage **marginal analysis** is the "deep insights" deliverable (§7.3):
*"averaged over everything else, the cross-encoder adds +0.07 nDCG for +180
ms/query."* A winner tells you what to ship; a marginal tells you why, and
which parts of the pipeline are earning their keep. It needs no machinery
beyond grouping the trial log — which is the whole reason `Trial` stores a
per-stage `pipeline_spec` rather than a flattened label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..core.errors import ConfigError
from .trial import Trial

__all__ = ["Leaderboard", "Marginal"]


@dataclass(frozen=True)
class Marginal:
    """What one choice at one stage is worth, averaged over everything else.

    `quality` and `cost` are *deltas against the overall mean*, not absolutes:
    the question a marginal answers is "what did picking this change?", and an
    absolute mean can't answer it.
    """

    stage: str
    option: str
    trials: int
    quality: float
    cost: float

    def __str__(self) -> str:
        # ASCII: this gets printed, and Windows stdout is cp1252.
        return (
            f"{self.stage}={self.option}: {self.quality:+.4f} quality "
            f"for {self.cost:+.1f} cost (n={self.trials})"
        )


class Leaderboard:
    """Rank and display trials.

        board = Leaderboard(log.read())
        for trial in board.top(5, by="ndcg@10"):
            ...
        print(board.to_table(by="ndcg@10"))
    """

    def __init__(self, trials: Sequence[Trial]) -> None:
        # Last write wins per trial_id: a re-run supersedes its original
        # rather than appearing twice (`trial_id_for` is deterministic, and
        # the log is append-only, so duplicates are re-runs by construction).
        deduped: dict[str, Trial] = {}
        for trial in trials:
            deduped[trial.trial_id] = trial
        self.trials = list(deduped.values())

    def __len__(self) -> int:
        return len(self.trials)

    def metrics(self) -> list[str]:
        """Every metric any trial reported, sorted. Trials are not required to
        share a metric set — a phase-1-only trial has no generation metrics,
        and that is a normal state, not a corrupt row."""
        return sorted({metric for t in self.trials for metric in t.metrics})

    def top(self, n: int = 5, *, by: str) -> list[Trial]:
        """The `n` best trials by metric `by`, best first.

        **Higher is better** — the one cross-stage guarantee metrics carry
        (the `ScoredChunk` convention, applied to scores about scores). A
        metric where lower wins (latency) is not ranked here; that is what the
        cost columns are for.

        Trials that never reported `by` are excluded, not sorted last: a trial
        that wasn't scored on a metric has no position in its ranking, and
        defaulting it to 0.0 would invent a loser.
        """
        if n <= 0:
            raise ConfigError(f"top(n) needs a positive n, got {n}")
        if by not in self.metrics():
            raise ConfigError(
                f"no trial reported {by!r}; available: {self.metrics()}"
            )
        scored = [t for t in self.trials if by in t.metrics]
        # trial_id breaks ties so equal scores order deterministically —
        # a leaderboard that reshuffles between runs is not a leaderboard.
        scored.sort(key=lambda t: (-t.metrics[by], t.trial_id))
        return scored[:n]

    def best(self, *, by: str) -> Optional[Trial]:
        """The winning trial, or None if nothing was scored on `by`."""
        winners = self.top(1, by=by)
        return winners[0] if winners else None

    def marginal(
        self, stage: str, *, by: str, cost: str = "query_ms"
    ) -> list[Marginal]:
        """What each option at `stage` was worth, averaged over all else.

        The "deep insights" deliverable (§7.3): *"averaged over every other
        choice, the cross-encoder refiner adds +0.07 nDCG@10 for +180
        ms/query."* That sentence is what a tuning run is actually for — a
        winner tells you what to ship, a marginal tells you **why**, and which
        parts of your pipeline are carrying their weight.

        It needs no new machinery: group the trials by the option they used at
        one stage, average, subtract the overall mean. That it falls out for
        free is not luck — it is why `Trial` stores a full per-stage
        `pipeline_spec` instead of a flattened label.

        Best first by quality delta. Options tried by only one trial are
        reported too: with n=1 the "average over everything else" is an average
        over one thing, so read `trials` before believing a number.
        """
        scored = [t for t in self.trials if by in t.metrics]
        if not scored:
            raise ConfigError(
                f"no trial reported {by!r}; available: {self.metrics()}"
            )

        groups: dict[str, list[Trial]] = {}
        for trial in scored:
            option = _option_at(trial, stage)
            if option is not None:
                groups.setdefault(option, []).append(trial)
        if not groups:
            raise ConfigError(
                f"no trial recorded a {stage!r} stage; recorded stages: "
                f"{sorted({s for t in scored for s in t.pipeline_spec})}"
            )

        overall_quality = _mean([t.metrics[by] for t in scored])
        overall_cost = _mean([t.cost.get(cost, 0.0) for t in scored])

        marginals = [
            Marginal(
                stage=stage,
                option=option,
                trials=len(group),
                quality=_mean([t.metrics[by] for t in group]) - overall_quality,
                cost=_mean([t.cost.get(cost, 0.0) for t in group]) - overall_cost,
            )
            for option, group in groups.items()
        ]
        marginals.sort(key=lambda m: (-m.quality, m.option))
        return marginals

    def to_table(self, *, by: str, n: int = 10, cost: str = "query_ms") -> str:
        """A plain-text table: rank, id, the metric, and what it cost.

        Quality and cost sit side by side and always will. A ranking that shows
        only the score invites picking a winner that is 0.3% better and 40x
        more expensive — showing the price at the moment of choice is the
        library's actual argument.

        `cost` defaults to **query_ms**, not total latency, and that default is
        load-bearing: within a tuning run the parse cache makes an early trial
        pay for work a later one inherits, so `latency_ms` partly ranks by
        running order. `query_ms` is untouched by that, and is what a user
        waiting on an answer actually experiences. Ask for `index_ms` when you
        want the one-time build cost — and read it beside `cache_hits`, or it
        will flatter whichever trial happened to run second.
        """
        # An empty board is a legitimate state (nothing has run yet), and
        # rendering is a display action — it must not explode. `top()` still
        # fails fast on an unknown metric, because that one is a typo.
        if not self.trials:
            return "(no trials)"
        rows = self.top(n, by=by)

        header = f"{'#':>2}  {'trial':<16}  {by:>12}  {cost:>12}"
        lines = [header, "-" * len(header)]
        for rank, trial in enumerate(rows, start=1):
            price = trial.cost.get(cost)
            # An unmeasured cost prints as "-", never as 0.
            shown = f"{price:.1f}" if isinstance(price, (int, float)) else "-"
            lines.append(
                f"{rank:>2}  {trial.trial_id:<16}  "
                f"{trial.metrics[by]:>12.4f}  {shown:>12}"
            )
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _option_at(trial: Trial, stage: str) -> Optional[str]:
    """How a trial spelled its choice at one stage, as a groupable label.

    A chain becomes "a+b" (and the empty chain "none" — the Null Object needs a
    name to be compared against, since "no reranker" is the baseline a
    cross-encoder has to beat). Config is folded in, so `fixed(chunk_chars=512)`
    and `fixed(chunk_chars=1024)` are different options: they are what the tuner
    was actually asked to compare.
    """
    entry = trial.pipeline_spec.get(stage)
    if entry is None:
        return None
    if isinstance(entry, list):
        return "+".join(_label(link) for link in entry) if entry else "none"
    return _label(entry)


def _label(described: dict) -> str:
    name = described.get("name", "?")
    config = described.get("config") or {}
    if not config:
        return str(name)
    inner = ",".join(f"{k}={v}" for k, v in sorted(config.items()))
    return f"{name}({inner})"
