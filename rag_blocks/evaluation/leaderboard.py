"""Leaderboard: a view over trials, not a store of them.

The trial log holds the facts; this ranks and prints them. Keeping it a view
(it takes a list of `Trial` and owns nothing) means it can rank a log, a
filtered SQL query, or trials still in memory mid-run, without any of them
knowing about it.

Per-stage **marginal analysis** — the "deep insights" deliverable (§7.3), where
"averaged over everything else, the cross-encoder adds +0.07 nDCG for +180
ms/query" comes from — arrives in v0.8 PR 4. It needs the search-space
vocabulary (PR 3) to know what a "dimension" is, and it falls out of this same
trial structure for free, which is the whole reason the structure looks like
this.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.errors import ConfigError
from .trial import Trial

__all__ = ["Leaderboard"]


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

    def to_table(self, *, by: str, n: int = 10, cost: str = "latency_ms") -> str:
        """A plain-text table: rank, id, the metric, and what it cost.

        Quality and cost sit side by side and always will. A ranking that shows
        only the score invites picking a winner that is 0.3% better and 40x
        more expensive — showing the price at the moment of choice is the
        library's actual argument.
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
