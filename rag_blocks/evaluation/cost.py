"""CostCollector: trace events in, a trial's cost attribution out.

This is what the `TraceEvent` seam was built for (pipeline.py: "the seam the
evaluation suite later hangs cost attribution on"). The collector *is* a
`TraceHook` — it is callable — so wiring it up is one keyword:

    collector = CostCollector()
    rag = RagPipeline(chunk_index=index, trace=collector)

No pipeline change, no instrumentation, no global state: cost attribution
falls out of a seam that already existed, which is the whole reason it was put
there in the first commit.

**On api_usd: prices are never guessed.** Vendor pricing drifts, and a plausible
wrong number is worse than an absent one — someone will make a real decision on
it. So there is no price table in this library, and none will be added. Supply
one (USD per unit, keyed by the usage keys your generator emits) and `api_usd`
is computed; supply nothing and **the key is absent**, not `0.0`. Same family
of honesty as `Page.ocr_applied` and skipped eval samples: the absence of a
number is information, and zero is a lie with a value.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from typing import Mapping, Optional

# A dataclass, not behavior: this depends on the *shape* of a trace event, not
# on the orchestrator that emits one (the `ABCMapping` split below follows
# indexing/chunk_index.py — `typing.Mapping` is for annotations, the abc is for
# isinstance).
from ..pipeline import TraceEvent

__all__ = ["CostCollector"]


class CostCollector:
    """A `TraceHook` that aggregates one trial's latency, tokens and cache hits.

    Not a `Component`: it holds mutable per-run state, which is exactly what a
    Component must never do (fingerprint caching assumes purity). It is
    bookkeeping wiring, like the pipelines it plugs into.

    Reuse across trials via `reset()` — or just build a new one per trial;
    they are cheap.
    """

    def __init__(self, prices: Optional[Mapping[str, float]] = None) -> None:
        """`prices` maps a usage key to USD per unit, e.g.
        `{"input_tokens": 3.0 / 1e6, "output_tokens": 15.0 / 1e6}`. Keys the
        map doesn't mention are counted but never priced.
        """
        self.prices = dict(prices or {})
        self.reset()

    def reset(self) -> None:
        self._latency_ms: dict[str, float] = {}
        self._usage: dict[str, float] = {}
        self._hits: dict[str, list[bool]] = {}
        self.events: int = 0

    # -- the TraceHook -------------------------------------------------------

    def __call__(self, event: TraceEvent) -> None:
        """Absorb one `TraceEvent`. Never raises: a collector that breaks a
        pipeline run would be a monitoring tool causing the outage it watches
        for."""
        self.events += 1
        self._latency_ms[event.stage] = (
            self._latency_ms.get(event.stage, 0.0) + event.duration_ms
        )
        hit = event.detail.get("cache_hit")
        if isinstance(hit, bool):
            self._hits.setdefault(event.stage, []).append(hit)
        usage = event.detail.get("usage")
        if isinstance(usage, ABCMapping):
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._usage[key] = self._usage.get(key, 0.0) + float(value)

    # -- the trial's fields --------------------------------------------------

    def cost(self) -> dict[str, float]:
        """`Trial.cost`: total latency, per-stage latency, tokens, and — only
        if priced — api_usd.

        Per-stage keys ride along because "which stage spent it" is the whole
        question the leaderboard's marginal analysis asks. They sum to
        `latency_ms` exactly: every event reports its OWN cost (pipeline.py's
        `_measured` makes sure nested stages don't double-count).
        """
        out: dict[str, float] = {
            "latency_ms": sum(self._latency_ms.values()),
            **{f"latency_ms.{stage}": ms for stage, ms in sorted(self._latency_ms.items())},
            **{key: value for key, value in sorted(self._usage.items())},
        }
        priced = {k: v for k, v in self._usage.items() if k in self.prices}
        if priced:
            # Only when something was actually priced. A run with prices
            # configured but no matching usage keys still gets no api_usd —
            # "we know it cost nothing" and "we can't price this" differ.
            out["api_usd"] = sum(v * self.prices[k] for k, v in priced.items())
        return out

    def cache_hits(self) -> dict[str, bool]:
        """`Trial.cache_hits`: per stage, was EVERY observation a cache hit?

        `all`, not `any`: with several sources, one parse hitting and nine
        missing is not a reused stage, and reporting True would explain away a
        slow trial with a cache that mostly wasn't there. Partial reuse shows
        up honestly in the latency instead.
        """
        return {stage: all(hits) for stage, hits in sorted(self._hits.items())}
