"""Trial: one tested pipeline combination, fully reproducible from its record.

The unit the whole tuning suite exists to produce. A `Trial` is deliberately
**self-contained**: it carries the full `describe()` of every stage, not a
reference to the objects that produced it, so a line of the log is enough to
reconstruct what ran, six months later, without the code that ran it.

Secrets are already gone by the time they arrive here — `Component.describe()`
redacts them at every depth (core/component.py) — which is why a trial log can
be committed to a repo at all. That is not this module's cleverness; it is the
payoff of putting redaction in `describe()` instead of in the logger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Trial", "trial_id_for"]


def trial_id_for(pipeline_spec: dict) -> str:
    """Deterministic id for a pipeline combination: sha256(spec)[:16].

    Same shape as `Component.fingerprint()` (same hash, same truncation) —
    ids and fingerprints are read side by side in logs, and two conventions
    would be one too many. Deterministic on purpose: re-running the same
    combination produces the same id, so a re-run *updates* its row instead of
    forking a second history of the same thing.
    """
    canonical = json.dumps(pipeline_spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Trial:
    """One combination's full record: what ran, how well, at what price.

    Frozen: a trial is a historical fact. Editing one after the fact would
    corrupt the only thing the leaderboard can trust.
    """

    #: sha256(pipeline_spec)[:16] — see `trial_id_for`.
    trial_id: str
    #: `describe()` per stage, keyed by stage slot ("chunker", "generator", …).
    #: Secrets already redacted. This is what makes a trial reproducible.
    pipeline_spec: dict
    #: `fingerprint()` per stage — the cache keyspace this trial ran in.
    #: Derivable from pipeline_spec, but stored: it is what you group and join
    #: on, and recomputing it later would mean re-deriving a hash convention.
    fingerprints: dict[str, str]
    #: The evaluators' aggregate output, flattened ("recall@5", "token_f1", …).
    metrics: dict[str, float] = field(default_factory=dict)
    #: latency_ms (+ per-stage), token counts, and api_usd IFF prices were
    #: supplied. See `CostCollector` — an unpriced run has NO api_usd key.
    cost: dict[str, float] = field(default_factory=dict)
    #: Per stage: was every observation of it served from cache?
    cache_hits: dict[str, bool] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    #: Free-text notes / dataset identity / anything the pressure valve is for.
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """One JSONL line. Sorted keys so a log diffs cleanly in review."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, line: str) -> "Trial":
        data: dict[str, Any] = json.loads(line)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # A log written by a NEWER version of this library. Refusing beats
            # silently dropping fields and re-writing a lossy copy.
            raise ValueError(
                f"trial record has unknown fields {sorted(unknown)} — it was "
                f"probably written by a newer rag-blocks than this one"
            )
        return cls(**data)
