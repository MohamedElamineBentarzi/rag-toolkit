"""JudgeCache: memoized LLM verdicts (§7.3).

An LLM judge is the only part of this library that charges money *per re-run*.
Without a cache, re-running a leaderboard to reformat a table costs the same as
producing it, so §7.3 requires verdicts be cached by **(question, answer,
judge-model)** — and that triple is exactly right:

- the *question* and the *answer* are what the judge was asked about;
- the *judge model* is who was asked, because two models legitimately disagree
  and one's verdict must never be served as the other's.

Nothing else belongs in the key. Not the retriever, not the chunker: a judge
scoring faithfulness sees the answer and its contexts, so two pipelines that
produced an identical answer deserve one verdict and one bill.

This lives on a `BlobStore` (unlike the trial log, which AGENTS.md §7.2 puts on
JSONL+SQLite) because it *is* a cache — derived, disposable, content-addressed
— which is what a blob store is for. Same shape as `CachingEmbedder`.

Without a store it is a **Null Object**: every lookup misses, every write is a
no-op, and the caller needs no `if cache is not None` branches. The judge simply
costs what it costs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..storage.base import BlobStore

__all__ = ["JudgeCache"]


class JudgeCache:
    """Memoize judge verdicts in a `BlobStore`, keyed per §7.3."""

    def __init__(
        self, store: Optional[BlobStore] = None, *, judge_model: str = "unknown"
    ) -> None:
        """`judge_model` identifies *who judged*. It is part of every key, so
        it must change when the judge does — see `RagasEvaluator.Config`."""
        self._store = store
        self._judge_model = judge_model

    def key(self, question: str, answer: str) -> str:
        """`judge/{sha256(judge_model | question | answer)}.json`.

        The separator is deliberate: without it, `("ab", "c")` and `("a", "bc")`
        would hash alike and one question's verdict would answer another's.
        `\\x00` cannot occur in the text being joined.
        """
        material = "\x00".join([self._judge_model, question, answer])
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"judge/{digest}.json"

    def get(self, question: str, answer: str) -> Optional[dict]:
        """The cached verdict, or None. A miss is never an error."""
        if self._store is None:
            return None
        key = self.key(question, answer)
        if not self._store.exists(key):
            return None
        verdict = json.loads(self._store.get(key).decode("utf-8"))
        return verdict if isinstance(verdict, dict) else None

    def put(self, question: str, answer: str, scores: dict) -> None:
        """Store one verdict. A no-op without a store (Null Object)."""
        if self._store is None:
            return
        self._store.put(
            self.key(question, answer),
            json.dumps(scores, sort_keys=True).encode("utf-8"),
        )
