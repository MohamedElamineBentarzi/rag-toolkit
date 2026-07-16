"""TrialLog: append-only JSONL, plus a SQLite index over it.

Two stores, one truth. AGENTS.md §7.2 settled the shape — *"trial logs are NOT
blobs — they go to JSONL + SQLite"* — and the division of labour is the point:

- **JSONL is the source of truth.** Append-only, diffable, greppable,
  commit-able, readable by anything. A tuning run that dies mid-write loses at
  most its last line, and every line before it is still valid.
- **SQLite is a derived index.** It exists so a leaderboard can ask "top 10 by
  ndcg@10 where the chunker was markdown-aware" without parsing 40k JSON lines.
  It is a cache: delete it and `rebuild()` regenerates it exactly.

That ordering is what makes the failure modes boring. If the two ever disagree,
JSONL wins by definition, so recovery is `rebuild()` rather than a forensic
merge. It is also why this is **not** a `BlobStore`: blobs have no append
(pipeline.py flags `put_stream` as a future fix), and a trial log is nothing
but appends.

Not a `Component`: it is storage wiring with mutable state, not a swappable
algorithm — same reason the pipelines aren't Components.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..core.errors import StorageError
from .trial import Trial

__all__ = ["TrialLog"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id    TEXT PRIMARY KEY,
    started_at  TEXT,
    finished_at TEXT,
    metrics     TEXT,
    cost        TEXT,
    spec        TEXT,
    line        INTEGER
);
CREATE INDEX IF NOT EXISTS trials_started ON trials (started_at);
"""


class TrialLog:
    """Append-only trial history at `path` (.jsonl), indexed in `path`.db.

        log = TrialLog("runs/2026-07-17.jsonl")
        log.append(trial)
        best = log.read()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.db_path = self.path.with_suffix(self.path.suffix + ".db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[sqlite3.Connection] = None
        #: Cached line count. Counted from the file once, then tracked — a
        #: re-count per append would make a 40k-trial run quadratic in the
        #: thing it is trying to measure. Assumes one writer per log, which is
        #: what an append-only run log is.
        self._lines: Optional[int] = None

    # -- write ---------------------------------------------------------------

    def append(self, trial: Trial) -> None:
        """Record one trial: JSONL line first, then the SQLite row.

        Order matters and is not arbitrary. Truth is written before the index,
        so a crash between them costs an index (rebuildable) rather than a
        result (not). The reverse order could leave SQLite claiming a trial
        that no line backs.
        """
        line = trial.to_json()
        if "\n" in line:  # would silently split one record into two
            raise StorageError(
                f"trial {trial.trial_id} serialized with a newline", key=str(self.path)
            )
        if self._lines is None:
            self._lines = self._line_count()
        line_no = self._lines + 1
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                # A tuning run is long and interruptible; the point of an
                # append-only log is that what it already said survives.
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise StorageError(
                f"could not append trial {trial.trial_id}: {exc}", key=str(self.path)
            ) from exc
        # Advanced only after the write landed, so a caller that survives a
        # failed append doesn't number every later line one too high.
        self._lines = line_no
        self._index(trial, line_no)

    def _index(self, trial: Trial, line_no: int) -> None:
        db = self._connect()
        try:
            # Upsert by trial_id: re-running a combination updates its row
            # rather than forking a second history of the same thing
            # (`trial_id_for` is deterministic for exactly this reason).
            db.execute(
                "INSERT INTO trials (trial_id, started_at, finished_at, metrics,"
                " cost, spec, line) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(trial_id) DO UPDATE SET"
                " started_at=excluded.started_at, finished_at=excluded.finished_at,"
                " metrics=excluded.metrics, cost=excluded.cost,"
                " spec=excluded.spec, line=excluded.line",
                (
                    trial.trial_id,
                    trial.started_at,
                    trial.finished_at,
                    json.dumps(trial.metrics, sort_keys=True),
                    json.dumps(trial.cost, sort_keys=True),
                    json.dumps(trial.pipeline_spec, sort_keys=True, default=str),
                    line_no,
                ),
            )
            db.commit()
        except sqlite3.Error as exc:
            raise StorageError(
                f"could not index trial {trial.trial_id}: {exc}", key=str(self.db_path)
            ) from exc

    # -- read ----------------------------------------------------------------

    def read(self) -> list[Trial]:
        """Every trial, in the order they were appended. Reads the JSONL —
        the truth — never the index."""
        return list(self.iter_trials())

    def iter_trials(self) -> Iterator[Trial]:
        """Stream the log. Kept lazy so a leaderboard over a long run never has
        to hold the whole history (the streaming-first rule applies here too)."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Trial.from_json(line)
                except (ValueError, TypeError) as exc:
                    raise StorageError(
                        f"corrupt trial log at line {number}: {exc}",
                        key=str(self.path),
                    ) from exc

    def latest(self, trial_id: str) -> Optional[Trial]:
        """The most recent record for `trial_id`, or None.

        Last wins: a re-run's line is appended after the original, and the
        SQLite row was upserted, so both stores agree on "latest".
        """
        found = None
        for trial in self.iter_trials():
            if trial.trial_id == trial_id:
                found = trial
        return found

    def query(self, sql: str, params: Sequence = ()) -> list[tuple]:
        """Run a read-only query against the SQLite index.

        The escape hatch, deliberately raw: a leaderboard is a view over
        trials, and inventing a query DSL to wrap SQL — which the user already
        knows and which already does this — would be a worse SQL.
        """
        try:
            return list(self._connect().execute(sql, params))
        except sqlite3.Error as exc:
            raise StorageError(f"trial query failed: {exc}", key=str(self.db_path)) from exc

    # -- recovery ------------------------------------------------------------

    def rebuild(self) -> int:
        """Regenerate the SQLite index from the JSONL. Returns rows written.

        The reason the two-store split is safe: the index is disposable. Delete
        the .db, or find it disagreeing with the log, and this makes it right
        again — no merge, no forensics. Idempotent.
        """
        db = self._connect()
        try:
            db.execute("DELETE FROM trials")
            db.commit()
        except sqlite3.Error as exc:
            raise StorageError(
                f"could not clear the index: {exc}", key=str(self.db_path)
            ) from exc
        count = 0
        for number, trial in enumerate(self.iter_trials(), start=1):
            self._index(trial, number)
            count += 1
        return count

    # -- plumbing ------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            try:
                self._db = sqlite3.connect(self.db_path)
                self._db.executescript(_SCHEMA)
            except sqlite3.Error as exc:
                raise StorageError(
                    f"could not open the trial index: {exc}", key=str(self.db_path)
                ) from exc
        return self._db

    def _line_count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "TrialLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
