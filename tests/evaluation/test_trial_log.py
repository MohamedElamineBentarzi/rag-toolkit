"""TrialLog: JSONL truth + a rebuildable SQLite index. Hermetic."""
from __future__ import annotations

import json

import pytest

from rag_blocks.core.errors import StorageError
from rag_blocks.evaluation import Trial, TrialLog, trial_id_for


def trial(name="fixed", ndcg=0.5, **over) -> Trial:
    spec = {"chunker": {"kind": "chunker", "name": name, "config": {}}}
    fields = dict(
        trial_id=trial_id_for(spec),
        pipeline_spec=spec,
        fingerprints={"chunker": "abc123"},
        metrics={"ndcg@10": ndcg},
        cost={"latency_ms": 12.5},
        cache_hits={"parse": True},
        started_at="2026-07-17T10:00:00",
        finished_at="2026-07-17T10:00:01",
    )
    fields.update(over)
    return Trial(**fields)


# -- the trial id --------------------------------------------------------


def test_trial_id_is_deterministic_and_key_order_independent():
    # Re-running a combination must land on the same row, not fork a second
    # history of the same thing.
    assert trial_id_for({"a": 1, "b": 2}) == trial_id_for({"b": 2, "a": 1})


def test_different_specs_get_different_ids():
    assert trial_id_for({"chunker": "fixed"}) != trial_id_for({"chunker": "markdown"})


def test_trial_id_matches_the_fingerprint_shape():
    assert len(trial_id_for({"a": 1})) == 16


# -- round trip ----------------------------------------------------------


def test_a_trial_round_trips_through_json():
    original = trial()
    assert Trial.from_json(original.to_json()) == original


def test_json_keys_are_sorted_so_logs_diff_cleanly():
    line = trial().to_json()
    keys = list(json.loads(line))
    assert keys == sorted(keys)


def test_a_record_from_a_newer_library_is_refused_not_silently_dropped():
    line = json.dumps({**json.loads(trial().to_json()), "future_field": 1})
    with pytest.raises(ValueError, match="unknown fields"):
        Trial.from_json(line)


# -- append + read -------------------------------------------------------


def test_append_then_read_returns_the_trials_in_order(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="fixed", ndcg=0.5))
    log.append(trial(name="markdown", ndcg=0.9))

    read = log.read()
    assert [t.metrics["ndcg@10"] for t in read] == [0.5, 0.9]
    assert read[0].pipeline_spec["chunker"]["name"] == "fixed"


def test_the_log_is_real_jsonl_one_record_per_line(tmp_path):
    path = tmp_path / "trials.jsonl"
    log = TrialLog(path)
    log.append(trial(name="a"))
    log.append(trial(name="b"))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert all(json.loads(line)["trial_id"] for line in lines)  # each line stands alone


def test_reading_a_log_that_does_not_exist_yet_is_empty_not_an_error(tmp_path):
    assert TrialLog(tmp_path / "nothing.jsonl").read() == []


def test_the_parent_directory_is_created(tmp_path):
    log = TrialLog(tmp_path / "deep" / "nested" / "trials.jsonl")
    log.append(trial())
    assert len(log.read()) == 1


def test_blank_lines_are_tolerated(tmp_path):
    path = tmp_path / "trials.jsonl"
    log = TrialLog(path)
    log.append(trial())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert len(log.read()) == 1


def test_a_corrupt_line_names_its_line_number(tmp_path):
    path = tmp_path / "trials.jsonl"
    log = TrialLog(path)
    log.append(trial())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(StorageError, match="line 2"):
        log.read()


def test_latest_returns_the_most_recent_record_for_a_rerun(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="fixed", ndcg=0.5))
    log.append(trial(name="fixed", ndcg=0.7))  # same spec ⇒ same trial_id

    assert log.latest(trial_id_for(trial().pipeline_spec)).metrics["ndcg@10"] == 0.7
    assert log.latest("nonexistent") is None


# -- the SQLite index ----------------------------------------------------


def test_the_index_agrees_with_the_jsonl(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="fixed", ndcg=0.5))
    log.append(trial(name="markdown", ndcg=0.9))

    rows = log.query("SELECT trial_id, metrics FROM trials ORDER BY line")
    assert len(rows) == 2
    assert [json.loads(m)["ndcg@10"] for _, m in rows] == [0.5, 0.9]
    assert {r[0] for r in rows} == {t.trial_id for t in log.read()}


def test_the_index_is_queryable_by_metric(tmp_path):
    # The reason SQLite exists at all: ask without parsing every line.
    log = TrialLog(tmp_path / "trials.jsonl")
    for i, name in enumerate(["a", "b", "c"]):
        log.append(trial(name=name, ndcg=i / 10))

    rows = log.query(
        "SELECT trial_id FROM trials"
        " WHERE json_extract(metrics, '$.\"ndcg@10\"') > ?",
        (0.05,),
    )
    assert len(rows) == 2


def test_a_rerun_upserts_its_row_rather_than_duplicating_it(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="fixed", ndcg=0.5))
    log.append(trial(name="fixed", ndcg=0.7))  # same id

    rows = log.query("SELECT metrics FROM trials")
    assert len(rows) == 1  # one row...
    assert json.loads(rows[0][0])["ndcg@10"] == 0.7  # ... holding the re-run
    assert len(log.read()) == 2  # ... while the log keeps both as history


def test_line_numbers_track_the_file(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="a"))
    log.append(trial(name="b"))
    log.append(trial(name="c"))
    rows = log.query("SELECT trial_id, line FROM trials ORDER BY line")
    assert [line for _, line in rows] == [1, 2, 3]


def test_a_reopened_log_keeps_counting_from_the_file(tmp_path):
    path = tmp_path / "trials.jsonl"
    first = TrialLog(path)
    first.append(trial(name="a"))
    first.close()

    second = TrialLog(path)  # fresh instance, cold line counter
    second.append(trial(name="b"))
    rows = second.query("SELECT line FROM trials ORDER BY line")
    assert [r[0] for r in rows] == [1, 2]


def test_a_bad_query_is_normalized_into_a_storage_error(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    with pytest.raises(StorageError, match="query failed"):
        log.query("SELECT * FROM nope")


# -- recovery: the index is disposable -----------------------------------


def test_rebuild_regenerates_the_index_from_the_jsonl(tmp_path):
    path = tmp_path / "trials.jsonl"
    log = TrialLog(path)
    log.append(trial(name="a"))
    log.append(trial(name="b"))
    log.close()

    # Lose the index entirely — the failure mode the two-store split exists for.
    log.db_path.unlink()

    recovered = TrialLog(path)
    assert recovered.rebuild() == 2
    assert len(recovered.query("SELECT trial_id FROM trials")) == 2


def test_rebuild_is_idempotent(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="a"))
    assert log.rebuild() == log.rebuild() == 1
    assert len(log.query("SELECT trial_id FROM trials")) == 1


def test_rebuild_drops_rows_the_jsonl_never_backed(tmp_path):
    # JSONL is truth by definition: if the two disagree, the log wins.
    log = TrialLog(tmp_path / "trials.jsonl")
    log.append(trial(name="a"))
    log._connect().execute(
        "INSERT INTO trials (trial_id, line) VALUES ('phantom', 99)"
    )
    log._connect().commit()
    assert len(log.query("SELECT trial_id FROM trials")) == 2

    log.rebuild()
    ids = [r[0] for r in log.query("SELECT trial_id FROM trials")]
    assert "phantom" not in ids


def test_rebuild_on_an_empty_log_writes_nothing(tmp_path):
    assert TrialLog(tmp_path / "empty.jsonl").rebuild() == 0


# -- the context manager -------------------------------------------------


def test_the_log_closes_its_connection(tmp_path):
    with TrialLog(tmp_path / "trials.jsonl") as log:
        log.append(trial())
    assert log._db is None
