"""JudgeCache: the only thing here that charges money per re-run. Hermetic."""
from __future__ import annotations

from rag_blocks.evaluation import JudgeCache
from rag_blocks.storage.local import LocalBlobStore


def cache(tmp_path, judge_model="judge-1") -> JudgeCache:
    return JudgeCache(LocalBlobStore(root=str(tmp_path)), judge_model=judge_model)


def test_a_verdict_round_trips(tmp_path):
    c = cache(tmp_path)
    c.put("q", "a", {"faithfulness": 0.9})
    assert c.get("q", "a") == {"faithfulness": 0.9}


def test_a_miss_is_none_not_an_error(tmp_path):
    assert cache(tmp_path).get("never", "asked") is None


# -- the key is (question, answer, judge-model) --------------------------


def test_a_different_question_is_a_different_verdict(tmp_path):
    c = cache(tmp_path)
    c.put("q1", "a", {"faithfulness": 0.9})
    assert c.get("q2", "a") is None


def test_a_different_answer_is_a_different_verdict(tmp_path):
    c = cache(tmp_path)
    c.put("q", "a1", {"faithfulness": 0.9})
    assert c.get("q", "a2") is None


def test_a_different_judge_never_sees_another_judges_verdict(tmp_path):
    # Two models legitimately disagree; serving one's verdict as the other's
    # would silently attribute an opinion to a model that never held it.
    first = cache(tmp_path, judge_model="gpt-4o")
    second = cache(tmp_path, judge_model="claude-opus-4-8")
    first.put("q", "a", {"faithfulness": 0.9})
    assert second.get("q", "a") is None


def test_the_judge_model_is_part_of_the_key(tmp_path):
    first = cache(tmp_path, judge_model="a")
    second = cache(tmp_path, judge_model="b")
    assert first.key("q", "x") != second.key("q", "x")


def test_field_boundaries_cannot_collide(tmp_path):
    # Without a separator, ("ab","c") and ("a","bc") would hash alike and one
    # question's verdict would answer another's.
    c = cache(tmp_path)
    assert c.key("ab", "c") != c.key("a", "bc")


def test_the_same_triple_is_the_same_key(tmp_path):
    c = cache(tmp_path)
    assert c.key("q", "a") == c.key("q", "a")


def test_keys_live_under_a_judge_prefix(tmp_path):
    assert cache(tmp_path).key("q", "a").startswith("judge/")


# -- the Null Object -----------------------------------------------------


def test_without_a_store_every_lookup_misses_and_writes_are_no_ops():
    # So callers need no `if cache is not None` branches — the judge just
    # costs what it costs.
    c = JudgeCache()
    c.put("q", "a", {"faithfulness": 1.0})
    assert c.get("q", "a") is None


def test_a_cache_survives_a_new_instance_over_the_same_store(tmp_path):
    # The point: re-running a leaderboard to reformat a table must not re-bill.
    cache(tmp_path).put("q", "a", {"faithfulness": 0.9})
    assert cache(tmp_path).get("q", "a") == {"faithfulness": 0.9}
