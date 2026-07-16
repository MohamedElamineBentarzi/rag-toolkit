"""CostCollector: TraceEvents in, a trial's cost attribution out. Hermetic."""
from __future__ import annotations

import pytest

from rag_blocks.evaluation import CostCollector
from rag_blocks.pipeline import TraceEvent


def event(stage, ms=10.0, **detail) -> TraceEvent:
    return TraceEvent(stage, "src.pdf", ms, detail)


# -- latency -------------------------------------------------------------


def test_latency_totals_and_attributes_per_stage():
    collector = CostCollector()
    for e in [event("parse", 100.0), event("chunk", 5.0), event("generate", 900.0)]:
        collector(e)

    cost = collector.cost()
    assert cost["latency_ms"] == pytest.approx(1005.0)
    assert cost["latency_ms.parse"] == pytest.approx(100.0)
    assert cost["latency_ms.generate"] == pytest.approx(900.0)


def test_repeated_stages_accumulate():
    # One trial indexes many documents; "parse" fires per source.
    collector = CostCollector()
    collector(event("parse", 10.0))
    collector(event("parse", 30.0))
    assert collector.cost()["latency_ms.parse"] == pytest.approx(40.0)


def test_per_stage_latency_sums_to_the_total():
    # Holds because every event reports its OWN cost — `_measured` in
    # pipeline.py stops nested stages double-counting.
    collector = CostCollector()
    for stage in ["parse", "chunk", "enrich", "retrieve", "refine", "generate"]:
        collector(event(stage, 7.0))
    cost = collector.cost()
    per_stage = sum(v for k, v in cost.items() if k.startswith("latency_ms."))
    assert per_stage == pytest.approx(cost["latency_ms"])


def test_a_collector_with_no_events_reports_zero_latency_and_no_tokens():
    cost = CostCollector().cost()
    assert cost == {"latency_ms": 0.0}


# -- tokens --------------------------------------------------------------


def test_usage_is_summed_across_events():
    collector = CostCollector()
    collector(event("generate", usage={"input_tokens": 100, "output_tokens": 20}))
    collector(event("generate", usage={"input_tokens": 50, "output_tokens": 5}))

    cost = collector.cost()
    assert cost["input_tokens"] == 150
    assert cost["output_tokens"] == 25


def test_an_empty_usage_dict_contributes_nothing():
    # ExtractiveGenerator reports {} — a free generator, not a broken one.
    collector = CostCollector()
    collector(event("generate", usage={}))
    assert "input_tokens" not in collector.cost()


def test_non_numeric_usage_values_are_ignored():
    collector = CostCollector()
    collector(event("generate", usage={"model": "claude", "input_tokens": 10}))
    cost = collector.cost()
    assert cost["input_tokens"] == 10
    assert "model" not in cost


def test_booleans_are_not_counted_as_tokens():
    # bool is an int in Python; a `usage={"cached": True}` must not become 1.
    collector = CostCollector()
    collector(event("generate", usage={"cached": True}))
    assert "cached" not in collector.cost()


# -- api_usd: never guessed ----------------------------------------------


def test_no_prices_means_no_api_usd_key_not_zero():
    # The honesty rule: absent, never 0.0. A zero would be a lie with a value.
    collector = CostCollector()
    collector(event("generate", usage={"input_tokens": 1000}))
    assert "api_usd" not in collector.cost()


def test_prices_are_applied_per_usage_key():
    collector = CostCollector(
        prices={"input_tokens": 3.0 / 1e6, "output_tokens": 15.0 / 1e6}
    )
    collector(event("generate", usage={"input_tokens": 1_000_000, "output_tokens": 100_000}))
    # 1M in @ $3/M + 100k out @ $15/M = 3.00 + 1.50
    assert collector.cost()["api_usd"] == pytest.approx(4.5)


def test_unpriced_usage_keys_are_counted_but_not_billed():
    collector = CostCollector(prices={"input_tokens": 1.0})
    collector(event("generate", usage={"input_tokens": 2, "cache_read_tokens": 999}))
    cost = collector.cost()
    assert cost["cache_read_tokens"] == 999   # visible
    assert cost["api_usd"] == pytest.approx(2.0)  # ... but not invented a price for


def test_prices_configured_but_nothing_matched_still_reports_no_api_usd():
    # "It cost nothing" and "we can't price this" are different claims.
    collector = CostCollector(prices={"input_tokens": 1.0})
    collector(event("generate", usage={"mystery_tokens": 5}))
    assert "api_usd" not in collector.cost()


# -- cache hits ----------------------------------------------------------


def test_cache_hits_are_read_from_the_trace_detail():
    collector = CostCollector()
    collector(event("parse", cache_hit=True))
    assert collector.cache_hits() == {"parse": True}


def test_a_stage_is_only_reused_if_every_observation_hit():
    # `all`, not `any`: one hit out of ten is not a reused stage, and
    # reporting True would explain away a slow trial with a cache that
    # mostly wasn't there.
    collector = CostCollector()
    collector(event("parse", cache_hit=True))
    collector(event("parse", cache_hit=False))
    assert collector.cache_hits() == {"parse": False}


def test_stages_that_never_report_a_cache_are_absent():
    collector = CostCollector()
    collector(event("generate"))
    assert collector.cache_hits() == {}


# -- lifecycle -----------------------------------------------------------


def test_the_collector_is_a_trace_hook():
    # Callable, so it drops straight into RagPipeline(trace=...) with no
    # adapter and no pipeline change.
    collector = CostCollector()
    hook = collector  # typed as TraceHook at the call site
    hook(event("parse", 5.0))
    assert collector.events == 1


def test_reset_clears_the_state_between_trials():
    collector = CostCollector()
    collector(event("parse", 10.0, cache_hit=True))
    collector.reset()

    assert collector.cost() == {"latency_ms": 0.0}
    assert collector.cache_hits() == {}
    assert collector.events == 0


def test_reset_keeps_the_prices():
    collector = CostCollector(prices={"input_tokens": 1.0})
    collector.reset()
    collector(event("generate", usage={"input_tokens": 3}))
    assert collector.cost()["api_usd"] == pytest.approx(3.0)
