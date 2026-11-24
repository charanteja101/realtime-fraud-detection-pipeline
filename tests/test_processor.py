from fraudstream.features import feature_vector, features_dict, update_state
from fraudstream.processor import Processor
from fraudstream.store import MemoryStore

from .helpers import StubScorer, make_event


def new_proc(lateness=6):
    store, scorer = MemoryStore(), StubScorer()
    return Processor(store, scorer, allowed_lateness=lateness), store, scorer


def stream():
    return [
        make_event(step=s, dest=d, amount=a, eid=f"e{i}")
        for i, (s, d, a) in enumerate(
            [(1, "C1", 10), (1, "C2", 20), (2, "C1", 30), (3, "C1", 40), (3, "C2", 50), (4, "C3", 5)]
        )
    ]


def test_batch_size_does_not_change_results():
    p1, s1, _ = new_proc()
    p1.process(stream())
    p2, s2, _ = new_proc()
    for e in stream():
        p2.process([e])
    assert {k: r["features"] for k, r in s1.rows.items()} == {
        k: r["features"] for k, r in s2.rows.items()
    }
    assert s1.states == s2.states


def test_features_match_manual_incremental_computation():
    proc, store, _ = new_proc()
    evs = stream()
    proc.process(evs)
    state = None
    for e in evs:
        if e["nameDest"] != "C1":
            continue
        expected = features_dict(feature_vector(e, state))
        assert store.rows[e["event_id"]]["features"] == expected
        state = update_state(state, e)


def test_duplicates_are_dropped_and_do_not_double_count_state():
    proc, store, scorer = new_proc()
    e = make_event(step=1, dest="C9", amount=10.0, eid="same")
    s1 = proc.process([e, dict(e)])  # duplicate inside one batch
    s2 = proc.process([dict(e)])  # duplicate across batches
    assert (s1.new, s1.duplicates, s2.new, s2.duplicates) == (1, 1, 0, 1)
    assert len(store.rows) == 1
    assert store.duplicates == ["same", "same"]
    assert store.states["C9"][0] == 1  # applied exactly once
    assert scorer.calls == 1  # the all-duplicate batch never hits the model


def test_late_events_are_scored_and_landed_but_do_not_touch_state():
    proc, store, _ = new_proc(lateness=6)
    proc.process([make_event(step=20, dest="C1", eid="a")])
    before = [list(x) if isinstance(x, list) else x for x in store.states["C1"]]
    stats = proc.process([make_event(step=5, dest="C1", amount=999.0, eid="late")])
    assert stats.late == 1
    assert store.rows["late"]["is_late"] is True
    assert store.rows["late"]["fraud_score"] is not None
    assert store.states["C1"][0] == 1
    assert store.states["C1"][1] == before[1]


def test_event_within_allowed_lateness_updates_state():
    proc, store, _ = new_proc(lateness=6)
    proc.process([make_event(step=20, dest="C1", eid="a")])
    stats = proc.process([make_event(step=15, dest="C1", amount=5.0, eid="b")])
    assert stats.late == 0 and store.rows["b"]["is_late"] is False
    assert store.states["C1"][0] == 2
    assert store.states["C1"][3] == 20  # last_step stays the max


def test_invalid_events_are_rejected_not_fatal():
    proc, store, _ = new_proc()
    bad = {"event_id": "x", "step": 1}
    notnum = make_event(amount="abc", eid="y")
    stats = proc.process([bad, notnum, make_event(eid="ok")])
    assert stats.rejected == 2 and stats.new == 1 and list(store.rows) == ["ok"]


def test_failed_scoring_leaves_no_trace_and_retry_succeeds():
    proc, store, scorer = new_proc()
    evs = stream()
    scorer.fail = True
    try:
        proc.process(evs)
        raise AssertionError("expected failure")
    except RuntimeError:
        pass
    assert not store.rows and not store.states and proc.watermark == 0
    scorer.fail = False
    stats = proc.process(evs)
    assert stats.new == len(evs) and proc.watermark == 4


def test_alert_flag_comes_from_scorer():
    proc, store, _ = new_proc()
    proc.process([make_event(amount=900_000.0, eid="big"), make_event(amount=10.0, eid="small")])
    assert store.rows["big"]["alert"] is True and store.rows["small"]["alert"] is False
