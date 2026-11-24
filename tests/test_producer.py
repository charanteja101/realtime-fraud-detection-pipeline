import os
import random
import tempfile
from collections import Counter

from fraudstream.producer import inject_disorder, read_events
from fraudstream.synth import generate

from .helpers import make_event


def events(n=2000, steps=50):
    return [make_event(step=1 + i * steps // n, eid=f"e{i}", dest=f"C{i % 40}") for i in range(n)]


def test_every_event_delivered_once_plus_duplicates():
    out = list(inject_disorder(events(), dup_rate=0.1, late_rate=0.1, rng=random.Random(1)))
    originals = [e["event_id"] for e, tag in out if tag in ("orig", "late")]
    assert sorted(originals) == sorted(e["event_id"] for e in events())
    dups = [e["event_id"] for e, tag in out if tag == "dup"]
    assert len(dups) > 0
    assert all(Counter(originals)[d] == 1 for d in dups)


def test_duplicates_always_arrive_after_their_original():
    out = list(inject_disorder(events(), dup_rate=0.2, rng=random.Random(2)))
    first = {}
    for i, (e, tag) in enumerate(out):
        if tag == "orig":
            first[e["event_id"]] = i
        else:
            assert first[e["event_id"]] < i


def test_late_events_really_arrive_behind_the_stream():
    out = list(inject_disorder(events(), late_rate=0.1, late_steps=10, rng=random.Random(3)))
    watermark, lags = 0, []
    for e, tag in out:
        watermark = max(watermark, e["step"])
        if tag == "late":
            lags.append(watermark - e["step"])
    assert lags and max(lags) >= 1


def test_no_disorder_by_default():
    out = list(inject_disorder(events()))
    assert [e["event_id"] for e, _ in out] == [e["event_id"] for e in events()]
    assert {t for _, t in out} == {"orig"}


def test_read_events_parses_paysim_csv():
    df = generate(500, seed=3)
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(path, index=False)
        evs = list(read_events(path, "r1", limit=100))
    finally:
        os.remove(path)
    assert len(evs) == 100 and evs[0]["event_id"] == "r1-0"
    assert isinstance(evs[0]["amount"], float) and isinstance(evs[0]["step"], int)
