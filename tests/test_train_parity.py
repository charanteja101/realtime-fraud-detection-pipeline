"""The key ML-engineering guarantee: offline training features == online streaming features."""
import numpy as np

from fraudstream.features import FEATURE_ORDER
from fraudstream.processor import Processor
from fraudstream.store import MemoryStore
from fraudstream.synth import generate
from fraudstream.train import pick_threshold, replay_features

from .helpers import StubScorer


def test_offline_replay_equals_online_processor():
    df = generate(4000, seed=11).sort_values("step", kind="stable").reset_index(drop=True)
    X, y, _ = replay_features(df)

    store = MemoryStore()
    proc = Processor(store, StubScorer(), allowed_lateness=10_000)
    events = []
    for i, r in enumerate(df.to_dict("records")):
        r["event_id"] = f"r-{i}"
        events.append(r)
    for i in range(0, len(events), 700):  # arbitrary batch boundaries
        proc.process(events[i : i + 700])

    online = np.array(
        [[store.rows[f"r-{i}"]["features"][c] for c in FEATURE_ORDER] for i in range(len(df))],
        dtype=np.float32,
    )
    assert np.allclose(X, online, rtol=1e-5, atol=1e-5)
    assert not np.isnan(X).any()
    assert y.sum() > 0


def test_pick_threshold_meets_precision_target():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.65])
    thr = pick_threshold(y, p, min_precision=0.9)
    flagged = p >= thr
    assert y[flagged].mean() >= 0.9
