from __future__ import annotations

import itertools

_ids = itertools.count()


def make_event(step=1, dest="C1", amount=100.0, type_="TRANSFER", eid=None, **kw) -> dict:
    ev = {
        "event_id": eid or f"t-{next(_ids)}",
        "step": step,
        "type": type_,
        "amount": amount,
        "nameOrig": "C0",
        "oldbalanceOrg": amount,
        "newbalanceOrig": 0.0,
        "nameDest": dest,
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
        "isFraud": 0,
        "produced_ms": 1,
    }
    ev.update(kw)
    return ev


class StubScorer:
    """Deterministic stand-in for the API: score depends only on the features."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def score(self, features):
        self.calls += 1
        if self.fail:
            raise RuntimeError("scoring down")
        scores = [min(f["amount"] / 1e6 + 0.01 * f["dest_txn_count"], 1.0) for f in features]
        return scores, [s >= 0.5 for s in scores]
