import pytest

pytest.importorskip("xgboost")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import json  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from fraudstream.api import create_app  # noqa: E402
from fraudstream.features import FEATURE_ORDER, feature_vector, features_dict  # noqa: E402
from fraudstream.model import ModelBundle  # noqa: E402
from fraudstream.synth import generate  # noqa: E402
from fraudstream.train import train  # noqa: E402

_cache: dict = {}


def trained_bundle(tmp_dir: str) -> ModelBundle:
    if "b" not in _cache:
        import os
        import tempfile

        d = tempfile.mkdtemp()
        csv = os.path.join(d, "s.csv")
        generate(12_000, seed=5, fraud_rate=0.02).to_csv(csv, index=False)
        m = train(csv, os.path.join(d, "m.json"), os.path.join(d, "meta.json"))
        assert m["recall"] is not None
        _cache["b"] = ModelBundle.load(os.path.join(d, "m.json"), os.path.join(d, "meta.json"))
        _cache["meta"] = json.load(open(os.path.join(d, "meta.json")))
    return _cache["b"]


FRAUDY = {
    "step": 10, "type": "TRANSFER", "amount": 181000.0, "oldbalanceOrg": 181000.0,
    "newbalanceOrig": 0.0, "nameDest": "C9000001", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}  # fmt: skip
NORMAL = {
    "step": 10, "type": "PAYMENT", "amount": 40.0, "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4960.0, "nameDest": "M2000001", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}  # fmt: skip


def feats(txn):
    return features_dict(feature_vector(txn, None))


def client():
    return TestClient(create_app(trained_bundle("")))


def test_health_and_model_info():
    with client() as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/model").json()["feature_order"] == FEATURE_ORDER


def test_score_returns_score_and_alert_flag():
    with client() as c:
        fraud = c.post("/score", json={"features": feats(FRAUDY)}).json()
        ok = c.post("/score", json={"features": feats(NORMAL)}).json()
    assert 0 <= ok["fraud_score"] <= 1 and fraud["fraud_score"] > ok["fraud_score"]
    assert fraud["alert"] is True and ok["alert"] is False
    assert fraud["alert"] == (fraud["fraud_score"] >= fraud["threshold"])


def test_batch_matches_single():
    with client() as c:
        batch = c.post(
            "/score/batch", json={"items": [feats(FRAUDY), feats(NORMAL)]}
        ).json()
        single = c.post("/score", json={"features": feats(FRAUDY)}).json()
    assert abs(batch["scores"][0] - single["fraud_score"]) < 1e-6
    assert batch["alerts"] == [True, False]


def test_missing_feature_is_422():
    with client() as c:
        bad = feats(NORMAL)
        bad.pop("amount")
        assert c.post("/score", json={"features": bad}).status_code == 422
