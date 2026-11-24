"""Train the XGBoost model on PaySim.

Training features are produced by REPLAYING the CSV in step order through the exact same
``feature_vector`` / ``update_state`` code the stream processor uses, so the model never sees a
feature the online path cannot reproduce.

Splits are chronological (not random): first 70% of rows train, next 10% validate (early stopping
and alert-threshold selection), last 20% test. That mirrors production, where you score the future.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_ORDER, feature_vector, update_state

log = logging.getLogger("train")
USECOLS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]  # fmt: skip


def replay_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, step) after replaying df (already sorted by step) through the online code."""
    n = len(df)
    codes, uniques = pd.factorize(df["nameDest"])
    dests = uniques.tolist()
    remaining = np.bincount(codes, minlength=len(dests)).tolist()  # lets us free finished accounts
    states: list = [None] * len(dests)

    cols = {c: df[c].tolist() for c in USECOLS if c not in ("nameDest", "isFraud")}
    code_list = codes.tolist()
    X = np.empty((n, len(FEATURE_ORDER)), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        c = code_list[i]
        txn = {
            "step": cols["step"][i],
            "type": cols["type"][i],
            "amount": cols["amount"][i],
            "oldbalanceOrg": cols["oldbalanceOrg"][i],
            "newbalanceOrig": cols["newbalanceOrig"][i],
            "nameDest": dests[c],
            "oldbalanceDest": cols["oldbalanceDest"][i],
            "newbalanceDest": cols["newbalanceDest"][i],
        }
        st = states[c]
        X[i] = feature_vector(txn, st)
        st = update_state(st, txn)
        remaining[c] -= 1
        states[c] = st if remaining[c] > 0 else None
        if i and i % 500_000 == 0:
            log.info("replayed %d/%d rows (%.0fs)", i, n, time.time() - t0)
    return X, df["isFraud"].to_numpy(dtype=np.int8), df["step"].to_numpy()


def pick_threshold(y: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    """Highest-recall threshold whose validation precision >= min_precision (else best F1)."""
    from sklearn.metrics import precision_recall_curve

    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.flatnonzero(prec >= min_precision)
    if len(ok):
        return float(thr[ok[np.argmax(rec[ok])]])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1))])


def train(
    data: str,
    model_out: str = "models/model.json",
    meta_out: str = "models/model_meta.json",
    max_rows: int | None = None,
    min_precision: float = 0.9,
    seed: int = 42,
) -> dict:
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(data, usecols=USECOLS, nrows=max_rows)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    log.info("loaded %d rows, %d fraud", len(df), int(df.isFraud.sum()))
    X, y, steps = replay_features(df)

    n = len(df)
    i_tr, i_va = int(n * 0.7), int(n * 0.8)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xva, yva = X[i_tr:i_va], y[i_tr:i_va]
    Xte, yte = X[i_va:], y[i_va:]

    pos = max(int(ytr.sum()), 1)
    spw = min((len(ytr) - pos) / pos, 100.0)
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    pva = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, pva, min_precision)
    pte = clf.predict_proba(Xte)[:, 1]
    flag = pte >= thr
    tp = int((flag & (yte == 1)).sum())
    fp = int((flag & (yte == 0)).sum())
    fn = int((~flag & (yte == 1)).sum())
    metrics = {
        "test_rows": int(len(yte)),
        "test_fraud": int(yte.sum()),
        "pr_auc": float(average_precision_score(yte, pte)) if yte.sum() else None,
        "roc_auc": float(roc_auc_score(yte, pte)) if 0 < yte.sum() < len(yte) else None,
        "threshold": thr,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "best_iteration": int(clf.best_iteration),
    }  # fmt: skip
    log.info("test metrics: %s", json.dumps(metrics))

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    clf.get_booster().save_model(model_out)
    meta = {
        "version": f"xgb-{time.strftime('%Y%m%d-%H%M%S')}",
        "feature_order": FEATURE_ORDER,
        "threshold": thr,
        "metrics": metrics,
        "train_rows": int(i_tr),
        "split": "chronological 70/10/20",
    }
    Path(meta_out).write_text(json.dumps(meta, indent=2))
    log.info("saved %s and %s", model_out, meta_out)
    return metrics


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="PaySim CSV")
    ap.add_argument("--model-out", default="models/model.json")
    ap.add_argument("--meta-out", default="models/model_meta.json")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample for a quick run")
    ap.add_argument("--min-precision", type=float, default=0.9)
    a = ap.parse_args()
    train(a.data, a.model_out, a.meta_out, a.max_rows, a.min_precision)


if __name__ == "__main__":  # pragma: no cover
    main()
