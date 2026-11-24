"""Generate a small PaySim-shaped CSV for CI, tests and smoke runs.

This is NOT the real PaySim data (download that from Kaggle for real results). It matches the
schema and the broad fraud pattern (fraud only in TRANSFER / CASH_OUT, origin account drained,
fresh "mule" destinations) so every stage of the pipeline can be exercised without the dataset.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]  # fmt: skip


def generate(n_rows: int = 30_000, seed: int = 7, fraud_rate: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows
    n_steps = min(743, max(48, n // 100))
    steps = np.sort(rng.integers(1, n_steps + 1, n))

    types = rng.choice(
        ["PAYMENT", "CASH_IN", "DEBIT", "CASH_OUT", "TRANSFER"],
        size=n,
        p=[0.34, 0.22, 0.01, 0.35, 0.08],
    )
    amount = np.round(rng.lognormal(8.5, 1.4, n), 2)
    old_o = np.round(amount * (1 + rng.lognormal(0.3, 1.0, n)), 2)
    new_o = np.where(types == "CASH_IN", old_o + amount, old_o - amount)

    n_cust = max(1000, n // 3)
    n_hub = max(10, n_cust // 50)
    dest_idx = np.where(
        rng.random(n) < 0.15, rng.integers(0, n_hub, n), rng.integers(0, n_cust, n)
    )
    cust_dest = np.array([f"C{1_000_000 + i}" for i in range(n_cust)])[dest_idx]
    merch_dest = np.array([f"M{2_000_000 + i}" for i in range(max(100, n // 50))])[
        rng.integers(0, max(100, n // 50), n)
    ]
    is_merchant = types == "PAYMENT"
    dest = np.where(is_merchant, merch_dest, cust_dest)

    old_d = np.where(rng.random(n) < 0.4, 0.0, np.round(rng.lognormal(8, 2, n), 2))
    new_d = np.where(types == "CASH_IN", np.maximum(old_d - amount, 0), old_d + amount)
    old_d = np.where(is_merchant, 0.0, old_d)
    new_d = np.where(is_merchant, 0.0, new_d)

    is_fraud = np.zeros(n, dtype=int)
    cand = np.flatnonzero(np.isin(types, ["TRANSFER", "CASH_OUT"]))
    n_fraud = min(int(n * fraud_rate), len(cand))
    fi = rng.choice(cand, n_fraud, replace=False)
    mules = np.array([f"C{9_000_000 + i}" for i in range(max(3, n_fraud // 3))])
    amount[fi] = np.round(rng.lognormal(11.5, 0.8, n_fraud), 2)
    noisy = rng.random(n_fraud) < 0.1  # a few frauds don't drain the account exactly
    old_o[fi] = np.where(noisy, amount[fi] * 1.5, amount[fi])
    new_o[fi] = old_o[fi] - amount[fi]
    dest[fi] = mules[rng.integers(0, len(mules), n_fraud)]
    old_d[fi] = np.where(rng.random(n_fraud) < 0.6, 0.0, np.round(rng.lognormal(9, 1, n_fraud), 2))
    new_d[fi] = np.where(rng.random(n_fraud) < 0.5, old_d[fi], old_d[fi] + amount[fi])
    is_fraud[fi] = 1

    orig = np.array([f"C{v}" for v in rng.integers(10**8, 10**9, n)])
    return pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": old_o,
            "newbalanceOrig": np.round(new_o, 2),
            "nameDest": dest,
            "oldbalanceDest": old_d,
            "newbalanceDest": np.round(new_d, 2),
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
        }
    )[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fraud-rate", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = generate(a.rows, a.seed, a.fraud_rate)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows ({int(df.isFraud.sum())} fraud) to {a.out}")


if __name__ == "__main__":
    main()
