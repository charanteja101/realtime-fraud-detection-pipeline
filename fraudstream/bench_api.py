"""Load-test the scoring API on its own: single-transaction latency and requests/second.

    python -m fraudstream.bench_api --url http://localhost:8000 --requests 5000 --concurrency 8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from .features import feature_vector, features_dict

SAMPLE_TXN = {
    "step": 10, "type": "TRANSFER", "amount": 181000.0, "oldbalanceOrg": 181000.0,
    "newbalanceOrig": 0.0, "nameDest": "C553264065", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}  # fmt: skip


def main() -> None:  # pragma: no cover - needs a running API
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    body = {"features": features_dict(feature_vector(SAMPLE_TXN, None))}

    def worker(n: int) -> list[float]:
        out = []
        with httpx.Client(base_url=a.url, timeout=10) as c:
            for _ in range(n):
                t = time.perf_counter()
                c.post("/score", json=body).raise_for_status()
                out.append((time.perf_counter() - t) * 1000)
        return out

    per = a.requests // a.concurrency
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        lats = np.array([x for r in ex.map(worker, [per] * a.concurrency) for x in r])
    wall = time.perf_counter() - t0
    print(f"{len(lats)} requests, concurrency {a.concurrency}: {len(lats) / wall:.0f} req/s")
    print(
        f"latency ms  p50 {np.percentile(lats, 50):.2f}  p95 {np.percentile(lats, 95):.2f}  "
        f"p99 {np.percentile(lats, 99):.2f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
