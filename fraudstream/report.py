"""Measure the run from what landed in Postgres: latency, throughput, quality, dedupe/late.

    python -m fraudstream.report                      # print a report
    python -m fraudstream.report --wait --expect N    # wait for N events to land, then report;
                                                      # exit 1 if the count is wrong (used by CI)

Latency definitions (all wall-clock ms on one host):
  end_to_end  = landed_ms   - produced_ms   (producer send  -> just before DB commit)
  kafka_wait  = consumed_ms - produced_ms   (broker + queueing)
  feat+score  = scored_ms   - consumed_ms   (dedupe, state load, features, HTTP scoring)
  db_write    = landed_ms   - scored_ms     (row building; the commit itself is not included)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import Settings

STAGES = {
    "end_to_end": "landed_ms - produced_ms",
    "kafka_wait": "consumed_ms - produced_ms",
    "feat_plus_score": "scored_ms - consumed_ms",
    "db_write": "landed_ms - scored_ms",
}


def _counts(conn) -> tuple[int, int]:
    n = conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    d = conn.execute("SELECT count(*) FROM duplicate_events").fetchone()[0]
    return n, d


def wait_for(conn, expect: int, timeout: float, quiet_s: float = 3.0) -> None:
    deadline = time.time() + timeout
    last, last_change = (-1, -1), time.time()
    while time.time() < deadline:
        cur = _counts(conn)
        if cur != last:
            last, last_change = cur, time.time()
        elif cur[0] >= expect and time.time() - last_change >= quiet_s:
            return
        time.sleep(1)
    print(f"timeout waiting for {expect} events (have {last[0]})", file=sys.stderr)


def build_report(conn) -> dict:
    r: dict = {}
    row = conn.execute(
        """SELECT count(*), count(*) FILTER (WHERE is_late), count(*) FILTER (WHERE alert),
                  count(*) FILTER (WHERE alert AND is_fraud = 1),
                  count(*) FILTER (WHERE is_fraud = 1),
                  min(produced_ms), max(landed_ms), count(DISTINCT name_dest)
           FROM transactions"""
    ).fetchone()
    total, late, alerts, tp, fraud, t0, t1, dests = row
    r["events_landed"] = total
    r["duplicates_dropped"] = conn.execute("SELECT count(*) FROM duplicate_events").fetchone()[0]
    r["late_events"] = late
    r["destination_accounts"] = dests
    if not total:
        return r
    secs = max((t1 - t0) / 1000.0, 1e-9)
    peak = conn.execute(
        "SELECT max(c) FROM (SELECT count(*) c FROM transactions GROUP BY landed_ms / 1000) t"
    ).fetchone()[0]
    r["throughput"] = {
        "avg_events_per_s": round(total / secs, 1),
        "peak_events_in_1s": peak,
        "wall_seconds": round(secs, 2),
    }
    lat = {}
    for name, expr in STAGES.items():
        p = conn.execute(
            f"""SELECT percentile_cont(ARRAY[0.5, 0.95, 0.99])
                WITHIN GROUP (ORDER BY {expr}) FROM transactions"""
        ).fetchone()[0]
        lat[name] = {"p50_ms": round(p[0], 1), "p95_ms": round(p[1], 1), "p99_ms": round(p[2], 1)}
    r["latency"] = lat
    r["alerts"] = alerts
    r["quality_on_replay"] = {
        "fraud_events": fraud,
        "alerts": alerts,
        "true_positives": tp,
        "precision": round(tp / alerts, 4) if alerts else None,
        "recall": round(tp / fraud, 4) if fraud else None,
    }
    return r


def print_report(r: dict) -> None:
    print("\n=== Fraud pipeline run report ===")
    print(f"events landed        : {r['events_landed']}")
    print(f"duplicates dropped   : {r['duplicates_dropped']}")
    print(f"late events          : {r['late_events']}  (scored + landed, state not updated)")
    if "throughput" in r:
        t = r["throughput"]
        print(f"throughput           : {t['avg_events_per_s']} ev/s avg, "
              f"{t['peak_events_in_1s']} ev/s peak 1s, over {t['wall_seconds']}s")
        print("\nlatency (ms)         p50      p95      p99")
        for k, v in r["latency"].items():
            print(f"  {k:<18} {v['p50_ms']:>7} {v['p95_ms']:>8} {v['p99_ms']:>8}")
        q = r["quality_on_replay"]
        print(f"\nalerts {q['alerts']}  TP {q['true_positives']}  fraud {q['fraud_events']}  "
              f"precision {q['precision']}  recall {q['recall']}")


def main() -> None:  # pragma: no cover - needs Postgres
    import psycopg

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--expect", type=int, default=0, help="expected number of unique events")
    ap.add_argument("--wait", action="store_true", help="wait until --expect events have landed")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--json-out", help="also write the report as JSON")
    a = ap.parse_args()

    with psycopg.connect(Settings().database_url, autocommit=True) as conn:
        if a.wait and a.expect:
            wait_for(conn, a.expect, a.timeout)
        rep = build_report(conn)
    print_report(rep)
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(rep, fh, indent=2)
    if a.expect and rep["events_landed"] != a.expect:
        print(f"FAIL: expected {a.expect} events, landed {rep['events_landed']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
