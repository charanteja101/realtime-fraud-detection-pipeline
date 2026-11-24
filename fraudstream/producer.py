"""Replay a PaySim CSV into Kafka/Redpanda as an event stream.

* key = nameDest, so every event touching a destination account lands on the same partition and
  is processed in order by one consumer (this is what makes incremental state safe).
* ``--rate`` paces events/second (0 = as fast as possible, for throughput tests).
* ``--dup-rate`` re-sends a fraction of events (same event_id) a little later.
* ``--late-rate`` holds a fraction of events back and releases them 1..``--late-steps``
  simulated hours later, i.e. after newer events from other steps were already sent.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
import time
from collections.abc import Iterable, Iterator

from .config import Settings

NUMERIC = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")


def read_events(path: str, run_id: str, limit: int = 0) -> Iterator[dict]:
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit and i >= limit:
                return
            ev = {
                "event_id": f"{run_id}-{i}",
                "step": int(row["step"]),
                "type": row["type"],
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
                "isFraud": int(row["isFraud"]),
            }
            for k in NUMERIC:
                ev[k] = float(row[k])
            yield ev


def inject_disorder(
    events: Iterable[dict],
    dup_rate: float = 0.0,
    late_rate: float = 0.0,
    late_steps: int = 12,
    rng: random.Random | None = None,
) -> Iterator[tuple[dict, str]]:
    """Yield (event, tag) with tag in {"orig", "dup", "late"}; every original appears once."""
    rng = rng or random.Random(0)
    held: list[tuple[int, int, dict]] = []  # (release_step, seq, event)
    dups: list[tuple[int, int, dict]] = []  # (release_after_n_sent, seq, event)
    seq = 0
    sent = 0
    for ev in events:
        while held and held[0][0] <= ev["step"]:
            yield heapq.heappop(held)[2], "late"
            sent += 1
        while dups and dups[0][0] <= sent:
            yield heapq.heappop(dups)[2], "dup"
            sent += 1
        seq += 1
        if late_rate and rng.random() < late_rate:
            heapq.heappush(held, (ev["step"] + rng.randint(1, late_steps), seq, ev))
            continue
        yield ev, "orig"
        sent += 1
        if dup_rate and rng.random() < dup_rate:
            heapq.heappush(dups, (sent + rng.randint(0, 500), seq, ev))
    for _, _, ev in sorted(held):
        yield ev, "late"
    for _, _, ev in sorted(dups):
        yield ev, "dup"


def main() -> None:  # pragma: no cover - needs Kafka
    from confluent_kafka import Producer

    from .kafka_utils import ensure_topic

    cfg = Settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="PaySim CSV path")
    ap.add_argument("--rate", type=float, default=2000, help="events/sec, 0 = unlimited")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (0 = all)")
    ap.add_argument("--run-id", default="ps", help="event_id prefix; same id => duplicates")
    ap.add_argument("--dup-rate", type=float, default=0.0)
    ap.add_argument("--late-rate", type=float, default=0.0)
    ap.add_argument("--late-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ensure_topic(cfg.kafka_bootstrap, cfg.topic, cfg.partitions)
    producer = Producer(
        {
            "bootstrap.servers": cfg.kafka_bootstrap,
            "linger.ms": 5,
            "compression.type": "lz4",
            "enable.idempotence": True,
        }
    )
    errors = 0

    def on_delivery(err, _msg):
        nonlocal errors
        if err is not None:
            errors += 1

    counts = {"orig": 0, "dup": 0, "late": 0}
    stream = inject_disorder(
        read_events(a.data, a.run_id, a.limit),
        a.dup_rate,
        a.late_rate,
        a.late_steps,
        random.Random(a.seed),
    )
    start = time.perf_counter()
    total = 0
    for ev, tag in stream:
        if a.rate > 0 and total % 50 == 0:
            ahead = total / a.rate - (time.perf_counter() - start)
            if ahead > 0.001:
                time.sleep(ahead)
        ev["produced_ms"] = int(time.time() * 1000)
        payload = json.dumps(ev, separators=(",", ":")).encode()
        while True:
            try:
                producer.produce(
                    cfg.topic, key=ev["nameDest"].encode(), value=payload, on_delivery=on_delivery
                )
                break
            except BufferError:
                producer.poll(0.1)
        producer.poll(0)
        counts[tag] += 1
        total += 1
    producer.flush()
    elapsed = time.perf_counter() - start
    print(
        "PRODUCER_SUMMARY "
        + json.dumps(
            {
                "unique_events": counts["orig"] + counts["late"],
                "duplicates_sent": counts["dup"],
                "late_events_sent": counts["late"],
                "messages_sent": total,
                "delivery_errors": errors,
                "seconds": round(elapsed, 2),
                "send_rate_per_s": round(total / elapsed, 1) if elapsed else None,
            }
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
