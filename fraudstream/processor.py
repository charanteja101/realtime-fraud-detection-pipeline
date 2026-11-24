"""Stream processor: Kafka -> dedupe -> stateful features -> scoring -> Postgres.

``Processor.process`` is pure orchestration over a Store and a Scorer, so it is unit-tested
without Kafka or Postgres. ``run`` wires it to Kafka.

Per batch:
  1. validate events; drop duplicates (already landed, or repeated inside the batch)
  2. walk events in order, advancing a watermark (max step seen); an event more than
     ``allowed_lateness`` steps behind the watermark is *late*
  3. features come from the destination state BEFORE the event; on-time events then update it
     (late events are scored and landed but never touch state)
  4. score the whole batch with one call, then write rows + state + duplicate log atomically
  5. (in ``run``) commit Kafka offsets only after the DB commit
"""
from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass

from .config import Settings
from .features import REQUIRED_FIELDS, feature_vector, features_dict, update_state
from .scorer import HttpScorer, Scorer
from .store import PostgresStore, Store

log = logging.getLogger("processor")


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class BatchStats:
    received: int = 0
    new: int = 0
    duplicates: int = 0
    late: int = 0
    rejected: int = 0
    alerts: int = 0

    def add(self, o: BatchStats) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(o, f))


def is_valid(ev: object) -> bool:
    if not isinstance(ev, dict) or any(k not in ev for k in REQUIRED_FIELDS):
        return False
    try:
        for k in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            float(ev[k])
        int(ev["step"])
    except (TypeError, ValueError):
        return False
    return isinstance(ev["nameDest"], str) and isinstance(ev["type"], str)


class Processor:
    def __init__(
        self, store: Store, scorer: Scorer, allowed_lateness: int = 6, watermark: int = 0
    ) -> None:
        self.store = store
        self.scorer = scorer
        self.allowed_lateness = allowed_lateness
        self.watermark = watermark

    def process(self, events: list[dict], consumed_ms: int | None = None) -> BatchStats:
        consumed_ms = consumed_ms or now_ms()
        stats = BatchStats(received=len(events))

        valid = [e for e in events if is_valid(e)]
        stats.rejected = len(events) - len(valid)

        existing = self.store.existing_ids([e["event_id"] for e in valid])
        seen: set[str] = set()
        fresh: list[dict] = []
        dup_ids: list[str] = []
        for ev in valid:
            eid = ev["event_id"]
            if eid in existing or eid in seen:
                dup_ids.append(eid)
            else:
                seen.add(eid)
                fresh.append(ev)
        stats.duplicates = len(dup_ids)
        stats.new = len(fresh)

        if not fresh:
            if dup_ids:
                self.store.write_batch([], {}, dup_ids)
            return stats

        states = self.store.load_states(sorted({e["nameDest"] for e in fresh}))
        wm = self.watermark
        feats: list[dict] = []
        late_flags: list[bool] = []
        dirty: set[str] = set()
        for ev in fresh:
            ev["step"] = int(ev["step"])
            wm = max(wm, ev["step"])
            late = ev["step"] < wm - self.allowed_lateness
            dest = ev["nameDest"]
            feats.append(features_dict(feature_vector(ev, states.get(dest))))
            late_flags.append(late)
            if not late:
                states[dest] = update_state(states.get(dest), ev)
                dirty.add(dest)
        stats.late = sum(late_flags)

        scores, alerts = self.scorer.score(feats)
        scored_ms = now_ms()
        stats.alerts = sum(bool(a) for a in alerts)

        landed_ms = now_ms()
        rows = [
            {
                "event_id": ev["event_id"],
                "step": ev["step"],
                "type": ev["type"],
                "amount": float(ev["amount"]),
                "name_orig": ev.get("nameOrig"),
                "old_balance_orig": float(ev["oldbalanceOrg"]),
                "new_balance_orig": float(ev["newbalanceOrig"]),
                "name_dest": ev["nameDest"],
                "old_balance_dest": float(ev["oldbalanceDest"]),
                "new_balance_dest": float(ev["newbalanceDest"]),
                "is_fraud": ev.get("isFraud"),
                "features": f,
                "fraud_score": float(s),
                "alert": bool(a),
                "is_late": late,
                "produced_ms": ev.get("produced_ms"),
                "consumed_ms": consumed_ms,
                "scored_ms": scored_ms,
                "landed_ms": landed_ms,
            }
            for ev, f, s, a, late in zip(fresh, feats, scores, alerts, late_flags)
        ]
        self.store.write_batch(rows, {d: states[d] for d in dirty}, dup_ids)
        self.watermark = wm  # only advance once the batch is durably written
        return stats


def run(settings: Settings | None = None) -> None:  # pragma: no cover - needs Kafka + Postgres
    from confluent_kafka import Consumer

    from .kafka_utils import ensure_topic

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = settings or Settings()
    ensure_topic(cfg.kafka_bootstrap, cfg.topic, cfg.partitions)

    store = PostgresStore(cfg.database_url)
    for _attempt in range(30):
        try:
            store.init_schema()
            break
        except Exception as exc:
            log.warning("postgres not ready (%s), retrying", exc)
            store.reset_connection()
            time.sleep(2)
    else:
        raise SystemExit("could not connect to postgres")

    proc = Processor(
        store,
        HttpScorer(cfg.scoring_url),
        cfg.allowed_lateness_steps,
        watermark=store.max_step(),
    )
    consumer = Consumer(
        {
            "bootstrap.servers": cfg.kafka_bootstrap,
            "group.id": cfg.group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "fetch.wait.max.ms": 20,
        }
    )
    consumer.subscribe([cfg.topic])

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    total = BatchStats()
    last_log = time.time()
    log.info("processor started: topic=%s lateness=%d", cfg.topic, cfg.allowed_lateness_steps)
    try:
        while running:
            msgs = consumer.consume(num_messages=cfg.batch_size, timeout=cfg.fetch_wait_s)
            if not msgs:
                continue
            consumed_ms = now_ms()
            events: list[dict] = []
            bad = 0
            for m in msgs:
                if m.error():
                    log.error("kafka error: %s", m.error())
                    continue
                try:
                    events.append(json.loads(m.value()))
                except ValueError:
                    bad += 1
            while running:  # retry the SAME batch until it lands; nothing was committed yet
                try:
                    stats = proc.process(events, consumed_ms)
                    break
                except Exception:
                    log.exception("batch failed, retrying in 2s")
                    store.reset_connection()
                    time.sleep(2)
            else:
                break
            stats.rejected += bad
            consumer.commit(asynchronous=False)
            total.add(stats)
            if time.time() - last_log >= 5:
                log.info("totals: %s", total)
                last_log = time.time()
    finally:
        log.info("shutting down, totals: %s", total)
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    run()
