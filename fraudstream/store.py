"""Storage for landed transactions, per-destination feature state and duplicate log.

``PostgresStore.write_batch`` writes transactions, state updates and duplicate records in ONE
transaction. Together with "commit Kafka offsets only after the DB commit" and the event_id
primary key, a crash anywhere gives effectively-once results: a replayed batch is recognised as
duplicates and does not touch the state again.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Protocol

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TX_COLUMNS = [
    "event_id", "step", "type", "amount", "name_orig", "old_balance_orig", "new_balance_orig",
    "name_dest", "old_balance_dest", "new_balance_dest", "is_fraud", "features", "fraud_score",
    "alert", "is_late", "produced_ms", "consumed_ms", "scored_ms", "landed_ms",
]  # fmt: skip


class Store(Protocol):
    def init_schema(self) -> None: ...
    def existing_ids(self, ids: list[str]) -> set[str]: ...
    def load_states(self, dests: list[str]) -> dict[str, list]: ...
    def write_batch(self, rows: list[dict], states: dict[str, list], dup_ids: list[str]) -> None: ...
    def max_step(self) -> int: ...
    def reset_connection(self) -> None: ...


class MemoryStore:
    """In-memory Store with the same semantics as Postgres (used by unit tests)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.states: dict[str, list] = {}
        self.duplicates: list[str] = []

    def init_schema(self) -> None:
        pass

    def existing_ids(self, ids: list[str]) -> set[str]:
        return {i for i in ids if i in self.rows}

    def load_states(self, dests: list[str]) -> dict[str, list]:
        # deep copy mimics a JSON round trip: callers cannot mutate stored state by accident
        return {d: copy.deepcopy(self.states[d]) for d in dests if d in self.states}

    def write_batch(self, rows: list[dict], states: dict[str, list], dup_ids: list[str]) -> None:
        for r in rows:
            self.rows.setdefault(r["event_id"], dict(r))
        for d, s in states.items():
            self.states[d] = copy.deepcopy(s)
        self.duplicates.extend(dup_ids)

    def max_step(self) -> int:
        return max((r["step"] for r in self.rows.values()), default=0)

    def reset_connection(self) -> None:
        pass


_INSERT_TX = (
    f"INSERT INTO transactions ({', '.join(TX_COLUMNS)}) "
    f"VALUES ({', '.join('%(' + c + ')s' for c in TX_COLUMNS)}) "
    "ON CONFLICT (event_id) DO NOTHING"
)
_UPSERT_STATE = (
    "INSERT INTO dest_state (dest, state) VALUES (%s, %s) "
    "ON CONFLICT (dest) DO UPDATE SET state = EXCLUDED.state, updated_at = now()"
)
_INSERT_DUP = "INSERT INTO duplicate_events (event_id, seen_ms) VALUES (%s, %s)"


class PostgresStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: Any = None

    @property
    def conn(self):
        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def reset_connection(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None

    def init_schema(self) -> None:
        self.conn.execute(SCHEMA_PATH.read_text())

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        cur = self.conn.execute(
            "SELECT event_id FROM transactions WHERE event_id = ANY(%s)", (ids,)
        )
        return {r[0] for r in cur.fetchall()}

    def load_states(self, dests: list[str]) -> dict[str, list]:
        if not dests:
            return {}
        cur = self.conn.execute("SELECT dest, state FROM dest_state WHERE dest = ANY(%s)", (dests,))
        return {d: s for d, s in cur.fetchall()}

    def write_batch(self, rows: list[dict], states: dict[str, list], dup_ids: list[str]) -> None:
        from psycopg.types.json import Jsonb

        now = int(time.time() * 1000)
        with self.conn.transaction(), self.conn.cursor() as cur:
            if rows:
                cur.executemany(
                    _INSERT_TX, [{**r, "features": Jsonb(r["features"])} for r in rows]
                )
            if states:
                cur.executemany(_UPSERT_STATE, [(d, Jsonb(s)) for d, s in states.items()])
            if dup_ids:
                cur.executemany(_INSERT_DUP, [(e, now) for e in dup_ids])

    def max_step(self) -> int:
        row = self.conn.execute("SELECT coalesce(max(step), 0) FROM transactions").fetchone()
        return int(row[0])
