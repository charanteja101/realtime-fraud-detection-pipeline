"""Runs only when TEST_DATABASE_URL is set (CI provides a Postgres service container)."""
import os

import pytest

pytest.importorskip("psycopg")
URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

from fraudstream.processor import Processor  # noqa: E402
from fraudstream.store import PostgresStore  # noqa: E402

from .helpers import StubScorer, make_event  # noqa: E402


def fresh_store() -> PostgresStore:
    s = PostgresStore(URL)
    s.init_schema()
    s.conn.execute("TRUNCATE transactions, dest_state, duplicate_events")
    return s


def test_end_to_end_with_postgres_store():
    store = fresh_store()
    proc = Processor(store, StubScorer(), allowed_lateness=6)
    a = make_event(step=20, dest="C1", amount=10.0, eid="a")
    b = make_event(step=21, dest="C1", amount=30.0, eid="b")
    late = make_event(step=2, dest="C1", amount=5.0, eid="late")
    proc.process([a, b, dict(a)])
    proc.process([dict(b), late])

    n = store.conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    d = store.conn.execute("SELECT count(*) FROM duplicate_events").fetchone()[0]
    is_late = store.conn.execute("SELECT is_late FROM transactions WHERE event_id='late'").fetchone()
    assert (n, d, is_late[0]) == (3, 2, True)
    assert store.load_states(["C1"])["C1"][0] == 2  # a and b only
    assert store.existing_ids(["a", "zzz"]) == {"a"}
    assert store.max_step() == 21
    feats = store.conn.execute("SELECT features FROM transactions WHERE event_id='b'").fetchone()[0]
    assert feats["dest_txn_count"] == 1
