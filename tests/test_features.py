import itertools
import json

from fraudstream.features import (
    FEATURE_ORDER,
    WINDOW_STEPS,
    feature_vector,
    features_dict,
    update_state,
)

from .helpers import make_event


def f(txn, state):
    return features_dict(feature_vector(txn, state))


def test_new_destination_has_empty_history():
    out = f(make_event(step=5, amount=50.0), None)
    assert out["dest_txn_count"] == 0
    assert out["dest_total_in"] == 0
    assert out["dest_steps_since_last"] == -1
    assert out["orig_drained"] == 1.0
    assert out["hour_of_day"] == 5
    assert len(feature_vector(make_event(), None)) == len(FEATURE_ORDER)


def test_history_accumulates_incrementally():
    st = None
    e1 = make_event(step=3, amount=100.0)
    e2 = make_event(step=5, amount=300.0, type_="CASH_OUT")
    st = update_state(st, e1)
    out = f(e2, st)
    assert out["dest_txn_count"] == 1
    assert out["dest_total_in"] == 100.0
    assert out["dest_avg_amount"] == 100.0
    assert out["dest_transfer_count"] == 1
    assert out["dest_steps_since_last"] == 2
    assert out["dest_txn_count_24h"] == 1
    st = update_state(st, e2)
    out3 = f(make_event(step=6, amount=10.0), st)
    assert out3["dest_txn_count"] == 2
    assert out3["dest_max_amount"] == 300.0
    assert out3["dest_transfer_count"] == 1  # the CASH_OUT is not a transfer


def test_features_use_state_before_the_event():
    e = make_event(step=2, amount=70.0)
    before = f(e, None)
    st = update_state(None, e)
    assert before["dest_txn_count"] == 0  # the event must not see itself
    assert f(e, st)["dest_txn_count"] == 1


def test_sliding_window_excludes_old_steps_and_prunes_state():
    st = None
    for step in (1, 10, 30):
        st = update_state(st, make_event(step=step, amount=10.0))
    out = f(make_event(step=30, amount=1.0), st)
    assert out["dest_txn_count"] == 3  # lifetime count keeps everything
    assert out["dest_txn_count_24h"] == 2  # steps 10 and 30 are inside (6, 30]
    assert all(b[0] > 30 - WINDOW_STEPS for b in st[5])  # step 1 was pruned from state


def test_out_of_order_events_converge_to_same_state():
    evs = [make_event(step=s, amount=a) for s, a in [(10, 5.0), (11, 7.0), (11, 3.0), (13, 9.0)]]
    states = []
    for perm in itertools.permutations(evs):
        st = None
        for e in perm:
            st = update_state(st, e)
        states.append(json.dumps(st, sort_keys=True))
    assert len(set(states)) == 1


def test_state_survives_json_round_trip():
    st = None
    for s in (1, 2, 2, 3):
        st = update_state(st, make_event(step=s, amount=4.0))
    restored = json.loads(json.dumps(st))
    e = make_event(step=4, amount=4.0)
    assert feature_vector(e, restored) == feature_vector(e, st)
    assert update_state(restored, e) == update_state(st, e)


def test_merchant_flag_and_unknown_type():
    out = f(make_event(dest="M123", type_="PAYMENT"), None)
    assert out["dest_is_merchant"] == 1.0 and out["type_code"] == 0
    assert f(make_event(type_="WEIRD"), None)["type_code"] == -1
