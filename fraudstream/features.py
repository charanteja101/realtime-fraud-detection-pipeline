"""Feature engineering shared by offline training and the online stream processor.

The same two functions are used in both places, so there is no train/serve skew:

    feature_vector(txn, state)  -> features for this txn, using state BEFORE the txn
    update_state(state, txn)    -> new state including the txn

State is kept per destination account (``nameDest``). Everything in it is a commutative
aggregate (count, sum, max, per-step buckets), so events that arrive out of order within the
allowed lateness converge to the same state as in-order arrival.

State layout (a plain list, so it is compact in memory and JSON-serialisable for Postgres):

    [n, amount_sum, amount_max, last_step, n_transfer_in, window]
    window = [[step, count, amount_sum], ...]  sorted by step, only the last WINDOW_STEPS steps
"""
from __future__ import annotations

from typing import Any

TYPE_CODES = {"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4}
WINDOW_STEPS = 24  # PaySim: 1 step = 1 hour, so this is a 24h sliding window

FEATURE_ORDER = [
    # stateless, from the transaction itself
    "hour_of_day",
    "type_code",
    "amount",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "err_balance_orig",
    "err_balance_dest",
    "orig_drained",
    "amount_eq_old_balance",
    "dest_is_merchant",
    # stateful, destination-account history (computed incrementally)
    "dest_txn_count",
    "dest_total_in",
    "dest_avg_amount",
    "dest_max_amount",
    "dest_transfer_count",
    "dest_steps_since_last",
    "dest_txn_count_24h",
    "dest_amount_24h",
    "amount_to_dest_avg",
]

REQUIRED_FIELDS = (
    "event_id",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
)

_N, _SUM, _MAX, _LAST, _NTR, _WIN = range(6)


def new_state() -> list:
    return [0, 0.0, 0.0, -1, 0, []]


def feature_vector(txn: dict[str, Any], state: list | None) -> list[float]:
    """Features for ``txn`` given the destination state *before* this txn is applied."""
    step = txn["step"]
    amount = txn["amount"]
    old_o = txn["oldbalanceOrg"]
    new_o = txn["newbalanceOrig"]
    old_d = txn["oldbalanceDest"]
    new_d = txn["newbalanceDest"]

    if state is None:
        n, total, mx, last, n_tr, win = 0, 0.0, 0.0, -1, 0, ()
    else:
        n, total, mx, last, n_tr, win = state

    lo = step - WINDOW_STEPS
    cnt24 = 0
    amt24 = 0.0
    for b_step, b_cnt, b_sum in win:
        if lo < b_step <= step:
            cnt24 += b_cnt
            amt24 += b_sum

    avg = total / n if n else 0.0
    since = (step - last if step > last else 0) if last >= 0 else -1

    return [
        float(step % 24),
        float(TYPE_CODES.get(txn["type"], -1)),
        amount,
        old_o,
        new_o,
        old_d,
        new_d,
        new_o + amount - old_o,
        old_d + amount - new_d,
        1.0 if (old_o > 0 and new_o == 0) else 0.0,
        1.0 if (old_o > 0 and abs(amount - old_o) < 0.01) else 0.0,
        1.0 if txn["nameDest"].startswith("M") else 0.0,
        float(n),
        total,
        avg,
        mx,
        float(n_tr),
        float(since),
        float(cnt24),
        amt24,
        amount / (avg + 1.0),
    ]


def update_state(state: list | None, txn: dict[str, Any]) -> list:
    """Apply ``txn`` to the destination state (mutates and returns it)."""
    if state is None:
        state = new_state()
    step = txn["step"]
    amount = txn["amount"]

    state[_N] += 1
    state[_SUM] += amount
    if amount > state[_MAX]:
        state[_MAX] = amount
    if step > state[_LAST]:
        state[_LAST] = step
    if txn["type"] == "TRANSFER":
        state[_NTR] += 1

    win = state[_WIN]
    i = len(win)
    while i > 0 and win[i - 1][0] > step:  # find the slot (late events land mid-list)
        i -= 1
    if i > 0 and win[i - 1][0] == step:
        win[i - 1][1] += 1
        win[i - 1][2] += amount
    else:
        win.insert(i, [step, 1, amount])

    cutoff = state[_LAST] - WINDOW_STEPS
    while win and win[0][0] <= cutoff:
        win.pop(0)
    return state


def features_dict(vec: list[float]) -> dict[str, float]:
    return dict(zip(FEATURE_ORDER, vec))
