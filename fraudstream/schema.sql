CREATE TABLE IF NOT EXISTS transactions (
    event_id          TEXT PRIMARY KEY,            -- idempotency key: duplicates are rejected here
    step              INTEGER NOT NULL,
    type              TEXT NOT NULL,
    amount            DOUBLE PRECISION NOT NULL,
    name_orig         TEXT,
    old_balance_orig  DOUBLE PRECISION,
    new_balance_orig  DOUBLE PRECISION,
    name_dest         TEXT NOT NULL,
    old_balance_dest  DOUBLE PRECISION,
    new_balance_dest  DOUBLE PRECISION,
    is_fraud          SMALLINT,                    -- ground truth label (for evaluation only)
    features          JSONB,                       -- exact features the model saw
    fraud_score       REAL,
    alert             BOOLEAN,
    is_late           BOOLEAN NOT NULL DEFAULT FALSE,
    produced_ms       BIGINT,                      -- producer wall clock when sent
    consumed_ms       BIGINT,                      -- processor fetched the batch
    scored_ms         BIGINT,                      -- scoring response received
    landed_ms         BIGINT                       -- just before COMMIT of the landing txn
);
CREATE INDEX IF NOT EXISTS idx_transactions_step ON transactions (step);

CREATE TABLE IF NOT EXISTS dest_state (
    dest        TEXT PRIMARY KEY,
    state       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS duplicate_events (
    id        BIGSERIAL PRIMARY KEY,
    event_id  TEXT NOT NULL,
    seen_ms   BIGINT NOT NULL
);
