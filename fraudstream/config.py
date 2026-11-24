"""Runtime settings, read from environment variables (see docker-compose.yml)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:19092"))
    topic: str = field(default_factory=lambda: _env("KAFKA_TOPIC", "transactions"))
    partitions: int = field(default_factory=lambda: int(_env("KAFKA_PARTITIONS", "4")))
    group_id: str = field(default_factory=lambda: _env("KAFKA_GROUP_ID", "fraud-processor"))
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "postgresql://fraud:fraud@localhost:5432/fraud")
    )
    scoring_url: str = field(default_factory=lambda: _env("SCORING_URL", "http://localhost:8000"))
    batch_size: int = field(default_factory=lambda: int(_env("BATCH_SIZE", "500")))
    fetch_wait_s: float = field(default_factory=lambda: float(_env("FETCH_WAIT_S", "0.05")))
    # Events whose step is more than this many steps behind the watermark are "late":
    # they are scored and landed, but do NOT update the destination state.
    allowed_lateness_steps: int = field(
        default_factory=lambda: int(_env("ALLOWED_LATENESS_STEPS", "6"))
    )
