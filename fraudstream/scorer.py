"""Scorers used by the stream processor: HTTP (production path) and local (tests/offline)."""
from __future__ import annotations

import logging
import time
from typing import Protocol

log = logging.getLogger(__name__)


class Scorer(Protocol):
    def score(self, features: list[dict]) -> tuple[list[float], list[bool]]: ...


class HttpScorer:
    """Calls the FastAPI service's /score/batch endpoint."""

    def __init__(self, base_url: str, timeout_s: float = 10.0, retries: int = 3):
        import httpx

        self._http_error = httpx.HTTPError
        self.client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self.retries = retries

    def score(self, features: list[dict]) -> tuple[list[float], list[bool]]:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.client.post("/score/batch", json={"items": features})
                r.raise_for_status()
                body = r.json()
                return body["scores"], body["alerts"]
            except (self._http_error, KeyError) as exc:
                last = exc
                log.warning("scoring attempt %d failed: %s", attempt + 1, exc)
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"scoring service unavailable: {last}")


class LocalScorer:
    """In-process scoring with a ModelBundle (no HTTP). Useful for tests and notebooks."""

    def __init__(self, bundle):
        self.bundle = bundle

    def score(self, features: list[dict]) -> tuple[list[float], list[bool]]:
        scores = [float(s) for s in self.bundle.predict(features)]
        return scores, [s >= self.bundle.threshold for s in scores]
