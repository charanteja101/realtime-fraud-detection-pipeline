"""Model-serving API.

    POST /score        {"features": {...}}          -> score + alert flag for one transaction
    POST /score/batch  {"items": [{...}, ...]}      -> scores + alert flags (used by the processor)
    GET  /health, GET /model

The model is loaded once at startup. ``def`` (not ``async def``) endpoints run in FastAPI's thread
pool, so XGBoost inference does not block the event loop.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .model import MissingFeatureError, ModelBundle

FeatureMap = dict[str, float | None]


class ScoreRequest(BaseModel):
    features: FeatureMap


class ScoreResponse(BaseModel):
    fraud_score: float
    alert: bool
    threshold: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    items: list[FeatureMap] = Field(max_length=5000)


class BatchResponse(BaseModel):
    scores: list[float]
    alerts: list[bool]
    threshold: float
    model_version: str
    latency_ms: float


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = bundle or ModelBundle.load()
        yield

    app = FastAPI(title="Fraud scoring API", version="1.0", lifespan=lifespan)

    def _predict(request: Request, items: list[FeatureMap]):
        model: ModelBundle = request.app.state.model
        t0 = time.perf_counter()
        try:
            scores = [float(s) for s in model.predict(items)]
        except MissingFeatureError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ms = (time.perf_counter() - t0) * 1000
        return model, scores, ms

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request):
        model, scores, ms = _predict(request, [req.features])
        return ScoreResponse(
            fraud_score=scores[0],
            alert=scores[0] >= model.threshold,
            threshold=model.threshold,
            model_version=model.version,
            latency_ms=round(ms, 3),
        )

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(req: BatchRequest, request: Request):
        model, scores, ms = _predict(request, req.items)
        return BatchResponse(
            scores=scores,
            alerts=[s >= model.threshold for s in scores],
            threshold=model.threshold,
            model_version=model.version,
            latency_ms=round(ms, 3),
        )

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.model.version}

    @app.get("/model")
    def model_info(request: Request):
        m: ModelBundle = request.app.state.model
        return {"version": m.version, "threshold": m.threshold, "feature_order": m.feature_order}

    return app


app = create_app()
