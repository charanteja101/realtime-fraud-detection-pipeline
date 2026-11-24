"""Loads the XGBoost model + metadata and turns feature dicts into predictions.

Bring-your-own-model: drop ``model.json`` (an XGBoost Booster saved with ``save_model``) and an
optional ``model_meta.json`` into ``models/``. The meta file's ``feature_order`` must be a subset
of the features computed in ``features.py``; the API picks those columns by name.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_ORDER


class MissingFeatureError(ValueError):
    pass


class ModelBundle:
    def __init__(self, booster: Any, feature_order: list[str], threshold: float, version: str):
        self.booster = booster
        self.feature_order = feature_order
        self.threshold = threshold
        self.version = version

    @classmethod
    def load(
        cls,
        model_path: str | os.PathLike | None = None,
        meta_path: str | os.PathLike | None = None,
        threshold: float | None = None,
    ) -> ModelBundle:
        import xgboost as xgb

        model_path = Path(model_path or os.getenv("MODEL_PATH", "models/model.json"))
        meta_path = Path(meta_path or os.getenv("MODEL_META_PATH", "models/model_meta.json"))
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Train one with `python -m fraudstream.train` "
                "or copy your own XGBoost model.json into models/."
            )
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        order = meta.get("feature_order", FEATURE_ORDER)
        env_thr = os.getenv("ALERT_THRESHOLD")
        thr = threshold if threshold is not None else (
            float(env_thr) if env_thr else float(meta.get("threshold", 0.5))
        )
        return cls(booster, order, thr, meta.get("version", model_path.stem))

    def matrix(self, items: list[dict[str, float | None]]) -> np.ndarray:
        order = self.feature_order
        out = np.empty((len(items), len(order)), dtype=np.float32)
        for i, item in enumerate(items):
            missing = [f for f in order if f not in item]
            if missing:
                raise MissingFeatureError(f"item {i} is missing features: {missing}")
            out[i] = [np.nan if item[f] is None else item[f] for f in order]
        return out

    def predict(self, items: list[dict[str, float | None]]) -> np.ndarray:
        import xgboost as xgb

        if not items:
            return np.empty(0, dtype=np.float32)
        return self.booster.predict(xgb.DMatrix(self.matrix(items)))
