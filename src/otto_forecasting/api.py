from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from otto_forecasting.data import FEATURE_COLUMNS, add_time_features
from otto_forecasting.dataset import inverse_targets
from otto_forecasting.model import DemandTransformer


class Observation(BaseModel):
    timestamp: str
    clicks: float = Field(ge=0)
    carts: float = Field(ge=0)
    orders: float = Field(ge=0)


class ForecastRequest(BaseModel):
    history: list[Observation]


class ForecastPoint(BaseModel):
    forecast_hour: int
    carts: float
    orders: float


class ForecastResponse(BaseModel):
    forecasts: list[ForecastPoint]


class ModelService:
    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing model artifacts in {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_scaler = joblib.load(root / "feature_scaler.joblib")
        self.target_scaler = joblib.load(root / "target_scaler.joblib")
        model_config = self.metadata["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DemandTransformer(
            num_features=len(self.metadata["feature_columns"]),
            target_dim=len(self.metadata["target_columns"]),
            horizon=int(self.metadata["horizon"]),
            **model_config,
        ).to(self.device)
        state = torch.load(root / "transformer.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def forecast(self, history: list[Observation]) -> np.ndarray:
        lookback = int(self.metadata["lookback"])
        if len(history) != lookback:
            raise ValueError(f"Expected exactly {lookback} observations")
        frame = pd.DataFrame([item.model_dump() for item in history])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("History timestamps must be unique and chronological")
        expected_frequency = timestamps.diff().dropna()
        if not expected_frequency.eq(pd.Timedelta(hours=1)).all():
            raise ValueError("History must contain consecutive hourly observations")
        frame = add_time_features(frame)
        values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        scaled = self.feature_scaler.transform(values).astype(np.float32)
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(tensor).cpu().numpy()
        return inverse_targets(prediction, self.target_scaler)[0]


_service: ModelService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _service
    _service = ModelService(os.getenv("OTTO_ARTIFACT_DIR", "artifacts"))
    yield
    _service = None


app = FastAPI(
    title="OTTO Demand Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        prediction = _service.forecast(request.history)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    points = [
        ForecastPoint(
            forecast_hour=index + 1,
            carts=float(values[0]),
            orders=float(values[1]),
        )
        for index, values in enumerate(prediction)
    ]
    return ForecastResponse(forecasts=points)
