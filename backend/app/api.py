"""api.py — FastAPI route definitions.

Error handling is delegated entirely to the global middleware registered in
core/error_handlers.py.  Individual route handlers raise typed exceptions
(ValidationError, NotFoundError, etc.) and let the middleware map them to the
correct HTTP status codes.

No catch-all ``except Exception → HTTP 400`` exists here.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.ml.travel_time_model import TravelTimeEstimator
from app.schemas import (
    MapLayoutPayload,
    MapListResponse,
    OptimizationRequest,
    OptimizationResponse,
    StoredMap,
    TrainTravelTimeRequest,
    TrainTravelTimeResponse,
)
from app.services.layout_store import LayoutStore
from app.services.optimizer_service import optimize_orders

router = APIRouter(prefix="/api/v1", tags=["optimization"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _cached_estimator(model_path: str) -> TravelTimeEstimator:
    return TravelTimeEstimator(model_path=model_path)


def _estimator(settings: Settings = Depends(get_settings)) -> TravelTimeEstimator:
    return _cached_estimator(settings.model_path)


def _layout_store(settings: Settings = Depends(get_settings)) -> LayoutStore:
    return LayoutStore(file_path=settings.layout_store_path)


@router.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize(
    payload: OptimizationRequest,
    estimator: TravelTimeEstimator = Depends(_estimator),
) -> OptimizationResponse | JSONResponse:
    """Run warehouse order-picking optimisation.

    Raises:
        asyncio.CancelledError: Returned as HTTP 499 (client disconnect).
        Any BaseAppException subclass: Handled by global middleware.
        Exception: Caught by global middleware and returned as HTTP 500.
    """
    try:
        return await run_in_threadpool(optimize_orders, payload, estimator)
    except asyncio.CancelledError:
        logger.info("optimize request cancelled by client")
        return JSONResponse(status_code=499, content={"detail": "Optimization request cancelled"})


@router.post("/ml/train", response_model=TrainTravelTimeResponse)
async def train_travel_model(
    payload: TrainTravelTimeRequest,
    estimator: TravelTimeEstimator = Depends(_estimator),
) -> TrainTravelTimeResponse | JSONResponse:
    """Retrain the travel-time estimation model on new samples."""
    try:
        report = await run_in_threadpool(estimator.train, payload.samples)
        return TrainTravelTimeResponse(
            model_path=report.model_path,
            r2=report.r2,
            mae=report.mae,
            rmse=report.rmse,
            evaluation_method=report.evaluation_method,
            cv_r2_mean=report.cv_r2_mean,
            cv_mae_mean=report.cv_mae_mean,
            cv_rmse_mean=report.cv_rmse_mean,
        )
    except asyncio.CancelledError:
        logger.info("train request cancelled by client")
        return JSONResponse(status_code=499, content={"detail": "Training request cancelled"})


@router.post("/maps", response_model=StoredMap)
def save_map(payload: MapLayoutPayload, store: LayoutStore = Depends(_layout_store)) -> StoredMap:
    """Persist a warehouse map layout."""
    return store.save_map(payload)


@router.get("/maps", response_model=MapListResponse)
def list_maps(store: LayoutStore = Depends(_layout_store)) -> MapListResponse:
    """List all stored warehouse map layouts."""
    return MapListResponse(maps=store.list_maps())


@router.get("/maps/{map_id}", response_model=StoredMap)
def get_map(map_id: str, store: LayoutStore = Depends(_layout_store)) -> StoredMap:
    """Retrieve a warehouse map layout by ID."""
    found = store.get_map(map_id)
    if not found:
        raise NotFoundError(f"Map '{map_id}' not found.")
    return found
