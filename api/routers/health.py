"""
Health check endpoints for F1 Stats API.
"""

import logging

from fastapi import APIRouter, Depends

from api.dependencies import get_db_persister, get_model_loader
from api.schemas.responses import HealthResponse, ModelInfoResponse
from data_ingestion.duckdb_persister import DuckDBPersister
from metrics.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    db: DuckDBPersister = Depends(get_db_persister),
):
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    model_loaded = model_loader.model is not None
    db_connected = db.conn is not None

    status = "healthy" if (model_loaded and db_connected) else "degraded"

    return HealthResponse(
        status=status, model_loaded=model_loaded, db_connected=db_connected
    )


@router.get("/model", response_model=ModelInfoResponse)
async def model_info(
    model_loader: ModelLoader = Depends(get_model_loader),
):
    """
    Get model information.

    Returns:
        Model checkpoint and configuration info
    """
    checkpoint_info = model_loader.checkpoint_info
    vocab_sizes = model_loader.vocab_sizes

    return ModelInfoResponse(
        checkpoint_path=checkpoint_info["path"],
        epoch=checkpoint_info.get("epoch"),
        best_val_metric=checkpoint_info.get("best_val_metric"),
        vocab_sizes=vocab_sizes,
    )
