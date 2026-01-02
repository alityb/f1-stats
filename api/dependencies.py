"""
FastAPI dependencies for F1 Stats API.

Provides dependency injection for model loader, database, and metric services.
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import Depends
from omegaconf import DictConfig, OmegaConf

from data_ingestion.duckdb_persister import DuckDBPersister
from metrics.computation.dab import DABComputer
from metrics.computation.tme import TMEComputer
from metrics.computation.tpi import TPIComputer
from metrics.inference.model_loader import ModelLoader
from metrics.inference.predictor import PredictionService

logger = logging.getLogger(__name__)


# Global singletons (initialized at startup)
_model_loader: Optional[ModelLoader] = None
_db_persister: Optional[DuckDBPersister] = None
_config: Optional[DictConfig] = None


def initialize_services(config: DictConfig):
    """
    Initialize global services at API startup.

    Args:
        config: Hydra configuration
    """
    global _model_loader, _db_persister, _config

    logger.info("Initializing F1 Stats API services...")

    _config = config

    # Initialize database
    logger.info("Connecting to database...")
    _db_persister = DuckDBPersister(config)
    logger.info("Database connected")

    # Initialize model loader (loads model at startup)
    logger.info("Loading model...")
    _model_loader = ModelLoader.get_instance(config)
    logger.info("Model loaded successfully")


def shutdown_services():
    """Cleanup services at API shutdown."""
    global _model_loader, _db_persister

    logger.info("Shutting down F1 Stats API services...")

    if _db_persister:
        _db_persister.close()

    # Reset singletons
    _model_loader = None
    _db_persister = None


@lru_cache()
def get_config() -> DictConfig:
    """Get Hydra configuration."""
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_services().")
    return _config


def get_model_loader() -> ModelLoader:
    """Get ModelLoader singleton."""
    if _model_loader is None:
        raise RuntimeError("ModelLoader not initialized. Call initialize_services().")
    return _model_loader


def get_db_persister() -> DuckDBPersister:
    """Get DuckDBPersister singleton."""
    if _db_persister is None:
        raise RuntimeError("Database not initialized. Call initialize_services().")
    return _db_persister


def get_prediction_service(
    model_loader: ModelLoader = Depends(get_model_loader),
    db: DuckDBPersister = Depends(get_db_persister),
) -> PredictionService:
    """Get PredictionService instance."""
    return PredictionService(model_loader, db)


def get_tpi_computer(
    predictor: PredictionService = Depends(get_prediction_service),
) -> TPIComputer:
    """Get TPIComputer instance."""
    return TPIComputer(predictor)


def get_dab_computer(
    predictor: PredictionService = Depends(get_prediction_service),
    db: DuckDBPersister = Depends(get_db_persister),
) -> DABComputer:
    """Get DABComputer instance."""
    return DABComputer(predictor, db)


def get_tme_computer(
    predictor: PredictionService = Depends(get_prediction_service),
    db: DuckDBPersister = Depends(get_db_persister),
) -> TMEComputer:
    """Get TMEComputer instance."""
    return TMEComputer(predictor, db)
