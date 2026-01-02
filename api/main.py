"""
F1 Stats API - Main FastAPI application.

Provides REST API for F1 race pace decomposition metrics:
- TPI (Track Position Index): Clean air pace
- DAB (Driver Ability Benchmark): Car-normalized driver contribution
- TME (Tyre Management Efficiency): Tyre degradation management

Usage:
    python -m api.server
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import drivers, health, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="F1 Stats API",
    description="Race pace decomposition metrics and driver comparisons",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def configure_cors(app: FastAPI, origins: list):
    """Configure CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_routers(app: FastAPI):
    """Setup API routers."""
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(
        metrics.router, prefix="/api/v1/metrics", tags=["Metrics"]
    )
    app.include_router(
        drivers.router, prefix="/api/v1/drivers", tags=["Drivers"]
    )


# Setup routers
setup_routers(app)

logger.info("F1 Stats API application created")
