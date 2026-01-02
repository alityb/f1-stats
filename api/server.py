"""
F1 Stats API Server - Uvicorn entry point with Hydra configuration.

Usage:
    python -m api.server
    python -m api.server server.port=8080
    python -m api.server server.reload=true
"""

import logging
import sys
from pathlib import Path

import hydra
import uvicorn
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.dependencies import initialize_services, shutdown_services
from api.main import app, configure_cors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Run FastAPI server with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("F1 STATS API SERVER")
    logger.info("=" * 80)

    # Initialize services (model loader, database)
    logger.info("Initializing services...")
    initialize_services(cfg)

    # Configure CORS
    if cfg.api.cors.enabled:
        configure_cors(app, cfg.api.cors.origins)
        logger.info(f"CORS enabled for origins: {cfg.api.cors.origins}")

    # Setup startup/shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("API startup complete")
        logger.info(f"Docs available at http://{cfg.api.server.host}:{cfg.api.server.port}/docs")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down API...")
        shutdown_services()

    # Run server
    logger.info(f"Starting server at {cfg.api.server.host}:{cfg.api.server.port}")
    logger.info(f"Reload: {cfg.api.server.reload}, Workers: {cfg.api.server.workers}")

    uvicorn.run(
        "api.main:app",
        host=cfg.api.server.host,
        port=cfg.api.server.port,
        reload=cfg.api.server.reload,
        workers=cfg.api.server.workers,
        log_level=cfg.api.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
