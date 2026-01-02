"""
Main script to create stint-level sequence datasets.

Usage:
    python -m ml_dataset.create_sequences
    python -m ml_dataset.create_sequences max_stints=100  # Test with subset
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_dataset.sequence_builder import SequenceBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Create stint-level sequence datasets."""

    logger.info("=" * 80)
    logger.info("SEQUENCE DATASET CREATION")
    logger.info("=" * 80)

    # Initialize builder
    builder = SequenceBuilder(cfg)

    # Build sequences
    max_stints = cfg.get("max_stints", None)
    num_sequences = builder.build_sequences(max_stints=max_stints)

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Created {num_sequences:,} stint sequences")
    logger.info("=" * 80)

    builder.close()


if __name__ == "__main__":
    main()
