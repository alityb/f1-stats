"""
Main script to build ML features from raw lap data.

Usage:
    python -m feature_engineering.build_features
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_engineering.feature_builder import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Build ML features from raw lap data."""

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)

    # Initialize feature builder
    builder = FeatureBuilder(cfg)

    # Step 1: Create features table
    logger.info("\nStep 1: Creating laps_ml_features table...")
    builder.create_features_table()

    # Step 2: Enhance traffic detection
    logger.info("\nStep 2: Enhancing traffic detection...")
    builder.enhance_traffic_detection()

    # Step 3: Get statistics
    logger.info("\nStep 3: Computing feature statistics...")
    stats = builder.get_feature_stats()

    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nTotal laps with features: {stats['total_laps']:,}")
    logger.info(f"Valid laps for training: {stats['valid_laps']:,}")
    logger.info(f"Valid lap percentage: {stats['valid_pct']:.2f}%")
    logger.info(f"\nTotal stints: {stats['total_stints']:,}")
    logger.info(f"Average stint length: {stats['avg_stint_length']:.1f} laps")

    logger.info("\n" + "-" * 80)
    logger.info("LAPS BY COMPOUND")
    logger.info("-" * 80)
    print(stats['by_compound'].to_string(index=False))

    logger.info("\n" + "-" * 80)
    logger.info("LAPS BY SEASON")
    logger.info("-" * 80)
    print(stats['by_season'].to_string(index=False))

    # Sample features
    logger.info("\n" + "-" * 80)
    logger.info("SAMPLE FEATURES (First 10 Valid Laps)")
    logger.info("-" * 80)

    sample = builder.conn.execute("""
        SELECT
            season_id,
            track_id,
            driver_id,
            stint_index,
            lap_in_stint,
            ROUND(lap_time_seconds, 3) as lap_time_sec,
            compound,
            tyre_age,
            ROUND(fuel_proxy, 3) as fuel_proxy,
            clean_air_flag,
            valid_lap_flag
        FROM laps_ml_features
        WHERE valid_lap_flag = TRUE
        LIMIT 10
    """).fetchdf()

    print(sample.to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("✅ Feature engineering complete!")
    logger.info("=" * 80)

    builder.close()


if __name__ == "__main__":
    main()
