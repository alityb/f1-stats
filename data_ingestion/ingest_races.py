"""
Main data ingestion script for F1 race data.

Usage:
    python -m data_ingestion.ingest_races
    python -m data_ingestion.ingest_races seasons=[2023,2024]
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.duckdb_persister import DuckDBPersister
from data_ingestion.fastf1_loader import FastF1Loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main ingestion pipeline.

    Loads F1 race data from FastF1 and persists to DuckDB.
    """
    logger.info("=" * 80)
    logger.info("F1 Race Data Ingestion Pipeline")
    logger.info("=" * 80)

    # Initialize components
    loader = FastF1Loader(cfg)
    persister = DuckDBPersister(cfg)

    # Get seasons to process
    seasons = cfg.seasons
    session_types = cfg.data.session_types

    logger.info(f"Processing seasons: {seasons}")
    logger.info(f"Session types: {session_types}")

    total_races_processed = 0
    total_laps_inserted = 0

    # Process each season
    for season in seasons:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {season} season")
        logger.info(f"{'=' * 80}")

        for session_type in session_types:
            # Load all races for the season
            sessions_data = loader.load_season_races(season, session_type)

            if not sessions_data:
                logger.warning(f"No {session_type} sessions found for {season}")
                continue

            # Process each race
            logger.info(f"\nPersisting {len(sessions_data)} {session_type} sessions to database...")
            for session, metadata in tqdm(sessions_data, desc=f"{season} {session_type}"):
                try:
                    # Get or create race_id
                    race_id = persister.get_or_create_race_id(metadata)

                    # Extract and persist laps
                    laps_df = loader.extract_laps_dataframe(session)
                    laps_inserted = persister.persist_laps(laps_df, race_id)

                    total_laps_inserted += laps_inserted
                    if laps_inserted > 0:
                        total_races_processed += 1

                except Exception as e:
                    logger.error(
                        f"Error processing {metadata['event_name']} {session_type}: {e}",
                        exc_info=True
                    )
                    continue

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Ingestion Complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Races processed: {total_races_processed}")
    logger.info(f"Laps inserted: {total_laps_inserted:,}")
    logger.info(f"Total races in DB: {persister.get_race_count()}")
    logger.info(f"Total laps in DB: {persister.get_lap_count():,}")

    # Close connection
    persister.close()


if __name__ == "__main__":
    main()
