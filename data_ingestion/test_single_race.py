"""
Test script to load a single race and check data flow.

Usage:
    python -m data_ingestion.test_single_race
"""

import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.duckdb_persister import DuckDBPersister
from data_ingestion.fastf1_loader import FastF1Loader


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Test loading a single race and inspect the data."""

    print("=" * 80)
    print("SINGLE RACE DATA FLOW TEST")
    print("=" * 80)

    # Load single race
    loader = FastF1Loader(cfg)
    print("\nStep 1: Loading 2024 Bahrain GP from FastF1...")
    session = loader.load_session(2024, 1, "Race")

    if session is None:
        print("❌ Failed to load session")
        return

    print("✅ Session loaded")

    # Extract laps
    print("\nStep 2: Extracting laps dataframe...")
    laps_df = loader.extract_laps_dataframe(session)
    print(f"✅ Extracted {len(laps_df)} laps")

    # Check LapTime in raw data
    print("\nStep 3: Inspecting LapTime column in raw FastF1 data...")
    print(f"   LapTime type: {laps_df['LapTime'].dtype}")
    print(f"   Non-null LapTime: {laps_df['LapTime'].notna().sum()}/{len(laps_df)}")
    print(f"   Sample LapTime values:")
    print(laps_df[['Driver', 'LapNumber', 'LapTime']].head(10))

    # Try manual conversion
    print("\nStep 4: Testing manual conversion to seconds...")
    lap_time_seconds = laps_df['LapTime'].dt.total_seconds()
    print(f"   Conversion result type: {lap_time_seconds.dtype}")
    print(f"   Non-null after conversion: {lap_time_seconds.notna().sum()}/{len(lap_time_seconds)}")
    print(f"   Sample converted values:")
    print(lap_time_seconds.head(10))

    # Get metadata
    print("\nStep 5: Extracting race metadata...")
    metadata = loader.extract_race_metadata(session)
    print(f"✅ Metadata: {metadata}")

    # Now test database insertion
    print("\nStep 6: Testing database insertion...")
    persister = DuckDBPersister(cfg)

    # Get race_id
    race_id = persister.get_or_create_race_id(metadata)
    print(f"✅ race_id: {race_id}")

    # Prepare laps for insertion (use internal method)
    print("\nStep 7: Preparing laps for database...")
    laps_prepared = persister._prepare_laps_dataframe(laps_df, race_id)
    print(f"✅ Prepared {len(laps_prepared)} laps")

    # Check lap_time_seconds in prepared data
    print("\nStep 8: Checking lap_time_seconds in prepared data...")
    print(f"   Type: {laps_prepared['lap_time_seconds'].dtype}")
    print(f"   Non-null: {laps_prepared['lap_time_seconds'].notna().sum()}/{len(laps_prepared)}")
    print(f"\n   Sample prepared data:")
    print(laps_prepared[['driver_code', 'lap_number', 'lap_time_seconds', 'lap_time_ms']].head(10))

    # Check what's actually in the database for this race
    print("\nStep 9: Checking what's in database for this race_id...")
    db_laps = persister.conn.execute(f"""
        SELECT driver_code, lap_number, lap_time_seconds, lap_time_ms, track_status
        FROM laps_raw
        WHERE race_id = {race_id}
        ORDER BY lap_number
        LIMIT 10
    """).fetchdf()

    print(db_laps)

    persister.close()


if __name__ == "__main__":
    main()
