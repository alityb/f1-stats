"""
Debug script to inspect raw FastF1 data before database insertion.

Usage:
    python -m data_ingestion.debug_fastf1
"""

import sys
from pathlib import Path

import fastf1
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Inspect raw FastF1 data structure."""

    # Enable cache
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    print("=" * 80)
    print("FASTF1 DATA STRUCTURE DEBUG")
    print("=" * 80)

    # Load a single race
    print("\nLoading 2024 Bahrain GP...")
    session = fastf1.get_session(2024, 1, 'Race')
    session.load()

    laps = session.laps

    print(f"\n1. LAPS DATAFRAME INFO")
    print("-" * 80)
    print(f"Total laps: {len(laps)}")
    print(f"DataFrame shape: {laps.shape}")
    print(f"DataFrame type: {type(laps)}")

    print(f"\n2. AVAILABLE COLUMNS")
    print("-" * 80)
    print(laps.columns.tolist())

    print(f"\n3. COLUMN DATA TYPES")
    print("-" * 80)
    print(laps.dtypes)

    print(f"\n4. SAMPLE OF FIRST 5 LAPS (ALL COLUMNS)")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(laps.head())

    print(f"\n5. LAP TIME COLUMN INSPECTION")
    print("-" * 80)

    if 'LapTime' in laps.columns:
        print("✅ LapTime column exists")
        print(f"   Type: {laps['LapTime'].dtype}")
        print(f"   Non-null count: {laps['LapTime'].notna().sum()}/{len(laps)}")
        print(f"   Null count: {laps['LapTime'].isna().sum()}")
        print(f"\n   First 10 LapTime values:")
        print(laps['LapTime'].head(10))

        # Try conversion
        print(f"\n   Trying .dt.total_seconds() conversion:")
        lap_time_seconds = laps['LapTime'].dt.total_seconds()
        print(f"   Result type: {lap_time_seconds.dtype}")
        print(f"   Non-null after conversion: {lap_time_seconds.notna().sum()}/{len(lap_time_seconds)}")
        print(f"   First 10 converted values:")
        print(lap_time_seconds.head(10))
    else:
        print("❌ LapTime column NOT found!")

    print(f"\n6. CHECKING FOR VALID LAP TIMES")
    print("-" * 80)

    # Filter for valid laps
    valid_laps = laps[laps['LapTime'].notna()]
    print(f"Laps with non-null LapTime: {len(valid_laps)}")

    if len(valid_laps) > 0:
        print("\nSample valid lap:")
        sample = valid_laps.iloc[0]
        print(f"  Driver: {sample['Driver']}")
        print(f"  LapNumber: {sample['LapNumber']}")
        print(f"  LapTime: {sample['LapTime']}")
        print(f"  LapTime (seconds): {sample['LapTime'].total_seconds()}")
        print(f"  TrackStatus: {sample['TrackStatus']}")
        print(f"  IsAccurate: {sample['IsAccurate']}")

    print(f"\n7. CHECKING PIT LAP COLUMNS")
    print("-" * 80)

    pit_cols = ['PitInTime', 'PitOutTime']
    for col in pit_cols:
        if col in laps.columns:
            print(f"✅ {col} exists: {laps[col].notna().sum()} non-null values")
        else:
            print(f"❌ {col} NOT found")

    print(f"\n8. RAW DATAFRAME SAMPLE (Key Columns)")
    print("-" * 80)

    key_cols = ['Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound',
                'TyreLife', 'TrackStatus', 'IsAccurate']
    available_key_cols = [col for col in key_cols if col in laps.columns]
    print(laps[available_key_cols].head(20))


if __name__ == "__main__":
    main()
