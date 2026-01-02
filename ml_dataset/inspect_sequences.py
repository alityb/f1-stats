"""
Inspection tool for stint sequences.

Usage:
    python -m ml_dataset.inspect_sequences
"""

import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Inspect stint sequences."""

    print("\n" + "=" * 80)
    print("STINT SEQUENCES INSPECTION")
    print("=" * 80)

    # Load sequences
    parquet_path = Path(cfg.paths.stints_dir) / "stint_sequences.parquet"

    if not parquet_path.exists():
        print(f"\n❌ Sequences not found at {parquet_path}")
        print("Run: python -m ml_dataset.create_sequences")
        return

    df = pd.read_parquet(parquet_path)
    print(f"\n✅ Loaded {len(df)} stint sequences")

    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)

    print(f"Total sequences: {len(df):,}")
    print(f"Total laps: {df['sequence_length'].sum():,}")
    print(f"Total valid laps: {df['num_valid_laps'].sum():,}")
    print(f"Valid lap percentage: {100 * df['num_valid_laps'].sum() / df['sequence_length'].sum():.2f}%")

    # Sequence length distribution
    print("\n" + "-" * 80)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("-" * 80)

    print(df['sequence_length'].describe())

    # Valid laps distribution
    print("\n" + "-" * 80)
    print("VALID LAPS PER SEQUENCE DISTRIBUTION")
    print("-" * 80)

    print(df['num_valid_laps'].describe())

    # By season
    print("\n" + "-" * 80)
    print("SEQUENCES BY SEASON")
    print("-" * 80)

    season_stats = df.groupby('season_id').agg({
        'stint_index': 'count',
        'sequence_length': 'sum',
        'num_valid_laps': 'sum',
    }).rename(columns={'stint_index': 'stints'})

    print(season_stats)

    # By driver (top 10)
    print("\n" + "-" * 80)
    print("TOP 10 DRIVERS BY NUMBER OF STINTS")
    print("-" * 80)

    driver_counts = df['driver_id'].value_counts().head(10)
    print(driver_counts)

    # By track (top 10)
    print("\n" + "-" * 80)
    print("TOP 10 TRACKS BY NUMBER OF STINTS")
    print("-" * 80)

    track_counts = df['track_id'].value_counts().head(10)
    print(track_counts)

    # Sample sequences
    print("\n" + "-" * 80)
    print("SAMPLE SEQUENCES (First 5)")
    print("-" * 80)

    sample_cols = [
        'race_id', 'season_id', 'driver_id', 'track_id',
        'stint_index', 'sequence_length', 'num_valid_laps', 'mean_lap_time'
    ]

    print(df[sample_cols].head().to_string(index=False))

    # Example detailed sequence
    print("\n" + "-" * 80)
    print("EXAMPLE: DETAILED SEQUENCE VIEW")
    print("-" * 80)

    example = df.iloc[0]
    print(f"\nRace ID: {example['race_id']}")
    print(f"Season: {example['season_id']}")
    print(f"Driver: {example['driver_id']}")
    print(f"Track: {example['track_id']}")
    print(f"Stint: {example['stint_index']}")
    print(f"Length: {example['sequence_length']} laps")
    print(f"Valid: {example['num_valid_laps']} laps")

    print(f"\nLap-by-lap data (first 10 laps):")
    print(f"{'Lap':<5} {'Compound':<12} {'TyreAge':<8} {'Fuel':<6} {'LapTime':<8} {'Valid':<6}")
    print("-" * 60)

    for i in range(min(10, example['sequence_length'])):
        lap = int(example['lap_in_stint_seq'][i])
        compound = example['compound_seq'][i]
        tyre_age = int(example['tyre_age_seq'][i])
        fuel = example['fuel_proxy_seq'][i]
        lap_time = example['lap_time_seconds_seq'][i]
        valid = example['valid_mask_seq'][i]

        print(
            f"{lap:<5} {compound:<12} {tyre_age:<8} {fuel:<6.3f} "
            f"{lap_time:<8.3f} {'✓' if valid else '✗':<6}"
        )

    # Stint length histogram
    print("\n" + "-" * 80)
    print("STINT LENGTH HISTOGRAM")
    print("-" * 80)

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    hist = pd.cut(df['sequence_length'], bins=bins).value_counts().sort_index()
    print(hist)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
