"""
Inspection tool for laps_ml_features table.

Usage:
    python -m feature_engineering.inspect_features
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_engineering.feature_builder import FeatureBuilder


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Inspect the laps_ml_features table."""

    builder = FeatureBuilder(cfg)

    print("\n" + "=" * 80)
    print("LAPS_ML_FEATURES TABLE INSPECTION")
    print("=" * 80)

    # Check if table exists
    table_check = builder.conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'laps_ml_features'
    """).fetchone()

    if table_check[0] == 0:
        print("\n❌ laps_ml_features table does not exist!")
        print("Run: python -m feature_engineering.build_features")
        builder.close()
        return

    print("\n✅ laps_ml_features table exists")

    # Get statistics
    stats = builder.get_feature_stats()

    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total laps: {stats['total_laps']:,}")
    print(f"Valid laps: {stats['valid_laps']:,} ({stats['valid_pct']:.2f}%)")
    print(f"Total stints: {stats['total_stints']:,}")
    print(f"Avg stint length: {stats['avg_stint_length']:.1f} laps")

    # Validation breakdown
    print("\n" + "-" * 80)
    print("VALIDATION FLAGS BREAKDOWN")
    print("-" * 80)

    validation = builder.conn.execute("""
        SELECT
            COUNT(*) as total_laps,
            COUNT(*) FILTER (WHERE valid_lap_flag = TRUE) as valid,
            COUNT(*) FILTER (WHERE outlap_flag = TRUE) as outlaps,
            COUNT(*) FILTER (WHERE inlap_flag = TRUE) as inlaps,
            COUNT(*) FILTER (WHERE lap_time_seconds IS NULL) as no_time,
            COUNT(*) FILTER (WHERE clean_air_flag = TRUE) as clean_air
        FROM laps_ml_features
    """).fetchdf()

    print(validation.to_string(index=False))

    # Invalid lap reasons
    print("\n" + "-" * 80)
    print("INVALID LAP REASONS (Sample)")
    print("-" * 80)

    invalid_reasons = builder.conn.execute("""
        SELECT
            track_status,
            COUNT(*) as count,
            CASE
                WHEN track_status = '1' THEN 'Green flag'
                WHEN track_status = '2' THEN 'Yellow flag'
                WHEN track_status = '4' THEN 'Safety Car'
                WHEN track_status = '6' THEN 'VSC'
                WHEN track_status = '12' THEN 'Formation/Warmup'
                ELSE 'Other'
            END as description
        FROM laps_ml_features
        WHERE valid_lap_flag = FALSE
        GROUP BY track_status
        ORDER BY count DESC
        LIMIT 10
    """).fetchdf()

    print(invalid_reasons.to_string(index=False))

    # Feature distributions
    print("\n" + "-" * 80)
    print("COMPOUND DISTRIBUTION")
    print("-" * 80)
    print(stats['by_compound'].to_string(index=False))

    print("\n" + "-" * 80)
    print("SEASON DISTRIBUTION")
    print("-" * 80)
    print(stats['by_season'].to_string(index=False))

    # Tyre age distribution
    print("\n" + "-" * 80)
    print("TYRE AGE DISTRIBUTION (Valid Laps)")
    print("-" * 80)

    tyre_age_dist = builder.conn.execute("""
        SELECT
            FLOOR(tyre_age / 5) * 5 as age_bucket,
            COUNT(*) as laps,
            AVG(lap_time_seconds) as avg_lap_time
        FROM laps_ml_features
        WHERE valid_lap_flag = TRUE AND tyre_age IS NOT NULL
        GROUP BY age_bucket
        ORDER BY age_bucket
        LIMIT 20
    """).fetchdf()

    print(tyre_age_dist.to_string(index=False))

    # Fuel proxy distribution
    print("\n" + "-" * 80)
    print("FUEL PROXY DISTRIBUTION (Valid Laps)")
    print("-" * 80)

    fuel_dist = builder.conn.execute("""
        SELECT
            ROUND(fuel_proxy * 10) / 10 as fuel_bucket,
            COUNT(*) as laps
        FROM laps_ml_features
        WHERE valid_lap_flag = TRUE
        GROUP BY fuel_bucket
        ORDER BY fuel_bucket
    """).fetchdf()

    print(fuel_dist.to_string(index=False))

    # Sample stint
    print("\n" + "-" * 80)
    print("SAMPLE STINT (Verstappen, 2024 Bahrain, Stint 1)")
    print("-" * 80)

    sample_stint = builder.conn.execute("""
        SELECT
            lap_in_stint,
            ROUND(lap_time_seconds, 3) as lap_time,
            compound,
            tyre_age,
            ROUND(fuel_proxy, 3) as fuel,
            clean_air_flag as clean,
            valid_lap_flag as valid,
            track_status
        FROM laps_ml_features
        WHERE season_id = 2024
            AND track_id = 'Sakhir'
            AND driver_id = 'VER'
            AND stint_index = 1.0
        ORDER BY lap_in_stint
    """).fetchdf()

    print(sample_stint.to_string(index=False))

    builder.close()


if __name__ == "__main__":
    main()
