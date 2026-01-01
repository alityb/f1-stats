"""
Data quality inspection script for F1 lap data.

Analyzes NaN patterns, track status, and data completeness.

Usage:
    python -m data_ingestion.data_quality_check
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Comprehensive data quality analysis."""
    persister = DuckDBPersister(cfg)

    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    # 1. Overall lap time completeness
    print("\n" + "-" * 80)
    print("1. LAP TIME COMPLETENESS")
    print("-" * 80)

    completeness = persister.conn.execute("""
        SELECT
            COUNT(*) as total_laps,
            COUNT(lap_time_seconds) as laps_with_time,
            COUNT(*) - COUNT(lap_time_seconds) as laps_without_time,
            ROUND(100.0 * COUNT(lap_time_seconds) / COUNT(*), 2) as pct_complete
        FROM laps_raw
    """).fetchdf()

    print(completeness.to_string(index=False))

    # 2. NaN breakdown by track status
    print("\n" + "-" * 80)
    print("2. NaN LAPS BY TRACK STATUS")
    print("-" * 80)

    status_breakdown = persister.conn.execute("""
        SELECT
            track_status,
            COUNT(*) as lap_count,
            COUNT(lap_time_seconds) as with_time,
            COUNT(*) - COUNT(lap_time_seconds) as without_time,
            ROUND(100.0 * (COUNT(*) - COUNT(lap_time_seconds)) / COUNT(*), 2) as pct_nan
        FROM laps_raw
        GROUP BY track_status
        ORDER BY lap_count DESC
    """).fetchdf()

    print(status_breakdown.to_string(index=False))

    # 3. Track status legend (common values)
    print("\n" + "-" * 80)
    print("3. TRACK STATUS CODES (Common Values)")
    print("-" * 80)
    print("""
    1 = Green flag (normal racing)
    2 = Yellow flag (caution)
    4 = Safety Car (SC)
    5 = Red flag (stopped)
    6 = Virtual Safety Car (VSC)
    7 = VSC Ending
    12 = Unknown/Formation/Warmup laps
    """)

    # 4. Pit lap analysis (inlap/outlap often have NaN or invalid times)
    print("\n" + "-" * 80)
    print("4. PIT LAPS ANALYSIS")
    print("-" * 80)

    pit_analysis = persister.conn.execute("""
        SELECT
            CASE
                WHEN pit_out_time IS NOT NULL THEN 'Out Lap'
                WHEN pit_in_time IS NOT NULL THEN 'In Lap'
                ELSE 'Normal Lap'
            END as lap_type,
            COUNT(*) as lap_count,
            COUNT(lap_time_seconds) as with_time,
            COUNT(*) - COUNT(lap_time_seconds) as without_time,
            ROUND(100.0 * (COUNT(*) - COUNT(lap_time_seconds)) / COUNT(*), 2) as pct_nan
        FROM laps_raw
        GROUP BY lap_type
        ORDER BY lap_count DESC
    """).fetchdf()

    print(pit_analysis.to_string(index=False))

    # 5. NaN distribution by race
    print("\n" + "-" * 80)
    print("5. TOP 10 RACES WITH MOST NaN LAP TIMES")
    print("-" * 80)

    race_nan_distribution = persister.conn.execute("""
        SELECT
            r.season,
            r.event_name,
            COUNT(l.lap_id) as total_laps,
            COUNT(*) - COUNT(l.lap_time_seconds) as nan_laps,
            ROUND(100.0 * (COUNT(*) - COUNT(l.lap_time_seconds)) / COUNT(*), 2) as pct_nan
        FROM laps_raw l
        JOIN race_metadata r ON l.race_id = r.race_id
        GROUP BY r.race_id, r.season, r.event_name
        ORDER BY nan_laps DESC
        LIMIT 10
    """).fetchdf()

    print(race_nan_distribution.to_string(index=False))

    # 6. Sample valid laps (with lap times)
    print("\n" + "-" * 80)
    print("6. SAMPLE VALID LAPS (With Lap Times)")
    print("-" * 80)

    valid_sample = persister.conn.execute("""
        SELECT
            r.event_name,
            l.driver_code,
            l.lap_number,
            ROUND(l.lap_time_seconds, 3) as lap_time_sec,
            l.stint,
            l.compound,
            l.tyre_life,
            l.track_status
        FROM laps_raw l
        JOIN race_metadata r ON l.race_id = r.race_id
        WHERE l.lap_time_seconds IS NOT NULL
            AND l.track_status = '1'  -- Green flag
        LIMIT 10
    """).fetchdf()

    print(valid_sample.to_string(index=False))

    # 7. Sample NaN laps (to understand why they're NaN)
    print("\n" + "-" * 80)
    print("7. SAMPLE NaN LAPS (Understanding Missing Times)")
    print("-" * 80)

    nan_sample = persister.conn.execute("""
        SELECT
            r.event_name,
            l.driver_code,
            l.lap_number,
            l.stint,
            l.compound,
            l.track_status,
            CASE
                WHEN l.pit_out_time IS NOT NULL THEN 'OUT'
                WHEN l.pit_in_time IS NOT NULL THEN 'IN'
                ELSE NULL
            END as pit_lap
        FROM laps_raw l
        JOIN race_metadata r ON l.race_id = r.race_id
        WHERE l.lap_time_seconds IS NULL
        LIMIT 20
    """).fetchdf()

    print(nan_sample.to_string(index=False))

    # 8. Stint-level analysis
    print("\n" + "-" * 80)
    print("8. STINT DATA QUALITY")
    print("-" * 80)

    stint_quality = persister.conn.execute("""
        SELECT
            COUNT(DISTINCT CONCAT(race_id, '-', driver_code, '-', stint)) as total_stints,
            AVG(laps_per_stint) as avg_laps_per_stint,
            AVG(valid_laps_per_stint) as avg_valid_laps_per_stint
        FROM (
            SELECT
                race_id,
                driver_code,
                stint,
                COUNT(*) as laps_per_stint,
                COUNT(lap_time_seconds) as valid_laps_per_stint
            FROM laps_raw
            WHERE stint IS NOT NULL
            GROUP BY race_id, driver_code, stint
        ) stint_stats
    """).fetchdf()

    print(stint_quality.to_string(index=False))

    # 9. Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    total_laps = completeness['total_laps'].iloc[0]
    pct_complete = completeness['pct_complete'].iloc[0]

    if pct_complete > 90:
        print(f"✅ EXCELLENT: {pct_complete}% of laps have valid times")
        print("   → NaN laps are mostly formation laps, pit laps, and SC periods")
        print("   → This is expected and normal for F1 data")
    elif pct_complete > 80:
        print(f"⚠️  GOOD: {pct_complete}% of laps have valid times")
        print("   → Some data quality issues may exist")
        print("   → Review races with high NaN percentages")
    else:
        print(f"❌ CONCERNING: Only {pct_complete}% of laps have valid times")
        print("   → Investigate data loading issues")

    print("\n📊 Next Steps:")
    print("   1. Valid laps will be used for model training")
    print("   2. Phase 2 will implement validation masks to filter:")
    print("      - Safety car periods (track_status 4, 6)")
    print("      - Pit in/out laps")
    print("      - Formation laps")
    print("      - NaN lap times")
    print("   3. All laps are kept in DB for context/analysis")

    persister.close()


if __name__ == "__main__":
    main()
