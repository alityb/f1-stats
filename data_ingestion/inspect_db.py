"""
Utility script to inspect the DuckDB database.

Usage:
    python -m data_ingestion.inspect_db
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.duckdb_persister import DuckDBPersister


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Inspect database contents."""
    persister = DuckDBPersister(cfg)

    print("\n" + "=" * 80)
    print("DATABASE SUMMARY")
    print("=" * 80)

    # Count races
    race_count = persister.get_race_count()
    lap_count = persister.get_lap_count()

    print(f"\nTotal races: {race_count}")
    print(f"Total laps: {lap_count:,}")

    if race_count > 0:
        print("\n" + "-" * 80)
        print("RACES IN DATABASE")
        print("-" * 80)

        races = persister.conn.execute("""
            SELECT
                race_id,
                season,
                round_number,
                event_name,
                circuit_key,
                session_type,
                event_date
            FROM race_metadata
            ORDER BY season, round_number
        """).fetchdf()

        print(races.to_string(index=False))

        print("\n" + "-" * 80)
        print("LAPS PER RACE")
        print("-" * 80)

        laps_per_race = persister.conn.execute("""
            SELECT
                r.season,
                r.round_number,
                r.event_name,
                COUNT(l.lap_id) as lap_count,
                COUNT(DISTINCT l.driver_code) as driver_count
            FROM race_metadata r
            LEFT JOIN laps_raw l ON r.race_id = l.race_id
            GROUP BY r.race_id, r.season, r.round_number, r.event_name
            ORDER BY r.season, r.round_number
        """).fetchdf()

        print(laps_per_race.to_string(index=False))

        print("\n" + "-" * 80)
        print("SAMPLE LAPS (First 10)")
        print("-" * 80)

        sample_laps = persister.conn.execute("""
            SELECT
                l.lap_id,
                r.event_name,
                l.driver_code,
                l.lap_number,
                l.lap_time_seconds,
                l.stint,
                l.compound,
                l.tyre_life,
                l.track_status
            FROM laps_raw l
            JOIN race_metadata r ON l.race_id = r.race_id
            LIMIT 10
        """).fetchdf()

        print(sample_laps.to_string(index=False))

    persister.close()


if __name__ == "__main__":
    main()
