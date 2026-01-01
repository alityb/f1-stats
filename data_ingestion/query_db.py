"""
Interactive SQL query tool for the F1 database.

Usage:
    python -m data_ingestion.query_db "SELECT * FROM race_metadata LIMIT 5"
    python -m data_ingestion.query_db  # Opens interactive mode
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
    """Run SQL queries against the database."""
    persister = DuckDBPersister(cfg)

    # Check if query provided as argument
    query_arg = None
    for arg in sys.argv:
        if not arg.startswith('-') and not arg.endswith('.py') and 'hydra' not in arg:
            if '=' not in arg:  # Not a hydra override
                query_arg = arg
                break

    if query_arg:
        # Single query mode
        print(f"\nExecuting query:\n{query_arg}\n")
        try:
            result = persister.conn.execute(query_arg).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Interactive mode
        print("\n" + "=" * 80)
        print("F1 DATABASE QUERY TOOL - Interactive Mode")
        print("=" * 80)
        print("\nAvailable tables:")
        print("  - race_metadata (race_id, season, round_number, event_name, ...)")
        print("  - laps_raw (lap_id, race_id, driver_code, lap_time_seconds, ...)")
        print("\nExample queries:")
        print("  1. Show all drivers from 2024 Monaco:")
        print("     SELECT DISTINCT l.driver_code, l.team")
        print("     FROM laps_raw l")
        print("     JOIN race_metadata r ON l.race_id = r.race_id")
        print("     WHERE r.season = 2024 AND r.event_name LIKE '%Monaco%'")
        print("\n  2. Get fastest lap by driver in a specific race:")
        print("     SELECT driver_code, MIN(lap_time_seconds) as fastest_lap")
        print("     FROM laps_raw")
        print("     WHERE race_id = 1 AND lap_time_seconds IS NOT NULL")
        print("     GROUP BY driver_code")
        print("     ORDER BY fastest_lap")
        print("\n  3. Count laps per compound:")
        print("     SELECT compound, COUNT(*) as laps")
        print("     FROM laps_raw")
        print("     WHERE compound IS NOT NULL")
        print("     GROUP BY compound")
        print("\nType 'schema' to see table schemas")
        print("Type 'examples' for more query examples")
        print("Type 'quit' to exit")
        print("-" * 80)

        while True:
            try:
                query = input("\nSQL> ").strip()

                if not query:
                    continue

                if query.lower() == 'quit':
                    break

                if query.lower() == 'schema':
                    show_schema(persister)
                    continue

                if query.lower() == 'examples':
                    show_examples()
                    continue

                # Execute query
                result = persister.conn.execute(query).fetchdf()
                print("\n" + result.to_string(index=False))

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    persister.close()


def show_schema(persister):
    """Display table schemas."""
    print("\n" + "=" * 80)
    print("DATABASE SCHEMA")
    print("=" * 80)

    # race_metadata schema
    print("\n📋 race_metadata:")
    schema = persister.conn.execute("""
        DESCRIBE race_metadata
    """).fetchdf()
    print(schema.to_string(index=False))

    # laps_raw schema
    print("\n📋 laps_raw:")
    schema = persister.conn.execute("""
        DESCRIBE laps_raw
    """).fetchdf()
    print(schema.to_string(index=False))


def show_examples():
    """Display example queries."""
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)

    examples = [
        ("All races from 2024", """
SELECT season, round_number, event_name, event_date
FROM race_metadata
WHERE season = 2024
ORDER BY round_number
        """),

        ("Verstappen's stint breakdown in 2024 Bahrain", """
SELECT
    l.stint,
    l.compound,
    COUNT(*) as laps,
    MIN(l.tyre_life) as tyre_start,
    MAX(l.tyre_life) as tyre_end
FROM laps_raw l
JOIN race_metadata r ON l.race_id = r.race_id
WHERE r.season = 2024
    AND r.event_name LIKE '%Bahrain%'
    AND l.driver_code = 'VER'
GROUP BY l.stint, l.compound
ORDER BY l.stint
        """),

        ("Average lap time by compound (2024 season)", """
SELECT
    compound,
    COUNT(*) as laps,
    ROUND(AVG(lap_time_seconds), 3) as avg_lap_time,
    ROUND(MIN(lap_time_seconds), 3) as fastest_lap
FROM laps_raw l
JOIN race_metadata r ON l.race_id = r.race_id
WHERE r.season = 2024
    AND lap_time_seconds IS NOT NULL
    AND track_status = '1'
GROUP BY compound
ORDER BY avg_lap_time
        """),

        ("Races with most Safety Car laps", """
SELECT
    r.season,
    r.event_name,
    COUNT(*) as sc_laps
FROM laps_raw l
JOIN race_metadata r ON l.race_id = r.race_id
WHERE l.track_status IN ('4', '6')
GROUP BY r.race_id, r.season, r.event_name
ORDER BY sc_laps DESC
LIMIT 10
        """),

        ("Driver with most laps on SOFT compound", """
SELECT
    driver_code,
    COUNT(*) as soft_laps
FROM laps_raw
WHERE compound = 'SOFT'
GROUP BY driver_code
ORDER BY soft_laps DESC
LIMIT 10
        """),
    ]

    for i, (title, query) in enumerate(examples, 1):
        print(f"\n{i}. {title}:")
        print(query.strip())


if __name__ == "__main__":
    main()
