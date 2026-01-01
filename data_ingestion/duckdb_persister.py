"""
DuckDB persistence layer for F1 lap data.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DuckDBPersister:
    """Handles persistence of F1 data to DuckDB."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.db_path = Path(config.data.duckdb.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")

    def _initialize_schema(self):
        """Create initial database schema."""
        # Race metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS race_metadata (
                race_id INTEGER PRIMARY KEY,
                season INTEGER NOT NULL,
                round_number INTEGER NOT NULL,
                event_name VARCHAR NOT NULL,
                country VARCHAR,
                location VARCHAR,
                circuit_key VARCHAR NOT NULL,
                event_date DATE,
                session_type VARCHAR NOT NULL,
                UNIQUE(season, round_number, session_type)
            )
        """)

        # Raw laps table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS laps_raw (
                lap_id INTEGER PRIMARY KEY,
                race_id INTEGER NOT NULL,
                driver_number VARCHAR NOT NULL,
                driver_code VARCHAR,
                team VARCHAR NOT NULL,
                lap_number DOUBLE NOT NULL,
                lap_time_seconds DOUBLE,
                lap_time_ms DOUBLE,
                stint DOUBLE,
                compound VARCHAR,
                tyre_life DOUBLE,
                fresh_tyre BOOLEAN,
                track_status VARCHAR,
                is_accurate BOOLEAN,
                pit_in_time DOUBLE,
                pit_out_time DOUBLE,
                sector_1_time DOUBLE,
                sector_2_time DOUBLE,
                sector_3_time DOUBLE,
                speed_i1 DOUBLE,
                speed_i2 DOUBLE,
                speed_fl DOUBLE,
                speed_st DOUBLE,
                position DOUBLE,
                FOREIGN KEY (race_id) REFERENCES race_metadata(race_id)
            )
        """)

        # Create indices for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_laps_race_driver
            ON laps_raw(race_id, driver_code)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_laps_stint
            ON laps_raw(race_id, driver_code, stint)
        """)

        logger.info("Database schema initialized")

    def get_or_create_race_id(self, metadata: dict) -> int:
        """
        Get existing race_id or create new race metadata entry.

        Args:
            metadata: Race metadata dictionary

        Returns:
            race_id (integer primary key)
        """
        # Check if race already exists
        result = self.conn.execute("""
            SELECT race_id FROM race_metadata
            WHERE season = ? AND round_number = ? AND session_type = ?
        """, [metadata["season"], metadata["round_number"], metadata["session_type"]]).fetchone()

        if result:
            race_id = result[0]
            logger.debug(f"Found existing race_id={race_id}")
            return race_id

        # Create new race entry
        result = self.conn.execute("""
            SELECT COALESCE(MAX(race_id), 0) + 1 FROM race_metadata
        """).fetchone()
        race_id = result[0]

        self.conn.execute("""
            INSERT INTO race_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            race_id,
            metadata["season"],
            metadata["round_number"],
            metadata["event_name"],
            metadata.get("country"),
            metadata.get("location"),
            metadata["circuit_key"],
            metadata.get("event_date"),
            metadata["session_type"],
        ])

        logger.info(
            f"Created new race_id={race_id} for {metadata['season']} "
            f"R{metadata['round_number']} {metadata['session_type']}"
        )
        return race_id

    def persist_laps(self, laps_df: pd.DataFrame, race_id: int) -> int:
        """
        Persist laps dataframe to database.

        Args:
            laps_df: DataFrame containing lap data
            race_id: Foreign key to race_metadata

        Returns:
            Number of laps inserted
        """
        # Check if data already exists
        existing_count = self.conn.execute("""
            SELECT COUNT(*) FROM laps_raw WHERE race_id = ?
        """, [race_id]).fetchone()[0]

        if existing_count > 0:
            logger.warning(
                f"race_id={race_id} already has {existing_count} laps. Skipping."
            )
            return 0

        # Prepare dataframe for insertion
        laps_clean = self._prepare_laps_dataframe(laps_df, race_id)

        # Get starting lap_id
        result = self.conn.execute("""
            SELECT COALESCE(MAX(lap_id), 0) FROM laps_raw
        """).fetchone()
        start_lap_id = result[0] + 1

        # Add lap_id column
        laps_clean.insert(0, "lap_id", range(start_lap_id, start_lap_id + len(laps_clean)))

        # Insert into database
        self.conn.execute("""
            INSERT INTO laps_raw SELECT * FROM laps_clean
        """)

        logger.info(f"Inserted {len(laps_clean)} laps for race_id={race_id}")
        return len(laps_clean)

    def _prepare_laps_dataframe(self, laps_df: pd.DataFrame, race_id: int) -> pd.DataFrame:
        """
        Clean and prepare laps dataframe for database insertion.

        Args:
            laps_df: Raw laps dataframe from FastF1
            race_id: Foreign key to race_metadata

        Returns:
            Cleaned dataframe matching laps_raw schema
        """
        df = laps_df.copy()

        # Map column names to schema (for direct FastF1 columns)
        column_mapping = {
            "DriverNumber": "driver_number",
            "Driver": "driver_code",
            "Team": "team",
            "LapNumber": "lap_number",
            "Stint": "stint",
            "Compound": "compound",
            "TyreLife": "tyre_life",
            "FreshTyre": "fresh_tyre",
            "TrackStatus": "track_status",
            "IsAccurate": "is_accurate",
            "SpeedI1": "speed_i1",
            "SpeedI2": "speed_i2",
            "SpeedFL": "speed_fl",
            "SpeedST": "speed_st",
            "Position": "position",
        }

        # Select and rename columns
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df_selected = df[available_cols].rename(columns=column_mapping)

        # Add race_id
        df_selected["race_id"] = race_id

        # Now add computed timing columns (AFTER selecting base columns)
        # Convert LapTime (timedelta) to seconds
        if "LapTime" in df.columns:
            df_selected["lap_time_seconds"] = df["LapTime"].dt.total_seconds()
            df_selected["lap_time_ms"] = df["LapTime"].dt.total_seconds() * 1000
        else:
            df_selected["lap_time_seconds"] = None
            df_selected["lap_time_ms"] = None

        # Convert pit times (timedelta) to seconds
        if "PitInTime" in df.columns:
            df_selected["pit_in_time"] = df["PitInTime"].dt.total_seconds()
        else:
            df_selected["pit_in_time"] = None

        if "PitOutTime" in df.columns:
            df_selected["pit_out_time"] = df["PitOutTime"].dt.total_seconds()
        else:
            df_selected["pit_out_time"] = None

        # Convert sector times
        for sector_num, sector_col in [(1, "Sector1Time"), (2, "Sector2Time"), (3, "Sector3Time")]:
            col_name = f"sector_{sector_num}_time"
            if sector_col in df.columns:
                df_selected[col_name] = df[sector_col].dt.total_seconds()
            else:
                df_selected[col_name] = None

        # Ensure required columns exist
        required_cols = [
            "race_id", "driver_number", "driver_code", "team", "lap_number",
            "lap_time_seconds", "lap_time_ms", "stint", "compound", "tyre_life",
            "fresh_tyre", "track_status", "is_accurate", "pit_in_time", "pit_out_time",
            "sector_1_time", "sector_2_time", "sector_3_time",
            "speed_i1", "speed_i2", "speed_fl", "speed_st", "position"
        ]

        for col in required_cols:
            if col not in df_selected.columns:
                df_selected[col] = None

        # Reorder columns to match schema (excluding lap_id which we'll add later)
        df_final = df_selected[required_cols]

        # Convert all numpy/pandas dtypes to DuckDB-compatible types
        df_final = self._convert_dtypes_for_duckdb(df_final)

        return df_final

    def _convert_dtypes_for_duckdb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pandas/numpy dtypes to DuckDB-compatible types.

        Args:
            df: DataFrame with potentially incompatible types

        Returns:
            DataFrame with converted types
        """
        df_converted = df.copy()

        for col in df_converted.columns:
            # Convert pandas nullable Int64 to standard float (DuckDB handles NULL better with float)
            if df_converted[col].dtype == "Int64":
                df_converted[col] = df_converted[col].astype("float64")

            # Convert other integer types to standard int64 or float64
            elif pd.api.types.is_integer_dtype(df_converted[col]):
                # Use float64 to preserve NaN/NULL values
                df_converted[col] = df_converted[col].astype("float64")

            # Convert boolean to object to handle NULL
            elif pd.api.types.is_bool_dtype(df_converted[col]):
                df_converted[col] = df_converted[col].astype("object")

        # Replace all NaN/NaT with None for proper SQL NULL handling
        df_converted = df_converted.where(pd.notna(df_converted), None)

        return df_converted

    def get_race_count(self) -> int:
        """Get total number of races in database."""
        result = self.conn.execute("SELECT COUNT(*) FROM race_metadata").fetchone()
        return result[0]

    def get_lap_count(self) -> int:
        """Get total number of laps in database."""
        result = self.conn.execute("SELECT COUNT(*) FROM laps_raw").fetchone()
        return result[0]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")
