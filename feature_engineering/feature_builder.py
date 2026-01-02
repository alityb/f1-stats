"""
Feature engineering pipeline for ML-ready lap features.

Transforms raw lap data into features for the pace decomposition model.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds ML features from raw lap data."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.db_path = Path(config.data.duckdb.db_path)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()

    def _connect(self):
        """Connect to DuckDB database."""
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")

    def create_features_table(self):
        """Create laps_ml_features table with all derived features."""
        logger.info("Creating laps_ml_features table...")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS laps_ml_features AS
            WITH lap_context AS (
                SELECT
                    -- Identifiers
                    l.race_id,
                    r.season as season_id,
                    r.circuit_key as track_id,
                    l.driver_code as driver_id,
                    CONCAT(l.team, '_', r.season) as car_id,
                    l.stint as stint_index,
                    l.lap_number as lap_in_race,

                    -- Lap within stint (starts at 1 for each stint)
                    ROW_NUMBER() OVER (
                        PARTITION BY l.race_id, l.driver_code, l.stint
                        ORDER BY l.lap_number
                    ) as lap_in_stint,

                    -- Target variable
                    l.lap_time_ms,
                    l.lap_time_seconds,

                    -- Tyre features
                    l.compound,
                    l.tyre_life as tyre_age,
                    l.fresh_tyre,

                    -- Stint length (total laps in this stint)
                    COUNT(*) OVER (
                        PARTITION BY l.race_id, l.driver_code, l.stint
                    ) as stint_length,

                    -- Context features
                    l.track_status,
                    l.is_accurate,

                    -- Pit lap flags
                    CASE WHEN l.pit_out_time IS NOT NULL THEN TRUE ELSE FALSE END as outlap_flag,
                    CASE WHEN l.pit_in_time IS NOT NULL THEN TRUE ELSE FALSE END as inlap_flag,

                    -- Race progress (fuel proxy)
                    -- Get total laps in race for fuel estimation
                    MAX(l.lap_number) OVER (PARTITION BY l.race_id) as total_race_laps,

                    -- Sector times
                    l.sector_1_time,
                    l.sector_2_time,
                    l.sector_3_time,

                    -- Speed traps
                    l.speed_i1,
                    l.speed_i2,
                    l.speed_fl,
                    l.speed_st,

                    -- Position
                    l.position

                FROM laps_raw l
                JOIN race_metadata r ON l.race_id = r.race_id
            ),
            features AS (
                SELECT
                    *,
                    -- Fuel proxy: progress through race (0.0 to 1.0)
                    CAST(lap_in_race AS DOUBLE) / NULLIF(total_race_laps, 0) as fuel_proxy,

                    -- Valid lap flag (for training mask)
                    CASE
                        WHEN lap_time_seconds IS NULL THEN FALSE
                        WHEN is_accurate = FALSE THEN FALSE
                        WHEN outlap_flag = TRUE THEN FALSE
                        WHEN inlap_flag = TRUE THEN FALSE
                        -- Safety car and VSC periods
                        WHEN track_status IN ('4', '6', '7') THEN FALSE
                        -- Formation/warmup laps
                        WHEN track_status = '12' THEN FALSE
                        -- Red flag
                        WHEN track_status = '5' THEN FALSE
                        -- Complex track status codes (combinations of flags)
                        WHEN LENGTH(track_status) > 2 THEN FALSE
                        ELSE TRUE
                    END as valid_lap_flag

                FROM lap_context
            )
            SELECT
                -- Identifiers
                race_id,
                season_id,
                track_id,
                driver_id,
                car_id,
                stint_index,
                lap_in_race,
                lap_in_stint,

                -- Target
                lap_time_ms,
                lap_time_seconds,

                -- Validation mask
                valid_lap_flag,

                -- Tyre features
                compound,
                tyre_age,
                fresh_tyre,
                stint_length,

                -- Context features
                fuel_proxy,
                track_status,
                outlap_flag,
                inlap_flag,

                -- Traffic proxy placeholder (will enhance later)
                FALSE as clean_air_flag,

                -- Sector times
                sector_1_time,
                sector_2_time,
                sector_3_time,

                -- Speed data
                speed_i1,
                speed_i2,
                speed_fl,
                speed_st,

                -- Position
                position

            FROM features
        """)

        logger.info("✅ laps_ml_features table created")

        # Add indices
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_race_driver
            ON laps_ml_features(race_id, driver_id)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_stint
            ON laps_ml_features(race_id, driver_id, stint_index)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_valid
            ON laps_ml_features(valid_lap_flag)
        """)

        logger.info("✅ Indices created")

    def enhance_traffic_detection(self):
        """
        Enhance clean_air_flag using gap to car ahead.

        Note: This is a simplified version. More sophisticated traffic
        detection would require telemetry data and gap analysis.
        """
        logger.info("Enhancing traffic detection...")

        # For now, use a simple heuristic:
        # - First lap in stint after pit stop = not clean air (outlap traffic)
        # - Laps immediately after safety car = not clean air
        # - Otherwise assume clean air (will improve in later versions)

        self.conn.execute("""
            UPDATE laps_ml_features
            SET clean_air_flag = CASE
                -- Outlaps typically have traffic
                WHEN outlap_flag = TRUE THEN FALSE
                -- First few laps after SC restart
                WHEN track_status IN ('4', '6', '7') THEN FALSE
                -- First lap in race (grid start = traffic)
                WHEN lap_in_race = 1 THEN FALSE
                -- Default: assume clean air for valid laps
                WHEN valid_lap_flag = TRUE THEN TRUE
                ELSE FALSE
            END
        """)

        logger.info("✅ Traffic detection updated")

    def get_feature_stats(self) -> dict:
        """Get statistics about the features table."""
        stats = {}

        # Total features
        result = self.conn.execute("""
            SELECT COUNT(*) FROM laps_ml_features
        """).fetchone()
        stats['total_laps'] = result[0]

        # Valid laps
        result = self.conn.execute("""
            SELECT COUNT(*) FROM laps_ml_features WHERE valid_lap_flag = TRUE
        """).fetchone()
        stats['valid_laps'] = result[0]
        stats['valid_pct'] = 100.0 * stats['valid_laps'] / stats['total_laps']

        # By compound
        compound_stats = self.conn.execute("""
            SELECT
                compound,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE valid_lap_flag = TRUE) as valid
            FROM laps_ml_features
            WHERE compound IS NOT NULL
            GROUP BY compound
            ORDER BY total DESC
        """).fetchdf()
        stats['by_compound'] = compound_stats

        # By season
        season_stats = self.conn.execute("""
            SELECT
                season_id,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE valid_lap_flag = TRUE) as valid
            FROM laps_ml_features
            GROUP BY season_id
            ORDER BY season_id
        """).fetchdf()
        stats['by_season'] = season_stats

        # Stint stats
        result = self.conn.execute("""
            SELECT
                COUNT(DISTINCT CONCAT(race_id, '-', driver_id, '-', stint_index)) as total_stints,
                AVG(stint_length) as avg_stint_length
            FROM laps_ml_features
        """).fetchone()
        stats['total_stints'] = result[0]
        stats['avg_stint_length'] = result[1]

        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
