"""
Sequence dataset builder for stint-level lap sequences.

Extracts features, targets, and masks grouped by stint for model training.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Builds stint-level sequence datasets for training."""

    # Feature schema definition
    CATEGORICAL_FEATURES = [
        "driver_id",
        "car_id",
        "track_id",
        "season_id",
        "compound",
    ]

    CONTINUOUS_FEATURES = [
        "lap_in_stint",
        "tyre_age",
        "fuel_proxy",
        "stint_length",
    ]

    # Additional context (not direct model inputs but useful metadata)
    METADATA_FEATURES = [
        "race_id",
        "stint_index",
        "lap_in_race",
        "track_status",
        "outlap_flag",
        "inlap_flag",
        "clean_air_flag",
    ]

    def __init__(self, config: DictConfig):
        self.config = config
        self.db_path = Path(config.data.duckdb.db_path)
        self.output_dir = Path(config.paths.stints_dir)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()

    def _connect(self):
        """Connect to DuckDB database."""
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")

    def get_stints_list(self) -> pd.DataFrame:
        """
        Get list of all stints to process.

        Returns:
            DataFrame with stint identifiers
        """
        stints = self.conn.execute("""
            SELECT DISTINCT
                race_id,
                season_id,
                track_id,
                driver_id,
                car_id,
                stint_index,
                COUNT(*) as stint_length
            FROM laps_ml_features
            WHERE stint_index IS NOT NULL
            GROUP BY race_id, season_id, track_id, driver_id, car_id, stint_index
            ORDER BY race_id, driver_id, stint_index
        """).fetchdf()

        logger.info(f"Found {len(stints)} stints to process")
        return stints

    def extract_stint_sequence(
        self,
        race_id: int,
        driver_id: str,
        stint_index: float
    ) -> Optional[dict]:
        """
        Extract feature sequence for a single stint.

        Args:
            race_id: Race identifier
            driver_id: Driver code
            stint_index: Stint number

        Returns:
            Dictionary containing sequences and metadata
        """
        # Query all laps in this stint, ordered by lap_in_stint
        query = f"""
            SELECT
                -- Identifiers (categorical)
                season_id,
                driver_id,
                car_id,
                track_id,
                compound,

                -- Continuous features
                lap_in_stint,
                tyre_age,
                fuel_proxy,
                stint_length,

                -- Target
                lap_time_seconds,
                lap_time_ms,

                -- Mask
                valid_lap_flag,

                -- Metadata
                race_id,
                stint_index,
                lap_in_race,
                track_status,
                outlap_flag,
                inlap_flag,
                clean_air_flag

            FROM laps_ml_features
            WHERE race_id = {race_id}
                AND driver_id = '{driver_id}'
                AND stint_index = {stint_index}
            ORDER BY lap_in_stint
        """

        df = self.conn.execute(query).fetchdf()

        if len(df) == 0:
            return None

        # Build sequence dictionary
        sequence = {
            # Stint identifiers (scalars)
            "race_id": int(race_id),
            "season_id": int(df["season_id"].iloc[0]),
            "driver_id": str(driver_id),
            "car_id": str(df["car_id"].iloc[0]),
            "track_id": str(df["track_id"].iloc[0]),
            "stint_index": int(stint_index),

            # Sequence length
            "sequence_length": len(df),

            # Categorical features (sequences)
            "compound_seq": df["compound"].fillna("UNKNOWN").tolist(),

            # Continuous features (sequences)
            "lap_in_stint_seq": df["lap_in_stint"].fillna(0).tolist(),
            "tyre_age_seq": df["tyre_age"].fillna(0).tolist(),
            "fuel_proxy_seq": df["fuel_proxy"].fillna(0.0).tolist(),
            "stint_length_seq": df["stint_length"].fillna(0).tolist(),

            # Targets (sequences)
            "lap_time_seconds_seq": df["lap_time_seconds"].fillna(-1.0).tolist(),
            "lap_time_ms_seq": df["lap_time_ms"].fillna(-1.0).tolist(),

            # Mask (sequence)
            "valid_mask_seq": df["valid_lap_flag"].fillna(False).tolist(),

            # Metadata (sequences) - useful for analysis
            "lap_in_race_seq": df["lap_in_race"].fillna(0).tolist(),
            "track_status_seq": df["track_status"].fillna("0").tolist(),
            "outlap_flag_seq": df["outlap_flag"].fillna(False).tolist(),
            "inlap_flag_seq": df["inlap_flag"].fillna(False).tolist(),
            "clean_air_flag_seq": df["clean_air_flag"].fillna(False).tolist(),

            # Stint-level aggregates (for filtering/analysis)
            "num_valid_laps": int(df["valid_lap_flag"].sum()),
            "mean_lap_time": float(df[df["valid_lap_flag"]]["lap_time_seconds"].mean())
                if df["valid_lap_flag"].sum() > 0 else -1.0,
        }

        return sequence

    def build_sequences(self, max_stints: Optional[int] = None) -> int:
        """
        Build and save all stint sequences to Parquet files.

        Args:
            max_stints: Optional limit on number of stints to process (for testing)

        Returns:
            Number of sequences created
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of stints
        stints = self.get_stints_list()

        if max_stints:
            stints = stints.head(max_stints)
            logger.info(f"Limited to first {max_stints} stints for testing")

        # Process each stint
        sequences = []
        skipped = 0

        for idx, row in tqdm(stints.iterrows(), total=len(stints), desc="Building sequences"):
            sequence = self.extract_stint_sequence(
                race_id=row["race_id"],
                driver_id=row["driver_id"],
                stint_index=row["stint_index"]
            )

            if sequence is None:
                skipped += 1
                continue

            # Filter out stints with no valid laps (can't train on these)
            if sequence["num_valid_laps"] == 0:
                skipped += 1
                continue

            sequences.append(sequence)

        logger.info(f"Created {len(sequences)} sequences ({skipped} skipped)")

        if len(sequences) == 0:
            logger.warning("No sequences to save!")
            return 0

        # Convert to DataFrame for Parquet export
        df_sequences = pd.DataFrame(sequences)

        # Save to Parquet
        output_path = self.output_dir / "stint_sequences.parquet"
        df_sequences.to_parquet(output_path, index=False)

        logger.info(f"✅ Saved {len(sequences)} sequences to {output_path}")

        # Print summary statistics
        self._print_summary(df_sequences)

        return len(sequences)

    def _print_summary(self, df_sequences: pd.DataFrame):
        """Print summary statistics about the sequences."""
        logger.info("\n" + "=" * 80)
        logger.info("SEQUENCE DATASET SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nTotal sequences: {len(df_sequences):,}")
        logger.info(f"Total laps: {df_sequences['sequence_length'].sum():,}")
        logger.info(f"Total valid laps: {df_sequences['num_valid_laps'].sum():,}")

        logger.info(f"\nSequence length statistics:")
        logger.info(f"  Min:  {df_sequences['sequence_length'].min()}")
        logger.info(f"  Mean: {df_sequences['sequence_length'].mean():.1f}")
        logger.info(f"  Max:  {df_sequences['sequence_length'].max()}")

        logger.info(f"\nValid laps per sequence:")
        logger.info(f"  Min:  {df_sequences['num_valid_laps'].min()}")
        logger.info(f"  Mean: {df_sequences['num_valid_laps'].mean():.1f}")
        logger.info(f"  Max:  {df_sequences['num_valid_laps'].max()}")

        logger.info(f"\nSequences by season:")
        season_counts = df_sequences['season_id'].value_counts().sort_index()
        for season, count in season_counts.items():
            logger.info(f"  {season}: {count:,}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
