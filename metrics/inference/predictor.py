"""
Prediction service for running model inference on stint data.

Loads data from DuckDB, prepares batches, and runs model predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from data_ingestion.duckdb_persister import DuckDBPersister
from ml_dataset.stint_dataset import collate_stints
from metrics.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for running model predictions on stint data."""

    def __init__(self, model_loader: ModelLoader, db_persister: DuckDBPersister):
        """
        Initialize prediction service.

        Args:
            model_loader: ModelLoader instance with loaded model
            db_persister: DuckDBPersister for database access
        """
        self.model = model_loader.model
        self.vocab = model_loader.vocab_mappings
        self.db = db_persister

    def predict_stint(
        self, race_id: int, driver_code: str, stint: int
    ) -> Dict[str, np.ndarray]:
        """
        Get model predictions for a single stint.

        Args:
            race_id: Race ID
            driver_code: Driver code (e.g., 'VER', 'HAM')
            stint: Stint number

        Returns:
            Dictionary with component predictions:
                - base_pace: [num_laps]
                - car_contrib: [num_laps]
                - driver_contrib: [num_laps]
                - tyre_deg: [num_laps]
                - traffic_penalty: [num_laps]
                - lap_time_pred: [num_laps]
        """
        # Load stint data from database
        stint_data = self._load_stint_from_db(race_id, driver_code, stint)

        if stint_data is None or len(stint_data) == 0:
            raise ValueError(
                f"No data found for race_id={race_id}, driver={driver_code}, stint={stint}"
            )

        # Prepare batch
        batch = self._prepare_batch(stint_data)

        # Run inference
        with torch.no_grad():
            output = self.model(batch)

        # Extract components (remove batch dimension)
        components = {
            name: values[0, :].cpu().numpy()  # [B=1, T] -> [T]
            for name, values in output["components"].items()
        }

        # Add predicted lap time
        components["lap_time_pred"] = output["lap_time_pred"][0, :].cpu().numpy()

        # Trim to actual sequence length (remove padding)
        seq_len = batch["sequence_lengths"][0].item()
        components = {k: v[:seq_len] for k, v in components.items()}

        return components

    def predict_race(
        self, race_id: int, driver_code: str
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Get predictions for all stints in a race.

        Args:
            race_id: Race ID
            driver_code: Driver code

        Returns:
            Dict mapping stint_index -> components dict
        """
        # Query all stints for driver in race
        stints = self._get_stints_for_race(race_id, driver_code)

        predictions = {}
        for stint_idx in stints:
            try:
                predictions[stint_idx] = self.predict_stint(race_id, driver_code, stint_idx)
            except Exception as e:
                logger.warning(
                    f"Failed to predict stint {stint_idx} for {driver_code} in race {race_id}: {e}"
                )
                continue

        return predictions

    def predict_season(
        self, season: int, driver_code: str, track_id: Optional[str] = None
    ) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
        """
        Get predictions for all races in a season.

        Args:
            season: Season year
            driver_code: Driver code
            track_id: Optional track filter

        Returns:
            Dict mapping race_id -> stint_index -> components dict
        """
        # Query all races for driver in season
        races = self._get_races_for_season(season, driver_code, track_id)

        predictions = {}
        for race_id in races:
            try:
                predictions[race_id] = self.predict_race(race_id, driver_code)
            except Exception as e:
                logger.warning(
                    f"Failed to predict race {race_id} for {driver_code}: {e}"
                )
                continue

        return predictions

    def _load_stint_from_db(
        self, race_id: int, driver_code: str, stint: int
    ) -> Optional[pd.DataFrame]:
        """
        Load stint laps from DuckDB.

        Returns:
            DataFrame with columns: lap_number, compound, tyre_life, lap_in_stint,
                                   stint_length, fuel_proxy, lap_time_seconds, is_accurate,
                                   driver_id, car_id, track_id, season_id
        """
        query = """
            SELECT
                l.race_id,
                l.stint,
                l.lap_number,
                l.driver_code,
                l.team,
                l.compound,
                l.tyre_life,
                l.lap_time_seconds,
                l.is_accurate,
                r.season,
                r.circuit_key as track_id,
                -- Compute derived features
                ROW_NUMBER() OVER (PARTITION BY l.race_id, l.driver_code, l.stint ORDER BY l.lap_number) as lap_in_stint,
                COUNT(*) OVER (PARTITION BY l.race_id, l.driver_code, l.stint) as stint_length,
                CAST(l.lap_number AS DOUBLE) / NULLIF((SELECT MAX(lap_number) FROM laps_raw WHERE race_id = l.race_id), 0) as fuel_proxy
            FROM laps_raw l
            JOIN race_metadata r ON l.race_id = r.race_id
            WHERE l.race_id = ?
              AND l.driver_code = ?
              AND l.stint = ?
            ORDER BY l.lap_number
        """
        df = self.db.conn.execute(query, [race_id, driver_code, stint]).df()

        if df.empty:
            return None

        return df

    def _get_stints_for_race(self, race_id: int, driver_code: str) -> List[int]:
        """Get list of stint numbers for a driver in a race."""
        query = """
            SELECT DISTINCT stint
            FROM laps_raw
            WHERE race_id = ? AND driver_code = ? AND stint IS NOT NULL
            ORDER BY stint
        """
        df = self.db.conn.execute(query, [race_id, driver_code]).df()
        return df["stint"].tolist()

    def _get_races_for_season(
        self, season: int, driver_code: str, track_id: Optional[str] = None
    ) -> List[int]:
        """Get list of race IDs for a driver in a season."""
        if track_id:
            query = """
                SELECT DISTINCT r.race_id
                FROM race_metadata r
                JOIN laps_raw l ON r.race_id = l.race_id
                WHERE r.season = ? AND l.driver_code = ? AND r.circuit_key = ?
                ORDER BY r.race_id
            """
            params = [season, driver_code, track_id]
        else:
            query = """
                SELECT DISTINCT r.race_id
                FROM race_metadata r
                JOIN laps_raw l ON r.race_id = l.race_id
                WHERE r.season = ? AND l.driver_code = ?
                ORDER BY r.race_id
            """
            params = [season, driver_code]

        df = self.db.conn.execute(query, params).df()
        return df["race_id"].tolist()

    def _prepare_batch(self, stint_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Convert DataFrame to model input batch.

        Based on pattern from StintDataset.__getitem__ and collate_stints.
        """
        # Extract stint-level info (same for all laps)
        driver_code = stint_df["driver_code"].iloc[0]
        team = stint_df["team"].iloc[0]
        track_id = stint_df["track_id"].iloc[0]
        season = int(stint_df["season"].iloc[0])

        # Create car_id (team_season combination)
        car_id = f"{team}_{season}"

        # Map to vocab indices
        driver_idx = self.vocab["driver_id"].get(
            driver_code, self.vocab["driver_id"]["<UNK>"]
        )
        car_idx = self.vocab["car_id"].get(car_id, self.vocab["car_id"]["<UNK>"])
        track_idx = self.vocab["track_id"].get(
            track_id, self.vocab["track_id"]["<UNK>"]
        )
        season_idx = self.vocab["season_id"].get(season, 0)

        # Sequence features
        seq_len = len(stint_df)
        compounds = [
            self.vocab["compound"].get(c, self.vocab["compound"]["<UNK>"])
            for c in stint_df["compound"]
        ]

        # Build single-item batch (following collate_stints pattern)
        batch = {
            # Stint-level categorical (batch_size=1)
            "driver_id": torch.tensor([driver_idx], dtype=torch.long),
            "car_id": torch.tensor([car_idx], dtype=torch.long),
            "track_id": torch.tensor([track_idx], dtype=torch.long),
            "season_id": torch.tensor([season_idx], dtype=torch.long),
            # Sequence features (batch_size=1, seq_len)
            "compound": torch.tensor([compounds], dtype=torch.long),
            "lap_in_stint": torch.tensor(
                [stint_df["lap_in_stint"].values], dtype=torch.float32
            ),
            "tyre_age": torch.tensor(
                [stint_df["tyre_life"].values], dtype=torch.float32
            ),
            "fuel_proxy": torch.tensor(
                [stint_df["fuel_proxy"].values], dtype=torch.float32
            ),
            "stint_length": torch.tensor(
                [stint_df["stint_length"].values], dtype=torch.float32
            ),
            "lap_time": torch.tensor(
                [stint_df["lap_time_seconds"].values], dtype=torch.float32
            ),
            "valid_mask": torch.tensor(
                [stint_df["is_accurate"].values], dtype=torch.bool
            ),
            "padding_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "sequence_lengths": torch.tensor([seq_len], dtype=torch.long),
            # Metadata
            "race_id": torch.tensor([stint_df["race_id"].iloc[0]], dtype=torch.long)
            if "race_id" in stint_df.columns
            else None,
            "stint_index": torch.tensor([stint_df["stint"].iloc[0]], dtype=torch.long)
            if "stint" in stint_df.columns
            else None,
        }

        return batch
