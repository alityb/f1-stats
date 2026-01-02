"""
Tyre Management Efficiency (TME) computation.

TME captures how well a driver preserves usable pace as tyres degrade.
It combines degradation slope + late-stint pace stability, normalized
by compound and stint length.

Formula: TME = 0.6 * slope_norm + 0.4 * pace_consistency_norm
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_ingestion.duckdb_persister import DuckDBPersister
from metrics.inference.predictor import PredictionService

logger = logging.getLogger(__name__)


# Baseline expectations by compound (empirical estimates)
# These represent "average" degradation behavior for each compound
COMPOUND_BASELINES = {
    "SOFT": {"slope": 0.05, "pace_ratio": 1.3},
    "MEDIUM": {"slope": 0.03, "pace_ratio": 1.15},
    "HARD": {"slope": 0.02, "pace_ratio": 1.05},
    "INTERMEDIATE": {"slope": 0.04, "pace_ratio": 1.2},
    "WET": {"slope": 0.03, "pace_ratio": 1.1},
}


class TMEComputer:
    """
    Compute Tyre Management Efficiency.

    TME combines degradation slope + late-stint pace stability.
    """

    def __init__(
        self, prediction_service: PredictionService, db_persister: DuckDBPersister
    ):
        """
        Initialize TME computer.

        Args:
            prediction_service: PredictionService for getting model predictions
            db_persister: DuckDBPersister for querying compound info
        """
        self.predictor = prediction_service
        self.db = db_persister

    def compute_stint_tme(
        self, race_id: int, driver_code: str, stint: int, compound: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute TME for a single stint.

        Components:
        1. Degradation slope: linear fit to tyre_deg component
        2. Pace consistency: std of late-stint pace vs early-stint

        Returns TME score normalized by compound and stint length.

        Args:
            race_id: Race ID
            driver_code: Driver code
            stint: Stint number
            compound: Tyre compound (if None, query from DB)

        Returns:
            Dictionary with TME statistics:
                - tme_score: Combined TME score (higher is better)
                - degradation_slope: Tyre deg slope (sec/lap per lap)
                - pace_consistency_ratio: late_std / early_std
                - early_pace_std: Std deviation of early stint pace
                - late_pace_std: Std deviation of late stint pace
                - stint_length: Number of laps
                - compound: Tyre compound
        """
        # Get compound if not provided
        if compound is None:
            compound = self._get_stint_compound(race_id, driver_code, stint)

        # Get predictions
        components = self.predictor.predict_stint(race_id, driver_code, stint)

        tyre_deg = components["tyre_deg"]
        base_pace = components["base_pace"]

        # 1. Degradation slope
        deg_slope = self._compute_degradation_slope(tyre_deg)

        # 2. Pace consistency (late vs early stint)
        stint_length = len(tyre_deg)
        pace_consistency_ratio = self._compute_pace_consistency(
            base_pace, stint_length
        )

        # 3. Combine into TME score
        tme_score = self._compute_tme_score(
            deg_slope, pace_consistency_ratio, compound, stint_length
        )

        # Early/late pace std for debugging
        early_laps = min(5, stint_length // 3)
        late_laps = min(5, stint_length // 3)
        early_std = np.std(base_pace[:early_laps]) if early_laps > 0 else 0.0
        late_std = np.std(base_pace[-late_laps:]) if late_laps > 0 else 0.0

        return {
            "tme_score": float(tme_score),
            "degradation_slope": float(deg_slope),
            "pace_consistency_ratio": float(pace_consistency_ratio),
            "early_pace_std": float(early_std),
            "late_pace_std": float(late_std),
            "stint_length": stint_length,
            "compound": compound,
        }

    def compute_race_tme(
        self, race_id: int, driver_code: str
    ) -> Dict[str, float]:
        """
        Aggregate TME across race stints.

        Args:
            race_id: Race ID
            driver_code: Driver code

        Returns:
            Dictionary with aggregated TME statistics
        """
        stint_predictions = self.predictor.predict_race(race_id, driver_code)

        if not stint_predictions:
            raise ValueError(
                f"No stints found for race_id={race_id}, driver={driver_code}"
            )

        tme_values = []
        for stint_idx, components in stint_predictions.items():
            # Get compound for stint
            compound = self._get_stint_compound(race_id, driver_code, stint_idx)

            tyre_deg = components["tyre_deg"]
            base_pace = components["base_pace"]

            deg_slope = self._compute_degradation_slope(tyre_deg)
            pace_ratio = self._compute_pace_consistency(base_pace, len(tyre_deg))

            tme_score = self._compute_tme_score(
                deg_slope, pace_ratio, compound, len(tyre_deg)
            )
            tme_values.append(tme_score)

        return {
            "mean_tme": float(np.mean(tme_values)),
            "median_tme": float(np.median(tme_values)),
            "std_tme": float(np.std(tme_values)),
            "num_stints": len(tme_values),
        }

    def compute_season_tme_by_compound(
        self, season: int, driver_code: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate TME by compound across season.

        Args:
            season: Season year
            driver_code: Driver code

        Returns:
            Dictionary mapping compound -> TME statistics
        """
        season_predictions = self.predictor.predict_season(season, driver_code)

        if not season_predictions:
            raise ValueError(
                f"No races found for season={season}, driver={driver_code}"
            )

        # Group by compound
        compound_tmes = {}

        for race_id, stint_dict in season_predictions.items():
            for stint_idx, components in stint_dict.items():
                compound = self._get_stint_compound(race_id, driver_code, stint_idx)

                tyre_deg = components["tyre_deg"]
                base_pace = components["base_pace"]

                deg_slope = self._compute_degradation_slope(tyre_deg)
                pace_ratio = self._compute_pace_consistency(base_pace, len(tyre_deg))

                tme_score = self._compute_tme_score(
                    deg_slope, pace_ratio, compound, len(tyre_deg)
                )

                if compound not in compound_tmes:
                    compound_tmes[compound] = []
                compound_tmes[compound].append(tme_score)

        # Aggregate per compound
        result = {}
        for compound, scores in compound_tmes.items():
            result[compound] = {
                "mean_tme": float(np.mean(scores)),
                "median_tme": float(np.median(scores)),
                "std_tme": float(np.std(scores)),
                "num_stints": len(scores),
            }

        return result

    def _compute_degradation_slope(self, tyre_deg: np.ndarray) -> float:
        """
        Compute degradation slope via linear regression.

        Args:
            tyre_deg: Tyre degradation sequence

        Returns:
            Slope (sec/lap per lap)
        """
        if len(tyre_deg) < 2:
            return 0.0

        laps = np.arange(len(tyre_deg))
        slope, _ = np.polyfit(laps, tyre_deg, deg=1)
        return float(slope)

    def _compute_pace_consistency(
        self, base_pace: np.ndarray, stint_length: int
    ) -> float:
        """
        Compute pace consistency ratio (late_std / early_std).

        Higher ratio = worse consistency in late stint.

        Args:
            base_pace: Base pace sequence
            stint_length: Stint length

        Returns:
            Pace consistency ratio
        """
        early_laps = min(5, stint_length // 3)
        late_laps = min(5, stint_length // 3)

        if early_laps == 0 or late_laps == 0:
            return 1.0

        early_pace = base_pace[:early_laps]
        late_pace = base_pace[-late_laps:]

        early_std = np.std(early_pace)
        late_std = np.std(late_pace)

        # Avoid division by zero
        if early_std < 1e-6:
            return 1.0

        return late_std / early_std

    def _compute_tme_score(
        self, deg_slope: float, pace_ratio: float, compound: str, stint_length: int
    ) -> float:
        """
        Normalize TME score by compound and stint length.

        TME Score:
        - Penalize high degradation slope
        - Penalize high late-stint variance
        - Normalize by expected values for compound/stint length

        Args:
            deg_slope: Degradation slope
            pace_ratio: Pace consistency ratio
            compound: Tyre compound
            stint_length: Stint length

        Returns:
            TME score where 0 = average, positive = better, negative = worse
        """
        # Get baseline for compound
        baseline = COMPOUND_BASELINES.get(
            compound.upper(), {"slope": 0.03, "pace_ratio": 1.2}
        )

        # Normalize by baseline (better if lower than baseline)
        slope_norm = (baseline["slope"] - deg_slope) / (baseline["slope"] + 1e-6)
        pace_norm = (baseline["pace_ratio"] - pace_ratio) / (
            baseline["pace_ratio"] + 1e-6
        )

        # Weighted combination (60% slope, 40% pace consistency)
        tme_score = 0.6 * slope_norm + 0.4 * pace_norm

        return tme_score

    def _get_stint_compound(
        self, race_id: int, driver_code: str, stint: int
    ) -> str:
        """
        Query compound from DuckDB for a stint.

        Args:
            race_id: Race ID
            driver_code: Driver code
            stint: Stint number

        Returns:
            Tyre compound (e.g., 'SOFT', 'MEDIUM', 'HARD')
        """
        query = """
            SELECT DISTINCT compound
            FROM laps_raw
            WHERE race_id = ? AND driver_code = ? AND stint = ?
            LIMIT 1
        """
        result = self.db.conn.execute(query, [race_id, driver_code, stint]).df()

        if result.empty:
            logger.warning(
                f"No compound found for race_id={race_id}, driver={driver_code}, stint={stint}"
            )
            return "MEDIUM"  # Default fallback

        return str(result["compound"].iloc[0])
