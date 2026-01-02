"""
Track Position Index (TPI) computation.

TPI represents the underlying, tyre & traffic-adjusted race pace
attributable to the car + driver in clean operating conditions.

Formula: TPI = predicted lap time in clean air (low traffic)
       = base_pace + car_contrib + driver_contrib + tyre_deg
       (excluding traffic_penalty for clean conditions)

TPI is reported in seconds per lap.
"""

import logging
from typing import Dict, Optional

import numpy as np

from metrics.inference.predictor import PredictionService

logger = logging.getLogger(__name__)


class TPIComputer:
    """Compute Track Position Index (clean air pace)."""

    def __init__(self, prediction_service: PredictionService):
        """
        Initialize TPI computer.

        Args:
            prediction_service: PredictionService for getting model predictions
        """
        self.predictor = prediction_service

    def compute_stint_tpi(
        self, race_id: int, driver_code: str, stint: int
    ) -> Dict[str, float]:
        """
        Compute TPI for a single stint.

        TPI = predicted lap time in clean air
            = base_pace + car_contrib + driver_contrib + tyre_deg
            (excluding traffic_penalty)

        Args:
            race_id: Race ID
            driver_code: Driver code (e.g., 'VER')
            stint: Stint number

        Returns:
            Dictionary with TPI statistics (in seconds per lap):
                - mean_tpi: Mean TPI across clean laps
                - median_tpi: Median TPI
                - std_tpi: Standard deviation
                - clean_laps_count: Number of clean laps used
                - early_stint_tpi: Mean TPI in first 5 laps
                - late_stint_tpi: Mean TPI in last 5 laps
        """
        # Get predictions
        components = self.predictor.predict_stint(race_id, driver_code, stint)

        base_pace = components["base_pace"]
        car_contrib = components["car_contrib"]
        driver_contrib = components["driver_contrib"]
        tyre_deg = components["tyre_deg"]
        traffic_penalty = components["traffic_penalty"]

        # TPI = predicted lap time WITHOUT traffic
        # This represents pace in clean air
        tpi_sequence = base_pace + car_contrib + driver_contrib + tyre_deg

        # Filter for clean laps (low traffic penalty)
        clean_mask = traffic_penalty < np.percentile(traffic_penalty, 25)
        tpi_clean = tpi_sequence[clean_mask]

        if len(tpi_clean) == 0:
            logger.warning(
                f"No clean laps found for race_id={race_id}, driver={driver_code}, stint={stint}"
            )
            tpi_clean = tpi_sequence  # Fall back to all laps

        # Compute statistics
        n_laps = len(tpi_sequence)
        early_laps = min(5, n_laps)
        late_laps = min(5, n_laps)

        return {
            "mean_tpi": float(np.mean(tpi_clean)),
            "median_tpi": float(np.median(tpi_clean)),
            "std_tpi": float(np.std(tpi_clean)),
            "clean_laps_count": int(clean_mask.sum()),
            "early_stint_tpi": float(np.mean(tpi_sequence[:early_laps])),
            "late_stint_tpi": float(np.mean(tpi_sequence[-late_laps:])),
        }

    def compute_race_tpi(
        self, race_id: int, driver_code: str
    ) -> Dict[str, float]:
        """
        Aggregate TPI across all stints in a race.

        Args:
            race_id: Race ID
            driver_code: Driver code

        Returns:
            Dictionary with aggregated TPI statistics
        """
        stint_predictions = self.predictor.predict_race(race_id, driver_code)

        if not stint_predictions:
            raise ValueError(
                f"No stints found for race_id={race_id}, driver={driver_code}"
            )

        stint_tpis = []
        for stint_idx, components in stint_predictions.items():
            # Compute from components directly (to avoid re-prediction)
            base_pace = components["base_pace"]
            car_contrib = components["car_contrib"]
            driver_contrib = components["driver_contrib"]
            tyre_deg = components["tyre_deg"]
            traffic_penalty = components["traffic_penalty"]

            # TPI = clean air lap time
            tpi_sequence = base_pace + car_contrib + driver_contrib + tyre_deg

            # Filter clean laps
            clean_mask = traffic_penalty < np.percentile(traffic_penalty, 25)
            tpi_clean = tpi_sequence[clean_mask] if clean_mask.sum() > 0 else tpi_sequence

            stint_tpis.append(float(np.mean(tpi_clean)))

        return {
            "mean_tpi": float(np.mean(stint_tpis)),
            "median_tpi": float(np.median(stint_tpis)),
            "std_tpi": float(np.std(stint_tpis)),
            "num_stints": len(stint_tpis),
        }

    def compute_season_tpi(
        self, season: int, driver_code: str, track_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Aggregate TPI across a season.

        Args:
            season: Season year
            driver_code: Driver code
            track_id: Optional track filter

        Returns:
            Dictionary with season-aggregated TPI statistics
        """
        season_predictions = self.predictor.predict_season(
            season, driver_code, track_id
        )

        if not season_predictions:
            raise ValueError(
                f"No races found for season={season}, driver={driver_code}, track={track_id}"
            )

        race_tpis = []
        for race_id, stint_dict in season_predictions.items():
            for stint_idx, components in stint_dict.items():
                base_pace = components["base_pace"]
                car_contrib = components["car_contrib"]
                driver_contrib = components["driver_contrib"]
                tyre_deg = components["tyre_deg"]
                traffic_penalty = components["traffic_penalty"]

                # TPI = clean air lap time
                tpi_sequence = base_pace + car_contrib + driver_contrib + tyre_deg

                # Filter clean laps
                clean_mask = traffic_penalty < np.percentile(traffic_penalty, 25)
                tpi_clean = tpi_sequence[clean_mask] if clean_mask.sum() > 0 else tpi_sequence

                race_tpis.append(float(np.mean(tpi_clean)))

        return {
            "mean_tpi": float(np.mean(race_tpis)),
            "median_tpi": float(np.median(race_tpis)),
            "std_tpi": float(np.std(race_tpis)),
            "percentile_75": float(np.percentile(race_tpis, 75)),
            "percentile_25": float(np.percentile(race_tpis, 25)),
            "num_stints": len(race_tpis),
        }

