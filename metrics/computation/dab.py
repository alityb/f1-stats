"""
Driver Ability Benchmark (DAB) computation.

DAB measures the driver's latent contribution relative to their car's
baseline performance. It is explicitly car-normalized, analogous to RAPM
in basketball or WAR-style metrics in baseball.

Formula: DAB = car_baseline - driver_contrib
- Positive DAB = driver is faster than car baseline
- Negative DAB = driver is slower than car baseline
- Zero DAB = driver at car baseline
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from data_ingestion.duckdb_persister import DuckDBPersister
from metrics.inference.predictor import PredictionService

logger = logging.getLogger(__name__)


class DABComputer:
    """
    Compute Driver Ability Benchmark (RAPM-style driver contribution).

    DAB is the driver_contrib component normalized within car+season context.
    """

    def __init__(
        self, prediction_service: PredictionService, db_persister: DuckDBPersister
    ):
        """
        Initialize DAB computer.

        Args:
            prediction_service: PredictionService for getting model predictions
            db_persister: DuckDBPersister for database queries (teammate lookups)
        """
        self.predictor = prediction_service
        self.db = db_persister

    def compute_stint_dab(
        self, race_id: int, driver_code: str, stint: int
    ) -> float:
        """
        Get raw driver contribution for a stint.

        DAB = mean(driver_contrib) across valid laps (constant per stint)

        Args:
            race_id: Race ID
            driver_code: Driver code
            stint: Stint number

        Returns:
            DAB value (sec/lap contribution)
        """
        components = self.predictor.predict_stint(race_id, driver_code, stint)
        driver_contrib = components["driver_contrib"]

        # driver_contrib is constant per stint (broadcast to sequence)
        # Just take the first value
        return float(driver_contrib[0])

    def compute_race_dab(
        self, race_id: int, driver_code: str
    ) -> Dict[str, float]:
        """
        Aggregate DAB for a race (normalized).

        Args:
            race_id: Race ID
            driver_code: Driver code

        Returns:
            Dictionary with normalized DAB statistics
        """
        stint_predictions = self.predictor.predict_race(race_id, driver_code)

        if not stint_predictions:
            raise ValueError(
                f"No stints found for race_id={race_id}, driver={driver_code}"
            )

        # Get season for normalization
        season = self._get_season_for_race(race_id)
        car_baseline = self._get_car_baseline(season, driver_code)

        # Normalize: DAB = baseline - raw_contrib (positive = better)
        dab_values = []
        for stint_idx, components in stint_predictions.items():
            raw_contrib = float(components["driver_contrib"][0])
            normalized_dab = car_baseline - raw_contrib
            dab_values.append(normalized_dab)

        return {
            "mean_dab": float(np.mean(dab_values)),
            "median_dab": float(np.median(dab_values)),
            "std_dab": float(np.std(dab_values)),
            "num_stints": len(dab_values),
        }

    def compute_season_dab(
        self, season: int, driver_code: str, track_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Aggregate DAB across a season with car-normalization.

        Returns driver contribution above/below their car's baseline.

        Args:
            season: Season year
            driver_code: Driver code
            track_id: Optional track filter

        Returns:
            Dictionary with normalized season DAB statistics
        """
        season_predictions = self.predictor.predict_season(
            season, driver_code, track_id
        )

        if not season_predictions:
            raise ValueError(
                f"No races found for season={season}, driver={driver_code}, track={track_id}"
            )

        # Get car baseline for normalization
        car_baseline = self._get_car_baseline(season, driver_code, track_id)

        # Normalize: DAB = baseline - raw_contrib (positive = better)
        dab_values = []
        for race_id, stint_dict in season_predictions.items():
            for stint_idx, components in stint_dict.items():
                raw_contrib = float(components["driver_contrib"][0])
                normalized_dab = car_baseline - raw_contrib
                dab_values.append(normalized_dab)

        return {
            "mean_dab": float(np.mean(dab_values)),
            "median_dab": float(np.median(dab_values)),
            "std_dab": float(np.std(dab_values)),
            "percentile_75": float(np.percentile(dab_values, 75)),
            "percentile_25": float(np.percentile(dab_values, 25)),
            "num_stints": len(dab_values),
        }

    def compute_teammate_delta(
        self, season: int, driver_code: str
    ) -> Dict[str, float]:
        """
        Compute DAB delta vs teammates.

        Uses teammate comparisons to stabilize estimates and reduce car bias.

        Args:
            season: Season year
            driver_code: Driver code

        Returns:
            Dictionary with teammate comparison:
                - driver_dab: Driver's season DAB
                - teammate_deltas: Dict mapping teammate -> DAB delta
                - mean_delta: Mean delta across all teammates
        """
        # Query teammates from same car in season
        teammates = self._get_teammates(season, driver_code)

        # Compute driver's DAB
        driver_dab_stats = self.compute_season_dab(season, driver_code)
        driver_dab = driver_dab_stats["mean_dab"]

        # Compute teammate DABs
        teammate_dabs = {}
        for teammate in teammates:
            try:
                tm_stats = self.compute_season_dab(season, teammate)
                teammate_dabs[teammate] = tm_stats["mean_dab"]
            except Exception as e:
                logger.warning(
                    f"Failed to compute DAB for teammate {teammate} in season {season}: {e}"
                )
                continue

        # Compute deltas (both are already normalized, so delta is meaningful)
        deltas = {}
        for teammate, tm_dab in teammate_dabs.items():
            deltas[teammate] = driver_dab - tm_dab  # Positive = driver better than teammate

        return {
            "driver_dab": driver_dab,
            "teammate_dabs": teammate_dabs,
            "teammate_deltas": deltas,
            "mean_delta": float(np.mean(list(deltas.values())))
            if deltas
            else 0.0,
            "num_teammates": len(teammates),
        }

    def _get_season_for_race(self, race_id: int) -> int:
        """Get season year for a race."""
        query = """
            SELECT season
            FROM race_metadata
            WHERE race_id = ?
        """
        result = self.db.conn.execute(query, [race_id]).df()
        if result.empty:
            raise ValueError(f"Race {race_id} not found")
        return int(result["season"].iloc[0])

    def _get_car_baseline(
        self, season: int, driver_code: str, track_id: Optional[str] = None
    ) -> float:
        """
        Compute car baseline for normalization.

        Car baseline = mean driver_contrib for all drivers in the same car(s)
        that this driver raced in during the season.

        Args:
            season: Season year
            driver_code: Driver code
            track_id: Optional track filter

        Returns:
            Mean driver_contrib for the car (baseline for normalization)
        """
        # Get driver's team(s) in this season
        teammates = self._get_teammates(season, driver_code)
        all_drivers = teammates + [driver_code]  # Include driver themselves

        # Get predictions for all drivers in the car
        all_contribs = []
        for driver in all_drivers:
            try:
                season_preds = self.predictor.predict_season(season, driver, track_id)
                for race_id, stint_dict in season_preds.items():
                    for stint_idx, components in stint_dict.items():
                        raw_contrib = float(components["driver_contrib"][0])
                        all_contribs.append(raw_contrib)
            except Exception as e:
                logger.debug(f"Skipping {driver} for baseline: {e}")
                continue

        if not all_contribs:
            logger.warning(
                f"No car baseline data for season={season}, driver={driver_code}. Using 0."
            )
            return 0.0

        # Return mean as baseline
        return float(np.mean(all_contribs))

    def _get_teammates(self, season: int, driver_code: str) -> List[str]:
        """
        Query teammates from DuckDB.

        Teammates are drivers who raced for the same team(s) as the given driver
        in the specified season.

        Args:
            season: Season year
            driver_code: Driver code

        Returns:
            List of teammate driver codes
        """
        query = """
            WITH driver_teams AS (
                SELECT DISTINCT l.team
                FROM laps_raw l
                JOIN race_metadata r ON l.race_id = r.race_id
                WHERE r.season = ? AND l.driver_code = ?
            )
            SELECT DISTINCT l.driver_code
            FROM laps_raw l
            JOIN race_metadata r ON l.race_id = r.race_id
            WHERE r.season = ?
              AND l.team IN (SELECT team FROM driver_teams)
              AND l.driver_code != ?
            ORDER BY l.driver_code
        """
        result = self.db.conn.execute(
            query, [season, driver_code, season, driver_code]
        ).df()
        return result["driver_code"].tolist()
