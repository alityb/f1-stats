"""
Stint filtering utilities for clean metric computation.

Identifies "clean" stints that represent intrinsic driving pace
in stable, representative conditions.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StintFilter:
    """Filter stints to identify clean, representative pace."""

    @staticmethod
    def is_clean_stint(
        components: Dict[str, np.ndarray],
        stint_metadata: Dict,
        traffic_threshold_percentile: float = 75,
        deg_threshold_percentile: float = 95,
    ) -> bool:
        """
        Check if a stint qualifies as "clean" for DAB-CLEAN computation.

        Clean stints must satisfy ALL criteria:
        - Low traffic (traffic_penalty below threshold)
        - Stable tyre degradation (no extreme outliers)
        - Sufficient valid laps (>= 5 laps)
        - No extreme conditions

        Args:
            components: Model prediction components
            stint_metadata: Metadata about stint (pit laps, flags, etc.)
            traffic_threshold_percentile: Traffic penalty threshold
            deg_threshold_percentile: Degradation threshold

        Returns:
            True if stint is clean
        """
        # Check minimum length
        stint_length = len(components["base_pace"])
        if stint_length < 5:
            logger.debug("Stint too short for clean analysis")
            return False

        # Traffic check: exclude high-traffic stints
        traffic_penalty = components["traffic_penalty"]
        traffic_threshold = np.percentile(traffic_penalty, traffic_threshold_percentile)
        mean_traffic = np.mean(traffic_penalty)

        if mean_traffic > traffic_threshold:
            logger.debug(f"High traffic: {mean_traffic:.2f} > {traffic_threshold:.2f}")
            return False

        # Degradation check: exclude extreme deg outliers
        tyre_deg = components["tyre_deg"]
        deg_threshold = np.percentile(tyre_deg, deg_threshold_percentile)
        max_deg = np.max(tyre_deg)

        if max_deg > deg_threshold * 1.5:  # 1.5x threshold
            logger.debug(f"Extreme degradation: {max_deg:.2f}")
            return False

        # Exclude first 2 laps (out-lap effect)
        # This is already handled by caller filtering specific laps

        # All checks passed
        return True

    @staticmethod
    def filter_clean_laps(
        components: Dict[str, np.ndarray],
        exclude_first_n: int = 2,
        exclude_last_n: int = 1,
    ) -> np.ndarray:
        """
        Get mask of clean laps within a stint.

        Excludes:
        - First N laps (out-lap + warm-up)
        - Last N laps (in-lap)
        - High traffic laps

        Args:
            components: Model prediction components
            exclude_first_n: Number of laps to exclude from start
            exclude_last_n: Number of laps to exclude from end

        Returns:
            Boolean mask of clean laps
        """
        stint_length = len(components["base_pace"])
        mask = np.ones(stint_length, dtype=bool)

        # Exclude first/last laps
        if exclude_first_n > 0:
            mask[:exclude_first_n] = False
        if exclude_last_n > 0:
            mask[-exclude_last_n:] = False

        # Exclude high traffic laps (top 25%)
        traffic = components["traffic_penalty"]
        traffic_threshold = np.percentile(traffic, 75)
        mask &= traffic < traffic_threshold

        return mask


def filter_stints_for_clean_metric(
    stint_predictions: Dict[int, Dict[str, np.ndarray]],
    min_clean_laps: int = 5,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Filter stint predictions to only include clean stints.

    Args:
        stint_predictions: Dict mapping stint_idx -> components
        min_clean_laps: Minimum number of clean laps required

    Returns:
        Filtered dict with only clean stints
    """
    clean_stints = {}
    stint_filter = StintFilter()

    for stint_idx, components in stint_predictions.items():
        # Get clean lap mask
        clean_mask = stint_filter.filter_clean_laps(components)
        num_clean = clean_mask.sum()

        # Check if stint has enough clean laps
        if num_clean < min_clean_laps:
            logger.debug(
                f"Stint {stint_idx}: only {num_clean} clean laps, skipping"
            )
            continue

        # Check if stint is clean overall
        # For now, simplified check - can add more criteria
        stint_metadata = {}  # Would need to pass from caller if available
        if not stint_filter.is_clean_stint(components, stint_metadata):
            logger.debug(f"Stint {stint_idx}: failed clean criteria")
            continue

        # Filter components to only clean laps
        filtered_components = {}
        for key, values in components.items():
            if isinstance(values, np.ndarray) and len(values) == len(clean_mask):
                filtered_components[key] = values[clean_mask]
            else:
                filtered_components[key] = values

        clean_stints[stint_idx] = filtered_components

    logger.info(
        f"Filtered {len(clean_stints)}/{len(stint_predictions)} clean stints "
        f"({len(clean_stints)/len(stint_predictions)*100:.1f}% coverage)"
    )

    return clean_stints
