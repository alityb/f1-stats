"""
Uncertainty quantification utilities for metrics.

Provides bootstrap confidence intervals and stability flags.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 500,
    confidence_levels: List[float] = [0.68, 0.95],
    random_seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals.

    Args:
        values: Array of stint-level values
        n_bootstrap: Number of bootstrap samples
        confidence_levels: List of confidence levels (e.g., [0.68, 0.95])
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping confidence level to (low, high) bounds
    """
    if len(values) < 2:
        logger.warning("Too few samples for bootstrap, returning point estimate")
        mean_val = float(np.mean(values))
        return {f"ci_{int(cl*100)}": (mean_val, mean_val) for cl in confidence_levels}

    # Use normal approximation for very small samples
    if len(values) < 10:
        logger.info("Small sample size, using normal approximation")
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        sem = std_val / np.sqrt(len(values))

        result = {}
        for cl in confidence_levels:
            # For 68% CI: ±1 std, for 95% CI: ±1.96 std
            z_score = 1.0 if cl == 0.68 else 1.96
            margin = z_score * sem
            result[f"ci_{int(cl*100)}"] = (
                float(mean_val - margin),
                float(mean_val + margin),
            )
        return result

    # Bootstrap resampling
    np.random.seed(random_seed)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentile-based confidence intervals
    result = {}
    for cl in confidence_levels:
        lower_percentile = (1 - cl) / 2 * 100
        upper_percentile = (1 + cl) / 2 * 100

        low = np.percentile(bootstrap_means, lower_percentile)
        high = np.percentile(bootstrap_means, upper_percentile)

        result[f"ci_{int(cl*100)}"] = (float(low), float(high))

    return result


def assess_stability(
    values: np.ndarray, std_threshold: float = 2.0
) -> str:
    """
    Assess metric stability and return warning flag.

    Args:
        values: Stint-level values
        std_threshold: Threshold for high variance flag

    Returns:
        Stability flag: "low_sample", "high_variance", or None
    """
    if len(values) < 5:
        return "low_sample"

    # Check coefficient of variation
    mean_val = np.mean(values)
    std_val = np.std(values)

    if abs(mean_val) < 0.01:  # Near-zero mean
        return None

    cv = abs(std_val / mean_val)
    if cv > std_threshold:
        return "high_variance"

    return None


def paired_bootstrap_delta_ci(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap CI for difference between two metrics.

    Used for comparing drivers with uncertainty.

    Args:
        values1: Stint-level values for driver 1
        values2: Stint-level values for driver 2
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed

    Returns:
        (low, high) CI bounds for delta (driver1 - driver2)
    """
    if len(values1) < 2 or len(values2) < 2:
        # Fallback to point estimate
        delta = float(np.mean(values1) - np.mean(values2))
        return (delta, delta)

    np.random.seed(random_seed)
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        # Resample each driver independently
        sample1 = np.random.choice(values1, size=len(values1), replace=True)
        sample2 = np.random.choice(values2, size=len(values2), replace=True)

        delta = np.mean(sample1) - np.mean(sample2)
        bootstrap_deltas.append(delta)

    bootstrap_deltas = np.array(bootstrap_deltas)

    # Compute percentile CI
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    low = np.percentile(bootstrap_deltas, lower_percentile)
    high = np.percentile(bootstrap_deltas, upper_percentile)

    return (float(low), float(high))


def add_uncertainty_to_metric(
    metric_dict: Dict[str, float], values: np.ndarray
) -> Dict[str, any]:
    """
    Add uncertainty bounds to a metric dictionary.

    Args:
        metric_dict: Existing metric dict (mean, median, std, etc.)
        values: Raw stint-level values

    Returns:
        Updated dict with CI bounds and stability flag
    """
    # Compute CI
    ci_dict = bootstrap_ci(values, n_bootstrap=500)

    # Assess stability
    stability_flag = assess_stability(values)

    # Add to result
    result = metric_dict.copy()
    result["ci_68"] = list(ci_dict["ci_68"])
    result["ci_95"] = list(ci_dict["ci_95"])

    if stability_flag:
        result["stability_flag"] = stability_flag
    else:
        result["stability_flag"] = None

    return result
