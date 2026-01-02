"""
Pydantic response models for F1 Stats API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MetricScope(BaseModel):
    """Scope parameters for a metric query."""

    race_id: Optional[int] = None
    season: Optional[int] = None
    track_id: Optional[str] = None
    compound: Optional[str] = None


class TPIMetricValue(BaseModel):
    """TPI metric value with statistics."""

    mean_tpi: float = Field(..., description="Mean TPI across clean laps")
    median_tpi: float = Field(..., description="Median TPI")
    std_tpi: float = Field(..., description="Standard deviation of TPI")
    clean_laps_count: Optional[int] = Field(
        None, description="Number of clean laps (stint-level only)"
    )
    early_stint_tpi: Optional[float] = Field(
        None, description="Mean TPI in first 5 laps (stint-level only)"
    )
    late_stint_tpi: Optional[float] = Field(
        None, description="Mean TPI in last 5 laps (stint-level only)"
    )
    num_stints: Optional[int] = Field(
        None, description="Number of stints aggregated (race/season-level)"
    )
    percentile_75: Optional[float] = Field(None, description="75th percentile")
    percentile_25: Optional[float] = Field(None, description="25th percentile")


class DABMetricValue(BaseModel):
    """DAB metric value with statistics."""

    mean_dab: float = Field(
        ..., description="Mean driver contribution (sec/lap above car baseline)"
    )
    median_dab: float = Field(..., description="Median DAB")
    std_dab: float = Field(..., description="Standard deviation of DAB")
    num_stints: int = Field(..., description="Number of stints aggregated")
    percentile_75: Optional[float] = Field(None, description="75th percentile")
    percentile_25: Optional[float] = Field(None, description="25th percentile")


class TMEMetricValue(BaseModel):
    """TME metric value with statistics."""

    mean_tme: Optional[float] = Field(None, description="Mean TME score")
    median_tme: Optional[float] = Field(None, description="Median TME score")
    std_tme: Optional[float] = Field(None, description="Standard deviation of TME")
    num_stints: Optional[int] = Field(None, description="Number of stints aggregated")
    # Stint-level details
    tme_score: Optional[float] = Field(
        None, description="TME score (stint-level only)"
    )
    degradation_slope: Optional[float] = Field(
        None, description="Tyre degradation slope (sec/lap per lap)"
    )
    pace_consistency_ratio: Optional[float] = Field(
        None, description="Late/early pace std ratio"
    )
    early_pace_std: Optional[float] = Field(
        None, description="Early stint pace std deviation"
    )
    late_pace_std: Optional[float] = Field(
        None, description="Late stint pace std deviation"
    )
    stint_length: Optional[int] = Field(None, description="Stint length in laps")
    compound: Optional[str] = Field(None, description="Tyre compound")


class MetricResponse(BaseModel):
    """Generic metric response."""

    driver_code: str = Field(..., description="Driver code (e.g., 'VER', 'HAM')")
    metric_type: str = Field(..., description="Metric type ('tpi', 'dab', 'tme')")
    scope: MetricScope = Field(..., description="Query scope")
    value: Dict = Field(..., description="Metric value (type depends on metric_type)")


class DriverComparisonItem(BaseModel):
    """Single driver in comparison."""

    driver_code: str
    value: Dict
    rank: int


class DriverComparisonResponse(BaseModel):
    """Driver comparison response."""

    metric_type: str = Field(..., description="Metric type being compared")
    scope: MetricScope = Field(..., description="Comparison scope")
    drivers: List[DriverComparisonItem] = Field(
        ..., description="Driver comparison data"
    )
    statistics: Dict[str, float] = Field(
        ..., description="Aggregate statistics (max_delta, mean_value, std_value)"
    )


class TeammateComparisonResponse(BaseModel):
    """Teammate comparison response."""

    driver_code: str
    season: int
    driver_dab: float = Field(..., description="Driver's season DAB")
    teammate_dabs: Dict[str, float] = Field(
        ..., description="Teammate DAB values"
    )
    teammate_deltas: Dict[str, float] = Field(
        ..., description="DAB deltas vs teammates"
    )
    mean_delta: float = Field(..., description="Mean delta across all teammates")
    num_teammates: int = Field(..., description="Number of teammates")


class DriverRankingItem(BaseModel):
    """Single driver ranking."""

    rank: int
    driver_code: str
    value: float
    num_stints: int


class DriverRankingsResponse(BaseModel):
    """Driver rankings response."""

    metric_type: str
    season: int
    track_id: Optional[str] = None
    rankings: List[DriverRankingItem]
    total_drivers: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loaded status")
    db_connected: bool = Field(..., description="Database connection status")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    checkpoint_path: str
    epoch: Optional[int] = None
    best_val_metric: Optional[float] = None
    vocab_sizes: Dict[str, int]
