"""
Driver comparison endpoints for F1 Stats API.
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_dab_computer, get_tme_computer, get_tpi_computer
from api.schemas.responses import (
    DriverComparisonItem,
    DriverComparisonResponse,
    MetricScope,
    TeammateComparisonResponse,
)
from metrics.computation.dab import DABComputer
from metrics.computation.tme import TMEComputer
from metrics.computation.tpi import TPIComputer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/compare", response_model=DriverComparisonResponse)
async def compare_drivers(
    driver_codes: List[str] = Query(
        ..., min_length=2, max_length=10, description="Driver codes to compare"
    ),
    metric_type: str = Query(
        ..., pattern="^(tpi|dab|tme)$", description="Metric type"
    ),
    race_id: Optional[int] = Query(None, description="Race ID"),
    season: Optional[int] = Query(None, description="Season year"),
    track_id: Optional[str] = Query(None, description="Track/circuit key"),
    tpi_computer: TPIComputer = Depends(get_tpi_computer),
    dab_computer: DABComputer = Depends(get_dab_computer),
    tme_computer: TMEComputer = Depends(get_tme_computer),
):
    """
    Compare drivers across specified metric.

    Examples:
    - `/drivers/compare?driver_codes=VER&driver_codes=HAM&metric_type=dab&season=2024`
    - `/drivers/compare?driver_codes=VER&driver_codes=HAM&driver_codes=LEC&metric_type=tpi&race_id=55`
    """
    try:
        # Select appropriate computer
        if metric_type == "tpi":
            computer = tpi_computer
            compute_fn = (
                computer.compute_race_tpi
                if race_id
                else lambda d: computer.compute_season_tpi(season, d, track_id)
            )
            key = "mean_tpi"
        elif metric_type == "dab":
            computer = dab_computer
            compute_fn = (
                computer.compute_race_dab
                if race_id
                else lambda d: computer.compute_season_dab(season, d, track_id)
            )
            key = "mean_dab"
        elif metric_type == "tme":
            computer = tme_computer
            compute_fn = (
                computer.compute_race_tme
                if race_id
                else lambda d: computer.compute_season_tme_by_compound(season, d)
            )
            key = "mean_tme"
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid metric type: {metric_type}"
            )

        # Validate parameters
        if race_id is None and season is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide either race_id or season parameter",
            )

        # Compute metrics for each driver
        results = {}
        for driver_code in driver_codes:
            try:
                value = compute_fn(driver_code)
                results[driver_code] = value
            except Exception as e:
                logger.warning(
                    f"Failed to compute {metric_type} for {driver_code}: {e}"
                )
                continue

        if not results:
            raise HTTPException(
                status_code=404, detail="No data found for any driver"
            )

        # Rank drivers
        # For TME: if season-level without compound filter, we get dict of compounds
        # Use mean_tme from first compound or overall mean
        sortable_values = []
        for driver_code, value in results.items():
            if isinstance(value, dict):
                if key in value:
                    sortable_values.append((driver_code, value[key]))
                else:
                    # TME by compound - compute overall mean
                    compound_means = [v.get("mean_tme", 0) for v in value.values()]
                    sortable_values.append((driver_code, np.mean(compound_means)))
            else:
                sortable_values.append((driver_code, value))

        # Sort based on metric type:
        # TPI: Lower is better (faster lap time)
        # DAB: Higher is better (positive = faster than car baseline)
        # TME: Higher is better (better tyre management)
        reverse = metric_type in ["dab", "tme"]  # Higher is better for DAB and TME
        sorted_drivers = sorted(sortable_values, key=lambda x: x[1], reverse=reverse)

        # Create comparison items with ranks
        comparison_items = []
        for rank, (driver_code, _) in enumerate(sorted_drivers, 1):
            comparison_items.append(
                DriverComparisonItem(
                    driver_code=driver_code, value=results[driver_code], rank=rank
                )
            )

        # Compute statistics
        values = [x[1] for x in sortable_values]
        statistics = {
            "max_delta": float(max(values) - min(values)),
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
        }

        return DriverComparisonResponse(
            metric_type=metric_type,
            scope=MetricScope(
                race_id=race_id, season=season, track_id=track_id
            ),
            drivers=comparison_items,
            statistics=statistics,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in driver comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/teammate-comparison/{driver_code}",
    response_model=TeammateComparisonResponse,
)
async def teammate_comparison(
    driver_code: str,
    season: int = Query(..., description="Season year"),
    dab_computer: DABComputer = Depends(get_dab_computer),
):
    """
    Compare driver to teammates in a season (DAB metric).

    Example:
    - `/drivers/teammate-comparison/VER?season=2024`
    """
    try:
        result = dab_computer.compute_teammate_delta(season, driver_code)

        return TeammateComparisonResponse(
            driver_code=driver_code,
            season=season,
            driver_dab=result["driver_dab"],
            teammate_dabs=result["teammate_dabs"],
            teammate_deltas=result["teammate_deltas"],
            mean_delta=result["mean_delta"],
            num_teammates=result["num_teammates"],
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in teammate comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
