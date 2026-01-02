"""
Metric endpoints for F1 Stats API.

Provides TPI, DAB, and TME metric retrieval.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_dab_computer, get_tme_computer, get_tpi_computer
from api.schemas.responses import MetricResponse, MetricScope
from metrics.computation.dab import DABComputer
from metrics.computation.tme import TMEComputer
from metrics.computation.tpi import TPIComputer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tpi", response_model=MetricResponse)
async def get_tpi(
    driver_code: str = Query(..., description="Driver code (e.g., 'VER', 'HAM')"),
    race_id: Optional[int] = Query(None, description="Race ID"),
    season: Optional[int] = Query(None, description="Season year"),
    track_id: Optional[str] = Query(None, description="Track/circuit key"),
    tpi_computer: TPIComputer = Depends(get_tpi_computer),
):
    """
    Get Track Position Index (TPI) for a driver.

    TPI represents clean air pace (tyre & traffic-adjusted).

    **Priority**: race_id > season > track_id

    Examples:
    - `/metrics/tpi?driver_code=VER&race_id=55` - TPI for VER in race 55
    - `/metrics/tpi?driver_code=HAM&season=2024` - Season TPI for HAM
    - `/metrics/tpi?driver_code=LEC&season=2024&track_id=monaco` - Track-specific TPI
    """
    try:
        if race_id is not None:
            # Race-level TPI
            value = tpi_computer.compute_race_tpi(race_id, driver_code)
        elif season is not None:
            # Season-level TPI
            value = tpi_computer.compute_season_tpi(season, driver_code, track_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either race_id or season parameter",
            )

        return MetricResponse(
            driver_code=driver_code,
            metric_type="tpi",
            scope=MetricScope(
                race_id=race_id, season=season, track_id=track_id
            ),
            value=value,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error computing TPI: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dab", response_model=MetricResponse)
async def get_dab(
    driver_code: str = Query(..., description="Driver code"),
    race_id: Optional[int] = Query(None, description="Race ID"),
    season: Optional[int] = Query(None, description="Season year"),
    track_id: Optional[str] = Query(None, description="Track/circuit key"),
    dab_computer: DABComputer = Depends(get_dab_computer),
):
    """
    Get Driver Ability Benchmark (DAB) for a driver.

    DAB is a car-normalized driver contribution metric (sec/lap above car baseline).

    **Priority**: race_id > season > track_id

    Examples:
    - `/metrics/dab?driver_code=VER&race_id=55` - DAB for VER in race 55
    - `/metrics/dab?driver_code=HAM&season=2024` - Season DAB for HAM
    """
    try:
        if race_id is not None:
            # Race-level DAB
            value = dab_computer.compute_race_dab(race_id, driver_code)
        elif season is not None:
            # Season-level DAB
            value = dab_computer.compute_season_dab(season, driver_code, track_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either race_id or season parameter",
            )

        return MetricResponse(
            driver_code=driver_code,
            metric_type="dab",
            scope=MetricScope(
                race_id=race_id, season=season, track_id=track_id
            ),
            value=value,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error computing DAB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tme", response_model=MetricResponse)
async def get_tme(
    driver_code: str = Query(..., description="Driver code"),
    race_id: Optional[int] = Query(None, description="Race ID"),
    season: Optional[int] = Query(None, description="Season year"),
    compound: Optional[str] = Query(None, description="Tyre compound filter"),
    tme_computer: TMEComputer = Depends(get_tme_computer),
):
    """
    Get Tyre Management Efficiency (TME) for a driver.

    TME combines degradation slope + late-stint pace stability.

    **Priority**: race_id > season

    Examples:
    - `/metrics/tme?driver_code=VER&race_id=55` - TME for VER in race 55
    - `/metrics/tme?driver_code=HAM&season=2024` - Season TME by compound
    - `/metrics/tme?driver_code=LEC&season=2024&compound=SOFT` - SOFT tyre TME
    """
    try:
        if race_id is not None:
            # Race-level TME
            value = tme_computer.compute_race_tme(race_id, driver_code)
        elif season is not None:
            # Season-level TME by compound
            value = tme_computer.compute_season_tme_by_compound(season, driver_code)

            # Filter by compound if specified
            if compound:
                compound_upper = compound.upper()
                if compound_upper in value:
                    value = value[compound_upper]
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No data for compound {compound_upper}",
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either race_id or season parameter",
            )

        return MetricResponse(
            driver_code=driver_code,
            metric_type="tme",
            scope=MetricScope(
                race_id=race_id, season=season, compound=compound
            ),
            value=value,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing TME: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
