"""
api/routes/segments.py
----------------------
GET /nearest-segment — GPS coordinates → nearest survey segment lookup
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..schemas import SegmentResponse
from ..segments import find_nearest
import api.segments as _seg_module

router = APIRouter()


@router.get(
    "/nearest-segment",
    response_model=SegmentResponse,
    summary="Find the nearest 1km survey segment for a GPS coordinate",
)
def nearest_segment(
    lat:       float           = Query(..., description="Latitude (decimal degrees)"),
    lon:       float           = Query(..., description="Longitude (decimal degrees)"),
    road_code: Optional[str]   = Query(None, description="Restrict search to a specific road code, e.g. A001N2"),
    max_dist:  float           = Query(5.0,  description="Maximum search radius in km"),
):
    """
    Returns the nearest survey segment within `max_dist` km.
    Includes the last known VCI and vVCI from the Excel survey for context.

    Used by the web app to show existing survey data before/after a new
    field prediction is made.
    """
    if _seg_module._segments is None:
        raise HTTPException(503, "Segment database not loaded — check server startup logs")

    seg = find_nearest(lat, lon, road_code=road_code, max_dist_km=max_dist)
    if seg is None:
        raise HTTPException(
            404,
            f"No segment found within {max_dist} km of ({lat:.4f}, {lon:.4f})"
            + (f" on road {road_code}" if road_code else ""),
        )

    return SegmentResponse(**seg)
