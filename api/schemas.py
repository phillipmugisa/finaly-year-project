"""
api/schemas.py
--------------
Pydantic request/response models for the VCI Estimator API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DefectPrediction(BaseModel):
    name:            str
    predicted_grade: int   = Field(..., ge=1, le=5, description="1=no defect … 5=worst")
    confidence:      float = Field(..., ge=0.0, le=1.0)


class GPSInfo(BaseModel):
    lat:    float
    lon:    float
    source: str   = Field(..., description="'exif' | 'ocr' | 'provided'")


class RoadMatch(BaseModel):
    road_code:  str
    road_name:  str
    km_start:   float
    km_end:     float
    survey_year: Optional[int]
    vci_survey:  Optional[float] = Field(None, description="Last known full VCI from Excel survey")
    vvci_survey: Optional[float] = Field(None, description="Last known visual VCI from Excel survey")


class PredictResponse(BaseModel):
    vvci:        float = Field(..., description="Predicted visual VCI [0–100]")
    vvci_label:  str   = Field(..., description="Good (>80) | Fair (60–80) | Poor (40–60) | Bad (<40)")
    pci:         Optional[float] = Field(None, description="Approximate PCI [0–100] derived from predicted defect grades (ASTM D6433 formula)")
    pci_label:   Optional[str]  = Field(None, description="Good | Satisfactory | Fair | Poor | Very Poor | Serious | Failed")
    defects:     List[DefectPrediction]
    gps_used:    Optional[GPSInfo]    = None
    road_matched: Optional[RoadMatch] = None
    images_used: int
    model_ready: bool


class BatchImageResult(BaseModel):
    filename:    str
    vvci:        Optional[float]
    vvci_label:  Optional[str]
    pci:         Optional[float]
    pci_label:   Optional[str]
    defects:     Optional[List[DefectPrediction]]
    gps_used:    Optional[GPSInfo]
    road_matched: Optional[RoadMatch]
    error:       Optional[str] = None


class BatchPredictResponse(BaseModel):
    total:   int
    results: List[BatchImageResult]


class SegmentResponse(BaseModel):
    road_code:    str
    road_name:    str
    km_start:     float
    km_end:       float
    lat_centroid: float
    lon_centroid: float
    survey_year:  Optional[int]
    vci:          Optional[float]
    vvci:         Optional[float]
    distance_km:  float
