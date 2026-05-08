"""
api/routes/predict.py
---------------------
POST /predict        — single segment (1+ images)
POST /predict-batch  — zip archive of images (ministry batch use)
"""

import io
import zipfile
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..auth import verify_api_key
from ..image_utils import detect_source, extract_gps, load_image, preprocess
from ..inference import predict
from ..schemas import (
    BatchImageResult,
    BatchPredictResponse,
    DefectPrediction,
    GPSInfo,
    PredictResponse,
    RoadMatch,
)
from ..segments import find_nearest

router = APIRouter()


def _build_road_match(gps: GPSInfo | None, road_code: str | None = None) -> RoadMatch | None:
    if gps is None:
        return None
    seg = find_nearest(gps.lat, gps.lon, road_code=road_code)
    if seg is None:
        return None
    return RoadMatch(
        road_code   = seg["road_code"],
        road_name   = seg["road_name"],
        km_start    = seg["km_start"],
        km_end      = seg["km_end"],
        survey_year = seg["survey_year"],
        vci_survey  = seg["vci"],
        vvci_survey = seg["vvci"],
    )


def _process_images(files_data: list[tuple[str, bytes]]) -> PredictResponse:
    """Core logic shared by /predict and /predict-batch single-segment calls."""
    tensors   = []
    gps_used  = None

    for _, data in files_data:
        img        = load_image(data)
        is_mobicap = detect_source(img) == "mobicap"
        tensors.append(preprocess(img, is_mobicap))

        if gps_used is None:
            lat, lon, src = extract_gps(img)
            if lat is not None:
                gps_used = GPSInfo(lat=lat, lon=lon, source=src)

    result = predict(tensors)

    road_matched = _build_road_match(gps_used)

    return PredictResponse(
        vvci         = result.get("vvci") or 0.0,
        vvci_label   = result.get("vvci_label") or "Unknown",
        pci          = result.get("pci"),
        pci_label    = result.get("pci_label"),
        defects      = [DefectPrediction(**d) for d in result.get("defects", [])],
        gps_used     = gps_used,
        road_matched = road_matched,
        images_used  = len(tensors),
        model_ready  = result.get("model_ready", False),
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse, summary="Predict VCI from 1+ images")
async def predict_endpoint(
    files:   List[UploadFile] = File(..., description="One or more pavement images for a single segment"),
    api_key: str = Depends(verify_api_key),
):
    """
    Upload 1 or more images from the same 1km road segment.
    Accepts Mobicap images (GPS in blue header) and smartphone images (GPS in EXIF).

    Multiple images are aggregated by averaging backbone feature vectors before
    the prediction heads — more accurate than predicting each image separately.
    """
    if not files:
        raise HTTPException(400, "At least one image file is required")

    files_data = [(f.filename, await f.read()) for f in files]
    return _process_images(files_data)


# ---------------------------------------------------------------------------
# POST /predict-batch
# ---------------------------------------------------------------------------

@router.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    summary="Batch predict from a zip of images",
)
async def predict_batch_endpoint(
    archive: UploadFile = File(..., description="ZIP file containing pavement images"),
    api_key: str = Depends(verify_api_key),
):
    """
    Upload a ZIP archive of images for batch processing (ministry use case).
    Images are processed individually — each image gets its own prediction row.
    Returns a JSON list; download as CSV via the Accept: text/csv header.

    For Mobicap archives where images should be grouped by segment, use the
    GPS coordinates in each image's blue header to group nearby images
    automatically.
    """
    raw = await archive.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Uploaded file is not a valid ZIP archive")

    image_names = [
        n for n in zf.namelist()
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
        and not n.startswith("__MACOSX")
    ]
    if not image_names:
        raise HTTPException(400, "No .jpg/.jpeg/.png images found in the ZIP")

    results = []
    for name in image_names:
        try:
            data = zf.read(name)
            resp = _process_images([(name, data)])
            results.append(BatchImageResult(
                filename     = name,
                vvci         = resp.vvci,
                vvci_label   = resp.vvci_label,
                pci          = resp.pci,
                pci_label    = resp.pci_label,
                defects      = resp.defects,
                gps_used     = resp.gps_used,
                road_matched = resp.road_matched,
            ))
        except Exception as e:
            results.append(BatchImageResult(
                filename=name, vvci=None, vvci_label=None,
                defects=None, gps_used=None, road_matched=None,
                error=str(e),
            ))

    return BatchPredictResponse(total=len(results), results=results)
