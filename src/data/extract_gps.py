"""
extract_gps.py
--------------
Unified GPS extractor for pavement images from two sources:

1. Smartphone / standard camera images
   GPS is in EXIF metadata (GPSLatitude / GPSLongitude tags).
   All modern Android and iOS cameras write this automatically.

2. Mobicap survey system images
   GPS is burned into a blue text overlay in the top 80px of the image.
   No EXIF GPS is present. Extracted via OCR (pytesseract).

The single public function get_gps(image_path) tries EXIF first, then OCR,
and returns (lat, lon, source) where source is 'exif' | 'ocr' | None.
"""

import json
import re
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from PIL.ExifTags import TAGS, GPSTAGS
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False


# Uganda bounding box (generous margins)
_LAT_MIN, _LAT_MAX = -1.5,  4.3
_LON_MIN, _LON_MAX = 29.5, 35.1

_HEADER_HEIGHT = 200     # px of Mobicap blue header to crop for OCR
                         # (Lat/Lon appear on line 4 of the header at ~120-160px)
_CACHE_SUFFIX  = ".gps.json"


# ---------------------------------------------------------------------------
# EXIF GPS (smartphones and standard cameras)
# ---------------------------------------------------------------------------

def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    try:
        if hasattr(dms[0], 'numerator'):           # IFDRational objects
            d = float(dms[0])
            m = float(dms[1])
            s = float(dms[2])
        else:
            d = dms[0][0] / dms[0][1]
            m = dms[1][0] / dms[1][1]
            s = dms[2][0] / dms[2][1]
        dec = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def _extract_exif_gps(image_path: Path) -> tuple[float | None, float | None]:
    """Read GPS from EXIF. Returns (lat, lon) or (None, None)."""
    if not _PIL_OK:
        return None, None
    try:
        img = Image.open(image_path)
        return _extract_exif_gps_img(img)
    except Exception:
        pass
    return None, None


def _extract_exif_gps_img(img) -> tuple[float | None, float | None]:
    """Read GPS from EXIF of an already-loaded PIL image."""
    if not _PIL_OK:
        return None, None
    try:
        exif = img._getexif()
        if not exif:
            return None, None
        gps_raw = {}
        for tag_id, val in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for k, v in val.items():
                    gps_raw[GPSTAGS.get(k, k)] = v
        if not gps_raw:
            return None, None
        lat = _dms_to_decimal(gps_raw.get("GPSLatitude"),  gps_raw.get("GPSLatitudeRef",  "N"))
        lon = _dms_to_decimal(gps_raw.get("GPSLongitude"), gps_raw.get("GPSLongitudeRef", "E"))
        if lat is not None and lon is not None:
            if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
                return lat, lon
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# OCR GPS (Mobicap images with blue header overlay)
# ---------------------------------------------------------------------------

def _fix_decimal(raw: str, min_val: float, max_val: float) -> float | None:
    """
    Insert a missing decimal point into an OCR-mangled coordinate string.
    e.g. '0445625068' → 0.445625068 when min_val=0, max_val=4.3
    """
    raw = raw.replace(" ", "").replace(",", ".")
    try:
        v = float(raw)
        if min_val <= v <= max_val:
            return v
    except ValueError:
        pass
    digits = raw.replace(".", "")
    for i in range(1, len(digits)):
        candidate = digits[:i] + "." + digits[i:]
        try:
            v = float(candidate)
            if min_val <= v <= max_val:
                return v
        except ValueError:
            continue
    return None


def _has_blue_header_img(img) -> bool:
    """Check if a PIL image has a Mobicap dark-blue header (top 20px).
    Returns True if >30% of top-20px pixels are dark-blue (R<120, G<120, B>120).
    """
    w, h  = img.size
    strip = np.array(img.crop((0, 0, w, min(20, h))), dtype=np.uint8)
    r, g, b = strip[:, :, 0], strip[:, :, 1], strip[:, :, 2]
    return float(np.mean((r < 120) & (g < 120) & (b > 120))) > 0.30


def _extract_ocr_gps_img(img) -> tuple[float | None, float | None]:
    """OCR the header of an already-loaded PIL image. No upscaling — source
    images are 3208px wide which is sufficient for tesseract."""
    if not _TESSERACT_OK:
        return None, None
    try:
        w, h  = img.size
        crop  = img.crop((0, 0, w, min(_HEADER_HEIGHT, h)))
        text  = pytesseract.image_to_string(crop, config="--psm 6")
        lat_m = re.search(r"Lat[:\s]+([0-9.,\s]+)", text, re.IGNORECASE)
        lon_m = re.search(r"Lon[:\s]+([0-9.,\s]+)", text, re.IGNORECASE)
        lat   = _fix_decimal(lat_m.group(1).strip().split()[0], _LAT_MIN, _LAT_MAX) if lat_m else None
        lon   = _fix_decimal(lon_m.group(1).strip().split()[0], _LON_MIN, _LON_MAX) if lon_m else None
        return lat, lon
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Filename metadata (road code, link, side, sequence)
# ---------------------------------------------------------------------------
#
# Real filename patterns observed:
#   A001N1_LINK01-PAVE-0-00001.jpg
#   A001_LINK06_RHS-PAVE-0-00001.jpg
#   A001_LINK08(RHS)-PAVE-0-00001.jpg
#   A001_LINK11-_RHS-PAVE-0-00001.jpg
#   LUGAZI-NJERU-PAVE-0-00001.jpg   (2021-22, no road code)

_FNAME_ROAD = re.compile(
    r"^(?P<road>[A-Z]\d+(?:N\d+)?)_?(?:LINK|Link)(?P<link>\d*)",
    re.IGNORECASE,
)
_FNAME_SIDE = re.compile(r"[_(](?P<side>RHS|LHS)", re.IGNORECASE)
_FNAME_SEQ  = re.compile(r"-PAVE-\d+-(?P<seq>\d+)$", re.IGNORECASE)


def parse_filename(fname: str) -> dict:
    """Extract road, link, side, seq from a Mobicap filename."""
    stem = Path(fname).stem
    m = _FNAME_ROAD.match(stem)
    if not m:
        return {"road": None, "link": None, "side": None, "seq": None}
    side_m = _FNAME_SIDE.search(stem)
    seq_m  = _FNAME_SEQ.search(stem)
    return {
        "road": m.group("road").upper(),
        "link": m.group("link") or None,
        "side": side_m.group("side").upper() if side_m else None,
        "seq":  seq_m.group("seq") if seq_m else None,
    }


# ---------------------------------------------------------------------------
# Unified public interface
# ---------------------------------------------------------------------------

def get_gps(
    image_path:  str | Path,
    use_cache:   bool = True,
) -> tuple[float | None, float | None, str | None]:
    """
    Extract GPS coordinates from any pavement image.

    Strategy
    --------
    1. Try EXIF GPS  (smartphone / standard camera)
    2. Try Mobicap OCR header
    3. Return (None, None, None) on failure

    Parameters
    ----------
    image_path : path to .jpg / .jpeg image
    use_cache  : read/write .gps.json sidecar to avoid re-processing

    Returns
    -------
    (lat, lon, source)  where source ∈ {'exif', 'ocr', None}
    """
    p          = Path(image_path)
    cache_file = p.with_suffix(_CACHE_SUFFIX)

    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return cached.get("lat"), cached.get("lon"), cached.get("source")
        except Exception:
            pass

    # Open image once for all extraction methods
    try:
        img = Image.open(p)
        img.load()
    except Exception:
        return None, None, None

    # 1. EXIF
    lat, lon = _extract_exif_gps_img(img)
    source   = "exif" if (lat is not None and lon is not None) else None

    # 2. OCR fallback (Mobicap blue-header images only)
    if source is None and _has_blue_header_img(img):
        lat, lon = _extract_ocr_gps_img(img)
        source   = "ocr" if (lat is not None and lon is not None) else None

    if use_cache:
        try:
            cache_file.write_text(json.dumps({"lat": lat, "lon": lon, "source": source}))
        except Exception:
            pass

    return lat, lon, source


# ---------------------------------------------------------------------------
# Batch extraction (backward-compatible: returns (lat, lon) dict for build_dataset)
# ---------------------------------------------------------------------------

def _mp_get_gps(args: tuple) -> tuple:
    """Top-level picklable worker for multiprocessing pool."""
    p, use_cache = args
    lat, lon, src = get_gps(p, use_cache=use_cache)
    return str(p), lat, lon, src


def extract_gps_batch(
    image_paths: list,
    use_cache:   bool = True,
    verbose:     bool = True,
    workers:     int  = 4,
) -> dict:
    """
    Extract GPS from a list of images using a spawn-based process pool.

    Uses multiprocessing.get_context('spawn') so each worker has a clean
    process — no inherited locks that cause tesseract/PIL deadlocks.
    Cache files (.gps.json sidecars) make repeat runs near-instant.

    Returns dict mapping str(path) → (lat, lon)  [source dropped for backward compat].
    """
    total    = len(image_paths)
    results  = {}
    n_exif   = 0
    n_ocr    = 0
    n_failed = 0

    args = [(p, use_cache) for p in image_paths]
    ctx  = get_context("spawn")

    with ctx.Pool(workers) as pool:
        for i, (path_str, lat, lon, src) in enumerate(
            pool.imap_unordered(_mp_get_gps, args, chunksize=8), 1
        ):
            results[path_str] = (lat, lon)
            if src == "exif":   n_exif   += 1
            elif src == "ocr":  n_ocr    += 1
            else:               n_failed += 1

            if verbose and i % 500 == 0:
                print(f"  GPS progress: {i}/{total}  "
                      f"(exif={n_exif}, ocr={n_ocr}, failed={n_failed})",
                      flush=True)

    if verbose:
        print(f"GPS results: {n_exif} EXIF  |  {n_ocr} OCR  |  {n_failed} failed  (total {total})",
              flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    lat, lon, src = get_gps(path, use_cache=False)
    print(f"Image  : {path}")
    print(f"GPS    : lat={lat}, lon={lon}")
    print(f"Source : {src}")
    print(f"Filename meta: {parse_filename(Path(path).name)}")
