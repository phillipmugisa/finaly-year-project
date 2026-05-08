"""
api/main.py
-----------
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000

Or for production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import load_model, is_ready
from .segments  import load_segments
from .routes    import predict, segments as seg_routes

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== VCI Estimator API starting up ===")

    model_ok = load_model()
    if model_ok:
        print("  Model       : loaded")
    else:
        print("  Model       : NOT FOUND — train first, then restart")
        print("                /predict will return model_ready=false until then")

    n_segs = load_segments()
    if n_segs:
        print(f"  Segments    : {n_segs:,} loaded")
    else:
        print("  Segments    : not loaded — /nearest-segment unavailable")

    print("=== Ready ===")
    yield
    # Nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "VCI Estimator API",
    description = (
        "Automated pavement Visual Condition Index (vVCI) estimation for Uganda roads.\n\n"
        "Accepts Mobicap survey images (GPS in blue header overlay) and smartphone "
        "images (GPS in EXIF). Returns predicted vVCI [0–100], per-defect grades, "
        "GPS coordinates used, and the nearest matched survey segment."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

app.include_router(predict.router,     tags=["Prediction"])
app.include_router(seg_routes.router,  tags=["Segments"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":      "ok",
        "model_ready": is_ready(),
    }


# ---------------------------------------------------------------------------
# Serve web app at /  (must come AFTER all API routes)
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(str(_WEB_DIR / "index.html"))

if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR)), name="web")
