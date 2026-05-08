/**
 * api.js — thin wrapper around the VCI Estimator FastAPI backend.
 * All functions return the parsed JSON response or throw on error.
 */

const API_BASE = "http://localhost:8000";

const _API_KEY = "dev-key-change-in-production";

/**
 * POST /predict — single segment, 1+ images.
 * @param {File[]} files
 * @param {number|null} lat
 * @param {number|null} lon
 * @returns {Promise<object>} PredictResponse
 */
async function apiPredict(files, lat = null, lon = null) {
  const form = new FormData();
  for (const f of files) form.append("files", f);

  const resp = await fetch(`${API_BASE}/predict`, {
    method:  "POST",
    headers: { "X-API-Key": _API_KEY },
    body:    form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * POST /predict-batch — ZIP archive.
 * @param {File} zipFile
 * @returns {Promise<object>} BatchPredictResponse
 */
async function apiBatch(zipFile) {
  const form = new FormData();
  form.append("archive", zipFile);

  const resp = await fetch(`${API_BASE}/predict-batch`, {
    method:  "POST",
    headers: { "X-API-Key": _API_KEY },
    body:    form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * GET /health — server liveness check.
 * @returns {Promise<{status: string, model_ready: boolean}>}
 */
async function apiHealth() {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error(`Server unreachable (${resp.status})`);
  return resp.json();
}

/**
 * GET /nearest-segment
 * @param {number} lat
 * @param {number} lon
 * @param {string|null} roadCode
 * @returns {Promise<object|null>}
 */
async function apiNearestSegment(lat, lon, roadCode = null) {
  let url = `${API_BASE}/nearest-segment?lat=${lat}&lon=${lon}`;
  if (roadCode) url += `&road_code=${encodeURIComponent(roadCode)}`;
  const resp = await fetch(url, { headers: { "X-API-Key": _API_KEY } });
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
