/**
 * camera.js — camera hardware logic for the Field Survey tab.
 *
 * Captures multiple images over a 1 km section. Each capture appends to
 * _capturedBlobs. field.js reads _capturedBlobs when "Analyse" is clicked
 * and sends all images together so the model can average feature vectors.
 */

let _cameraStream  = null;
let _capturedBlobs = [];   // [{blob, url}, ...]
let _cameraGps     = { lat: null, lon: null };

// ── Camera lifecycle ─────────────────────────────────────────────────────────

async function startCamera() {
  const video  = document.getElementById("cam-video");
  const status = document.getElementById("cam-status");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status.textContent = "Camera API not available — use the file picker below.";
    document.getElementById("cam-fallback").classList.remove("hidden");
    return;
  }

  try {
    _cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } },
    });
    video.srcObject = _cameraStream;
    document.getElementById("cam-preview-wrap").classList.remove("hidden");
    document.getElementById("cam-fallback").classList.add("hidden");
    status.textContent = "Camera active — point at the road surface.";
    status.className   = "text-xs text-green-600";
    fetchCameraGps();
  } catch (err) {
    status.textContent = `Camera access denied (${err.message}). Use the file picker below.`;
    status.className   = "text-xs text-red-500";
    document.getElementById("cam-fallback").classList.remove("hidden");
  }
}

function stopCamera() {
  if (_cameraStream) {
    _cameraStream.getTracks().forEach(t => t.stop());
    _cameraStream = null;
  }
  const video = document.getElementById("cam-video");
  if (video) video.srcObject = null;
}

function initCamera() {
  if (!_cameraStream) startCamera();
}

// ── GPS ───────────────────────────────────────────────────────────────────────

function fetchCameraGps() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    pos => {
      _cameraGps.lat = pos.coords.latitude;
      _cameraGps.lon = pos.coords.longitude;

      // Auto-fill the shared GPS inputs if still empty
      const latEl = document.getElementById("gps-lat");
      const lonEl = document.getElementById("gps-lon");
      if (latEl && !latEl.value) latEl.value = _cameraGps.lat.toFixed(6);
      if (lonEl && !lonEl.value) lonEl.value = _cameraGps.lon.toFixed(6);

      const el = document.getElementById("cam-gps-display");
      if (el) {
        el.textContent = `GPS: ${_cameraGps.lat.toFixed(5)}N, ${_cameraGps.lon.toFixed(5)}E  ✓`;
        el.className   = "text-xs text-green-600 mt-0.5 font-mono";
      }
    },
    () => {
      const el = document.getElementById("cam-gps-display");
      if (el) {
        el.textContent = "GPS not available — coordinates will not be embedded.";
        el.className   = "text-xs text-yellow-600 mt-0.5";
      }
    },
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

// ── Capture ───────────────────────────────────────────────────────────────────

function capturePhoto() {
  const video  = document.getElementById("cam-video");
  const canvas = document.getElementById("cam-canvas");

  if (!video || !video.srcObject) { alert("Camera not active."); return; }

  // Use shared GPS inputs, fall back to auto-detected GPS
  const latIn = parseFloat(document.getElementById("gps-lat")?.value);
  const lonIn = parseFloat(document.getElementById("gps-lon")?.value);
  const lat   = !isNaN(latIn) ? latIn : _cameraGps.lat;
  const lon   = !isNaN(lonIn) ? lonIn : _cameraGps.lon;

  const vw = video.videoWidth  || 1280;
  const vh = video.videoHeight || 960;

  const headerH = Math.max(60, Math.round(vh * 0.083));
  canvas.width  = vw;
  canvas.height = vh + headerH;

  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#003087";
  ctx.fillRect(0, 0, vw, headerH);

  const fontSize  = Math.max(12, Math.round(headerH * 0.32));
  const fontSize2 = Math.max(11, Math.round(headerH * 0.28));
  ctx.fillStyle = "#FFFFFF";
  ctx.font      = `bold ${fontSize}px monospace`;

  const gpsText = (lat !== null && lon !== null)
    ? `LAT:${lat.toFixed(6)}  LON:${lon.toFixed(6)}`
    : "LAT:-------  LON:-------";
  ctx.fillText("FIELD SURVEY", 12, headerH * 0.45);
  ctx.font = `${fontSize}px monospace`;
  ctx.fillText(gpsText, 12, headerH * 0.80);

  const now = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  ctx.font      = `${fontSize2}px monospace`;
  ctx.textAlign = "right";
  ctx.fillText(now, vw - 12, headerH * 0.55);
  ctx.textAlign = "left";

  ctx.drawImage(video, 0, headerH, vw, vh);

  canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    _capturedBlobs.push({ blob, url });

    // Fill shared GPS inputs on first capture if still blank
    if (_capturedBlobs.length === 1) {
      const latEl = document.getElementById("gps-lat");
      const lonEl = document.getElementById("gps-lon");
      if (latEl && !latEl.value && lat !== null) latEl.value = lat.toFixed(6);
      if (lonEl && !lonEl.value && lon !== null) lonEl.value = lon.toFixed(6);
    }

    _renderCamThumbs();
    _updateCamButtons();
  }, "image/jpeg", 0.92);
}

// Fallback: file input (iOS / no camera permission)
function handleCameraFileInput(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];

  const lat = parseFloat(document.getElementById("gps-lat")?.value) || _cameraGps.lat;
  const lon = parseFloat(document.getElementById("gps-lon")?.value) || _cameraGps.lon;

  const img    = new Image();
  const reader = new FileReader();
  reader.onload = e => {
    img.onload = () => {
      const canvas  = document.getElementById("cam-canvas");
      const headerH = Math.max(60, Math.round(img.height * 0.083));
      canvas.width  = img.width;
      canvas.height = img.height + headerH;

      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#003087";
      ctx.fillRect(0, 0, img.width, headerH);

      const fontSize = Math.max(12, Math.round(headerH * 0.32));
      ctx.fillStyle  = "#FFFFFF";
      ctx.font       = `bold ${fontSize}px monospace`;
      const gpsText  = (lat && lon)
        ? `LAT:${lat.toFixed(6)}  LON:${lon.toFixed(6)}`
        : "LAT:-------  LON:-------";
      ctx.fillText("FIELD SURVEY", 12, headerH * 0.45);
      ctx.font = `${fontSize}px monospace`;
      ctx.fillText(gpsText, 12, headerH * 0.80);
      const now = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
      ctx.font = `${Math.max(11, Math.round(headerH * 0.28))}px monospace`;
      ctx.textAlign = "right";
      ctx.fillText(now, img.width - 12, headerH * 0.55);
      ctx.textAlign = "left";
      ctx.drawImage(img, 0, headerH);

      canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        _capturedBlobs.push({ blob, url });
        _renderCamThumbs();
        _updateCamButtons();
      }, "image/jpeg", 0.92);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ── Thumbnail management ──────────────────────────────────────────────────────

function _renderCamThumbs() {
  const wrap  = document.getElementById("cam-captured-wrap");
  const list  = document.getElementById("cam-thumb-list");
  const label = document.getElementById("cam-capture-count");

  if (_capturedBlobs.length === 0) {
    if (wrap) wrap.classList.add("hidden");
    return;
  }

  if (wrap) wrap.classList.remove("hidden");
  if (label) {
    const n = _capturedBlobs.length;
    label.textContent = `${n} image${n === 1 ? "" : "s"} captured — ready to analyse`;
  }
  if (list) {
    list.innerHTML = _capturedBlobs.map((item, i) => `
      <div class="relative inline-block flex-shrink-0">
        <img src="${item.url}" class="thumb" title="Image ${i + 1}" />
        <button onclick="removeCamCapture(${i})" class="cam-thumb-remove" title="Remove">×</button>
      </div>`).join("");
  }
}

function removeCamCapture(i) {
  URL.revokeObjectURL(_capturedBlobs[i].url);
  _capturedBlobs.splice(i, 1);
  _renderCamThumbs();
  _updateCamButtons();
}

function clearCamCaptures() {
  _capturedBlobs.forEach(item => URL.revokeObjectURL(item.url));
  _capturedBlobs = [];
  _renderCamThumbs();
  _updateCamButtons();
}

function _updateCamButtons() {
  const n          = _capturedBlobs.length;
  const captureBtn = document.getElementById("cam-capture-btn");
  const clearBtn   = document.getElementById("cam-clear-btn");

  if (captureBtn) {
    captureBtn.textContent = n === 0 ? "📷 Capture" : `📷 Capture (${n})`;
  }
  if (clearBtn) {
    clearBtn.classList.toggle("hidden", n === 0);
  }

  // Sync the shared Analyse button (only when camera mode is active)
  if (typeof updatePredictBtn === "function") updatePredictBtn();
}
