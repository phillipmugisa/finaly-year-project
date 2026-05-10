/**
 * field.js — field capture mode logic.
 */

let _selectedFiles = [];
let _sessionRows   = [];
let _lastResult    = null;
let _inputMode     = "upload";   // "upload" | "camera"
let _defectChart   = null;       // Chart.js radar instance

// ── Tab switch ──────────────────────────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`panel-${tab}`).classList.remove("hidden");
  document.getElementById(`tab-${tab}`).classList.add("active");
  if (tab === "batch") initBatchMap();
  else if (tab !== "monitor") initFieldMap();
  // Stop camera when leaving the field tab
  if (tab !== "field" && typeof stopCamera === "function") stopCamera();
  // Start/stop monitor polling
  if (tab === "monitor") { if (typeof initMonitor === "function") initMonitor(); }
  else { if (typeof stopMonitor === "function") stopMonitor(); }
  // Init visualizer
  if (tab === "visualizer") { if (typeof initVisualizer === "function") initVisualizer(); }
  else { if (typeof stopVizAnimation === "function") stopVizAnimation(); }
}

// ── Input mode (upload / camera) ─────────────────────────────────────────────

function setInputMode(mode) {
  _inputMode = mode;
  document.getElementById("input-upload").classList.toggle("hidden", mode !== "upload");
  document.getElementById("input-camera").classList.toggle("hidden", mode !== "camera");
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`mode-btn-${mode}`).classList.add("active");

  if (mode === "camera") {
    if (typeof initCamera === "function") initCamera();
  } else {
    if (typeof stopCamera === "function") stopCamera();
  }
  updatePredictBtn();
}

function updatePredictBtn() {
  const btn = document.getElementById("predict-btn");
  if (!btn) return;
  if (_inputMode === "camera") {
    const n = (typeof _capturedBlobs !== "undefined") ? _capturedBlobs.length : 0;
    btn.disabled    = n === 0;
    btn.textContent = n > 1 ? `Analyse ${n} Images` : "Analyse Pavement";
  } else {
    btn.disabled    = false;
    btn.textContent = "Analyse Pavement";
  }
}

// ── File handling ────────────────────────────────────────────────────────────

function handleFiles(files) {
  _selectedFiles = Array.from(files);
  renderThumbs();
}

function handleDrop(e) {
  e.preventDefault();
  _selectedFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
  renderThumbs();
}

function renderThumbs() {
  const list = document.getElementById("thumb-list");
  list.innerHTML = "";
  _selectedFiles.forEach(f => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.className = "thumb";
    img.title = f.name;
    list.appendChild(img);
  });
}

// ── GPS ──────────────────────────────────────────────────────────────────────

function useDeviceGPS() {
  if (!navigator.geolocation) { alert("Geolocation not supported."); return; }
  navigator.geolocation.getCurrentPosition(
    pos => {
      document.getElementById("gps-lat").value = pos.coords.latitude.toFixed(6);
      document.getElementById("gps-lon").value = pos.coords.longitude.toFixed(6);
    },
    err => alert("GPS error: " + err.message)
  );
}

// ── Predict ──────────────────────────────────────────────────────────────────

async function runPredict() {
  let files;
  if (_inputMode === "camera") {
    files = (typeof _capturedBlobs !== "undefined")
      ? _capturedBlobs.map((item, i) =>
          new File([item.blob], `field_capture_${String(i + 1).padStart(3, "0")}.jpg`, { type: "image/jpeg" }))
      : [];
    if (files.length === 0) { alert("Capture at least one photo first."); return; }
  } else {
    files = _selectedFiles;
    if (files.length === 0) { alert("Please select at least one image."); return; }
  }

  showState("spinner");

  const lat = parseFloat(document.getElementById("gps-lat").value) || null;
  const lon = parseFloat(document.getElementById("gps-lon").value) || null;

  try {
    const result = await apiPredict(files, lat, lon);
    _lastResult = result;
    renderResults(result);
    showState("content");
  } catch (err) {
    document.getElementById("results-error").textContent = "Error: " + err.message;
    showState("error");
  }
}

function showState(state) {
  ["placeholder", "content", "error", "spinner"].forEach(s => {
    document.getElementById(`results-${s}`).classList.toggle("hidden", s !== state);
  });
}

// ── Render results ────────────────────────────────────────────────────────────

function condClass(label, prefix = "cond") {
  const map = {
    "good":         "good",
    "satisfactory": "satisfactory",
    "fair":         "fair",
    "poor":         "poor",
    "very poor":    "verypoor",
    "serious":      "serious",
    "failed":       "failed",
    "bad":          "bad",
  };
  const key = (label || "").toLowerCase();
  return `${prefix}-${map[key] || "fair"}`;
}

function renderResults(r) {
  // vVCI
  const vvci = r.vvci ?? 0;
  const vvciEl = document.getElementById("vvci-score");
  vvciEl.textContent = vvci.toFixed(1);
  document.getElementById("vvci-label").textContent = r.vvci_label || "—";
  document.getElementById("vvci-box").className =
    `flex-1 rounded-xl p-3 text-center ${condClass(r.vvci_label, "cond-vvci")}`;

  // PCI
  const pciEl = document.getElementById("pci-score");
  if (r.pci != null) {
    pciEl.textContent = r.pci.toFixed(1);
    document.getElementById("pci-label").textContent = r.pci_label || "—";
    document.getElementById("pci-box").className =
      `flex-1 rounded-xl p-3 text-center ${condClass(r.pci_label)}`;
  } else {
    pciEl.textContent = "—";
    document.getElementById("pci-box").className =
      "flex-1 rounded-xl p-3 text-center bg-gray-100 text-gray-400";
  }

  // GPS / segment
  if (r.gps_used) {
    const g = r.gps_used;
    document.getElementById("gps-text").textContent =
      `${g.lat.toFixed(4)}, ${g.lon.toFixed(4)} (${g.source})`;
    if (r.road_matched) {
      const m = r.road_matched;
      document.getElementById("segment-text").textContent =
        `${m.road_name}  km ${m.km_start.toFixed(1)}–${m.km_end.toFixed(1)}`;
    } else {
      document.getElementById("segment-text").textContent = "No segment matched";
    }
    document.getElementById("gps-row").classList.remove("hidden");
    setFieldPreviewMarker(g.lat, g.lon, r);
  } else {
    document.getElementById("gps-row").classList.add("hidden");
  }

  // Defects — probability bars + radar chart
  const defects = r.defects || [];
  renderDefectList(defects);
  renderDefectRadar(defects);

  if (!r.model_ready) {
    document.getElementById("defect-list").innerHTML =
      `<p class="text-yellow-600 text-xs">Model not loaded — train first, then restart the API.</p>`;
  }
}

const _GRADE_LABELS  = ["", "None", "Slight", "Moderate", "Severe", "V.Severe"];
const _GRADE_COLOURS = ["", "#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"];
const _GRADE_BG      = ["", "#dcfce7", "#f7fee7", "#fef9c3", "#ffedd5", "#fee2e2"];

function renderDefectList(defects) {
  const list = document.getElementById("defect-list");
  list.innerHTML = "";
  defects.forEach(d => {
    const g     = d.predicted_grade;
    const probs = d.grade_probs || [];
    const conf  = (d.confidence * 100).toFixed(0);

    // Probability mini-bars (grades 1–5)
    const miniBarHtml = probs.map((p, i) => {
      const grade  = i + 1;
      const height = Math.max(4, Math.round(p * 44));
      const active = grade === g;
      const col    = active ? _GRADE_COLOURS[grade] : "#d1d5db";
      return `
        <div class="flex flex-col items-center gap-0.5" title="Grade ${grade}: ${(p*100).toFixed(1)}%">
          <div style="width:12px;height:${height}px;background:${col};border-radius:2px 2px 0 0;transition:height .4s"></div>
          <span style="font-size:9px;color:${active ? _GRADE_COLOURS[grade] : '#9ca3af'};font-weight:${active?'700':'400'}">${grade}</span>
        </div>`;
    }).join("");

    list.innerHTML += `
      <div class="rounded-xl border p-3" style="border-color:${_GRADE_COLOURS[g]}33;background:${_GRADE_BG[g]}">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold text-gray-700">${formatDefectName(d.name)}</span>
          <span class="text-xs px-2 py-0.5 rounded-full font-semibold"
                style="background:${_GRADE_COLOURS[g]};color:white">
            ${_GRADE_LABELS[g]} · ${g}/5
          </span>
        </div>
        <div class="flex items-end gap-1 h-12 mb-1">
          ${miniBarHtml}
          <div class="ml-auto text-right">
            <div class="text-xs text-gray-400">confidence</div>
            <div class="text-lg font-bold" style="color:${_GRADE_COLOURS[g]}">${conf}%</div>
          </div>
        </div>
      </div>`;
  });
}

function renderDefectRadar(defects) {
  const ctx = document.getElementById("defect-radar");
  if (!ctx || !defects.length) return;

  const labels = defects.map(d => formatDefectName(d.name));
  const grades = defects.map(d => d.predicted_grade);

  // Colour gradient per grade level for fill
  const avgGrade = grades.reduce((a, b) => a + b, 0) / grades.length;
  const fillCol  = avgGrade <= 2 ? "rgba(34,197,94,0.25)"
                 : avgGrade <= 3 ? "rgba(234,179,8,0.25)"
                 : avgGrade <= 4 ? "rgba(249,115,22,0.25)"
                 :                 "rgba(239,68,68,0.25)";
  const lineCol  = avgGrade <= 2 ? "#22c55e"
                 : avgGrade <= 3 ? "#eab308"
                 : avgGrade <= 4 ? "#f97316"
                 :                 "#ef4444";

  if (_defectChart) { _defectChart.destroy(); _defectChart = null; }

  _defectChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "Grade (1–5)",
        data:  grades,
        fill:  true,
        backgroundColor: fillCol,
        borderColor:     lineCol,
        borderWidth:     2,
        pointBackgroundColor: grades.map(g => _GRADE_COLOURS[g]),
        pointRadius: 5,
        pointHoverRadius: 7,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 5,
          ticks: { stepSize: 1, font: { size: 9 },
                   callback: v => v === 0 ? "" : _GRADE_LABELS[v] || v },
          pointLabels: { font: { size: 10, weight: "600" } },
          grid:  { color: "rgba(0,0,0,0.06)" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Grade ${ctx.raw} — ${_GRADE_LABELS[ctx.raw] || ""}`,
          },
        },
      },
    },
  });
}

function formatDefectName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ── Session ───────────────────────────────────────────────────────────────────

function saveToSession() {
  if (!_lastResult) return;
  const r = _lastResult;
  const row = {
    time:     new Date().toLocaleTimeString(),
    images:   r.images_used,
    vvci:     r.vvci?.toFixed(1),
    pci:      r.pci?.toFixed(1) ?? "—",
    cond:     r.vvci_label,
    gps:      r.gps_used ? `${r.gps_used.lat.toFixed(4)}, ${r.gps_used.lon.toFixed(4)}` : "—",
    segment:  r.road_matched
      ? `${r.road_matched.road_name} km${r.road_matched.km_start.toFixed(1)}`
      : "—",
    _raw: r,
  };
  _sessionRows.push(row);
  renderSession();

  // Pin this segment permanently on the map
  if (r.gps_used) {
    addFieldSessionMarker(r.gps_used.lat, r.gps_used.lon, r);
  }
}

function renderSession() {
  const tbody = document.getElementById("session-tbody");
  document.getElementById("session-count").textContent = _sessionRows.length;
  if (_sessionRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="px-3 py-4 text-gray-400 text-center">No segments recorded yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = _sessionRows.map(r => `
    <tr class="hover:bg-gray-50">
      <td class="px-3 py-2">${r.time}</td>
      <td class="px-3 py-2">${r.images}</td>
      <td class="px-3 py-2 font-semibold">${r.vvci}</td>
      <td class="px-3 py-2">${r.pci}</td>
      <td class="px-3 py-2"><span class="px-2 py-0.5 rounded-full text-xs ${condClass(r.cond, "cond-vvci")}">${r.cond}</span></td>
      <td class="px-3 py-2 text-xs">${r.gps}</td>
      <td class="px-3 py-2 text-xs">${r.segment}</td>
    </tr>`).join("");
}

function clearSession() {
  _sessionRows = [];
  renderSession();
  clearFieldMarkers();
}

function exportSession() {
  if (_sessionRows.length === 0) { alert("No session data to export."); return; }

  const headers = ["time","images","vvci","pci","condition","gps","segment"];
  const rows    = _sessionRows.map(r =>
    [r.time, r.images, r.vvci, r.pci, r.cond, r.gps, r.segment]
      .map(v => `"${v}"`).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  downloadBlob(csv, `session_${dateStamp()}.csv`, "text/csv");
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function dateStamp() {
  return new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
}

function downloadBlob(content, filename, mime) {
  const a = document.createElement("a");
  a.href     = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = filename;
  a.click();
}
