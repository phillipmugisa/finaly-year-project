/**
 * map.js — Leaflet maps for field session and batch review.
 *
 * Each map card has two toolbar buttons:
 *   ⤢ Expand  — CSS fullscreen overlay; Escape or button to exit.
 *   ↗ New window — opens a self-contained HTML page in a new tab.
 */

const UGANDA_CENTER = [1.37, 32.29];
const INIT_ZOOM     = 7;

let _fieldMap     = null;
let _sessionLayer = null;
let _previewLayer = null;
let _batchMap     = null;
let _batchLayer   = null;

// Raw result data kept so the "new window" can rebuild the map from scratch
let _fieldPins = [];   // [{lat, lon, result}] — saved session entries only
let _batchPins = [];   // [{lat, lon, result}] — last batch run

let _expandedMap = null;   // "field" | "batch" | null

// ── Field map ─────────────────────────────────────────────────────────────────

function initFieldMap() {
  if (_fieldMap) return;
  _fieldMap = L.map("field-map").setView(UGANDA_CENTER, INIT_ZOOM);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(_fieldMap);
  _sessionLayer = L.layerGroup().addTo(_fieldMap);
  _previewLayer = L.layerGroup().addTo(_fieldMap);
}

// Dashed ring for current (unsaved) analysis — replaced on each new result
function setFieldPreviewMarker(lat, lon, result) {
  if (!_fieldMap) initFieldMap();
  _previewLayer.clearLayers();

  const color = condColor(result.vvci_label);
  L.circle([lat, lon], {
    radius: 500, color, fillColor: color, fillOpacity: 0.08,
    dashArray: "6 4", weight: 2,
  }).addTo(_previewLayer);

  L.circleMarker([lat, lon], {
    radius: 9, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2.5,
  })
    .bindPopup(buildPopup(result), { maxWidth: 290 })
    .openPopup()
    .addTo(_previewLayer);

  _fieldMap.setView([lat, lon], Math.max(_fieldMap.getZoom() || 0, 14));
}

// Permanent marker added when "Save to Session" is clicked
function addFieldSessionMarker(lat, lon, result) {
  if (!_fieldMap) initFieldMap();
  _previewLayer.clearLayers();

  const color = condColor(result.vvci_label);
  L.circle([lat, lon], {
    radius: 500, color, fillColor: color, fillOpacity: 0.18, weight: 2,
  }).addTo(_sessionLayer);
  L.circleMarker([lat, lon], {
    radius: 9, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2.5,
  })
    .bindPopup(buildPopup(result), { maxWidth: 290 })
    .addTo(_sessionLayer);

  _fieldPins.push({ lat, lon, result });
  _fitLayer(_fieldMap, _sessionLayer);
}

function clearFieldMarkers() {
  if (_sessionLayer) _sessionLayer.clearLayers();
  if (_previewLayer) _previewLayer.clearLayers();
  _fieldPins = [];
}

// ── Batch map ─────────────────────────────────────────────────────────────────

function initBatchMap() {
  if (_batchMap) return;
  _batchMap   = L.map("batch-map").setView(UGANDA_CENTER, INIT_ZOOM);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(_batchMap);
  _batchLayer = L.layerGroup().addTo(_batchMap);
}

function updateBatchMapPins(pins) {
  if (!_batchMap) initBatchMap();
  _batchLayer.clearLayers();
  _batchPins = pins;
  if (!pins.length) return;

  _drawPins(_batchLayer, pins, "batch");

  const bounds = pins.map(p => [p.lat, p.lon]);
  _batchMap.fitBounds(L.latLngBounds(bounds), { padding: [30, 30] });
}

// ── Shared pin renderer (used by both live map and new-window page) ───────────

function _drawPins(layer, pins, type) {
  if (type === "batch") {
    // Group by road and draw coloured polylines
    const roads = {};
    pins.forEach(p => {
      const code = p.result?.road_matched?.road_code ?? "__ungrouped__";
      if (!roads[code]) roads[code] = [];
      roads[code].push(p);
    });

    Object.values(roads).forEach(group => {
      const sorted = [...group].sort(
        (a, b) => (a.result?.road_matched?.km_start ?? 0) - (b.result?.road_matched?.km_start ?? 0)
      );
      if (sorted.length > 1) {
        for (let i = 0; i < sorted.length - 1; i++) {
          const color = condColor(sorted[i].result?.vvci_label);
          L.polyline(
            [[sorted[i].lat, sorted[i].lon], [sorted[i + 1].lat, sorted[i + 1].lon]],
            { color, weight: 6, opacity: 0.85, lineCap: "round" }
          ).addTo(layer);
        }
      }
      group.forEach(({ lat, lon, result }) => {
        const color = condColor(result.vvci_label);
        L.circleMarker([lat, lon], {
          radius: 7, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2,
        })
          .bindPopup(buildPopup(result), { maxWidth: 290 })
          .addTo(layer);
      });
    });
  } else {
    pins.forEach(({ lat, lon, result }) => {
      const color = condColor(result.vvci_label);
      L.circle([lat, lon], {
        radius: 500, color, fillColor: color, fillOpacity: 0.18, weight: 2,
      }).addTo(layer);
      L.circleMarker([lat, lon], {
        radius: 9, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2.5,
      })
        .bindPopup(buildPopup(result), { maxWidth: 290 })
        .addTo(layer);
    });
  }
}

function _fitLayer(map, layer) {
  const pts = [];
  layer.eachLayer(l => { if (l.getLatLng) pts.push(l.getLatLng()); });
  if (pts.length > 1) map.fitBounds(L.latLngBounds(pts), { padding: [30, 30] });
}

// ── Fullscreen toggle ─────────────────────────────────────────────────────────

function toggleMapFullscreen(type) {
  const divId  = type === "field" ? "field-map" : "batch-map";
  const mapObj = type === "field" ? _fieldMap   : _batchMap;
  const el     = document.getElementById(divId);
  if (!el) return;

  const expanding = !el.classList.contains("map-fullscreen");
  el.classList.toggle("map-fullscreen", expanding);
  _expandedMap = expanding ? type : null;

  // Exit button injected into the map div
  const exitId  = `${divId}-exit-btn`;
  let   exitBtn = document.getElementById(exitId);
  if (expanding) {
    if (!exitBtn) {
      exitBtn = document.createElement("button");
      exitBtn.id        = exitId;
      exitBtn.className = "map-exit-btn";
      exitBtn.textContent = "✕  Exit fullscreen";
      exitBtn.onclick   = () => toggleMapFullscreen(type);
      el.appendChild(exitBtn);
    }
    exitBtn.style.display = "block";
    document.addEventListener("keydown", _onEscapeFullscreen);
  } else {
    if (exitBtn) exitBtn.style.display = "none";
    document.removeEventListener("keydown", _onEscapeFullscreen);
  }

  // Update toolbar button label
  const btn = document.getElementById(`${type}-map-expand-btn`);
  if (btn) btn.textContent = expanding ? "⊠  Exit" : "⤢  Expand";

  setTimeout(() => { if (mapObj) mapObj.invalidateSize(); }, 60);
}

function _onEscapeFullscreen(e) {
  if (e.key === "Escape" && _expandedMap) toggleMapFullscreen(_expandedMap);
}

// ── Open in new window ────────────────────────────────────────────────────────

function openMapWindow(type) {
  const pins  = type === "field" ? _fieldPins : _batchPins;
  const mapObj = type === "field" ? _fieldMap : _batchMap;
  const center = mapObj ? mapObj.getCenter() : { lat: UGANDA_CENTER[0], lng: UGANDA_CENTER[1] };
  const zoom   = mapObj ? mapObj.getZoom()   : INIT_ZOOM;
  const title  = type === "field" ? "Session Map" : "Batch Results Map";

  if (!pins.length) {
    alert("No map data to show yet.");
    return;
  }

  const html = _buildMapPageHtml(title, pins, center, zoom, type);
  const blob = new Blob([html], { type: "text/html" });
  const url  = URL.createObjectURL(blob);
  window.open(url, "_blank");
  // Revoke after the new tab has had time to load
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function _buildMapPageHtml(title, pins, center, zoom, type) {
  // Embed the helper functions verbatim so the new page is self-contained
  const helpers = `
${condColor.toString()}
${buildPopup.toString()}
`;

  const setup = `
const PINS   = ${JSON.stringify(pins)};
const TYPE   = ${JSON.stringify(type)};
const CENTER = [${center.lat}, ${center.lng}];
const ZOOM   = ${zoom};

${helpers}

const map = L.map("map").setView(CENTER, ZOOM);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors", maxZoom: 19
}).addTo(map);

const layer = L.layerGroup().addTo(map);
_drawPins(layer, PINS, TYPE);

function _drawPins(layer, pins, type) {
  if (type === "batch") {
    const roads = {};
    pins.forEach(p => {
      const code = p.result?.road_matched?.road_code ?? "__ungrouped__";
      if (!roads[code]) roads[code] = [];
      roads[code].push(p);
    });
    Object.values(roads).forEach(group => {
      const sorted = [...group].sort(
        (a, b) => (a.result?.road_matched?.km_start ?? 0) - (b.result?.road_matched?.km_start ?? 0)
      );
      if (sorted.length > 1) {
        for (let i = 0; i < sorted.length - 1; i++) {
          const color = condColor(sorted[i].result?.vvci_label);
          L.polyline([[sorted[i].lat, sorted[i].lon],[sorted[i+1].lat, sorted[i+1].lon]],
            { color, weight: 6, opacity: 0.85, lineCap: "round" }).addTo(layer);
        }
      }
      group.forEach(({ lat, lon, result }) => {
        const color = condColor(result.vvci_label);
        L.circleMarker([lat, lon], { radius: 7, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2 })
          .bindPopup(buildPopup(result), { maxWidth: 290 }).addTo(layer);
      });
    });
  } else {
    pins.forEach(({ lat, lon, result }) => {
      const color = condColor(result.vvci_label);
      L.circle([lat, lon], { radius: 500, color, fillColor: color, fillOpacity: 0.18, weight: 2 }).addTo(layer);
      L.circleMarker([lat, lon], { radius: 9, color: "#fff", fillColor: color, fillOpacity: 1, weight: 2.5 })
        .bindPopup(buildPopup(result), { maxWidth: 290 }).addTo(layer);
    });
  }
}

if (PINS.length > 1) {
  map.fitBounds(L.latLngBounds(PINS.map(p => [p.lat, p.lon])), { padding: [30, 30] });
}
`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>${title} — Uganda Road VCI Estimator</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\/script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; }
    #header {
      background: #14532d; color: #fff; padding: 10px 16px;
      display: flex; align-items: center; justify-content: space-between;
      font-size: 14px; font-weight: 600;
    }
    #header small { font-size: 11px; opacity: .65; font-weight: 400; }
    #map { width: 100%; height: calc(100vh - 40px); }
  </style>
</head>
<body>
  <div id="header">
    ${title}
    <small>Uganda Road VCI Estimator &nbsp;·&nbsp; Mugisa Phillip &amp; Ater Maluac Ater</small>
  </div>
  <div id="map"></div>
  <script>${setup}<\/script>
</body>
</html>`;
}

// ── Popup builder ─────────────────────────────────────────────────────────────

function buildPopup(r) {
  const condStyle = {
    good:         { bg: "#d1fae5", text: "#065f46" },
    satisfactory: { bg: "#dcfce7", text: "#166534" },
    fair:         { bg: "#fef9c3", text: "#854d0e" },
    poor:         { bg: "#fed7aa", text: "#9a3412" },
    "very poor":  { bg: "#fecaca", text: "#991b1b" },
    bad:          { bg: "#fecaca", text: "#991b1b" },
    serious:      { bg: "#fca5a5", text: "#7f1d1d" },
    failed:       { bg: "#ef4444", text: "#ffffff" },
  };
  const gradeColor = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"];

  const vs = condStyle[(r.vvci_label || "").toLowerCase()] || { bg: "#f3f4f6", text: "#374151" };
  const ps = condStyle[(r.pci_label  || "").toLowerCase()] || { bg: "#f3f4f6", text: "#374151" };

  const segLine = r.road_matched
    ? `${r.road_matched.road_name} &nbsp; km ${(r.road_matched.km_start ?? 0).toFixed(1)} – ${(r.road_matched.km_end ?? 0).toFixed(1)}`
    : "No segment matched";

  const defectRows = (r.defects || []).map(d => {
    const pct   = ((d.predicted_grade - 1) / 4) * 100;
    const color = gradeColor[d.predicted_grade - 1] || "#6b7280";
    const name  = (d.name || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `<div style="display:flex;align-items:center;gap:5px;margin:3px 0">
      <span style="width:90px;font-size:10px;color:#4b5563;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            title="${name}">${name}</span>
      <div style="flex:1;background:#f3f4f6;height:6px;border-radius:3px">
        <div style="width:${pct}%;height:6px;border-radius:3px;background:${color}"></div>
      </div>
      <span style="font-size:10px;font-weight:700;color:#374151;width:24px;text-align:right">
        ${d.predicted_grade}/5
      </span>
    </div>`;
  }).join("");

  const gpsLine = r.gps_used
    ? `<div style="font-size:10px;color:#9ca3af;margin-top:6px">
         \u{1F4CD} ${r.gps_used.lat.toFixed(5)}, ${r.gps_used.lon.toFixed(5)}
         <span style="opacity:.7">(${r.gps_used.source})</span>
       </div>`
    : "";

  return `<div style="min-width:230px;font-family:system-ui,sans-serif;line-height:1.4">
    <div style="font-size:12px;font-weight:700;color:#111827;margin-bottom:6px">${segLine}</div>
    <div style="display:flex;gap:6px;margin-bottom:8px">
      <div style="flex:1;background:${vs.bg};color:${vs.text};padding:5px 7px;border-radius:8px;text-align:center">
        <div style="font-size:10px;opacity:.75">Visual VCI</div>
        <div style="font-size:18px;font-weight:700;line-height:1.1">${(r.vvci ?? 0).toFixed(1)}</div>
        <div style="font-size:11px;font-weight:600">${r.vvci_label || "—"}</div>
      </div>
      <div style="flex:1;background:${ps.bg};color:${ps.text};padding:5px 7px;border-radius:8px;text-align:center">
        <div style="font-size:10px;opacity:.75">Est. PCI</div>
        <div style="font-size:18px;font-weight:700;line-height:1.1">${r.pci != null ? r.pci.toFixed(1) : "—"}</div>
        <div style="font-size:11px;font-weight:600">${r.pci_label || "—"}</div>
      </div>
    </div>
    ${defectRows
      ? `<div style="font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Defect Grades</div>
         ${defectRows}`
      : ""}
    ${gpsLine}
  </div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function condColor(label) {
  const map = {
    good:         "#22c55e",
    satisfactory: "#84cc16",
    fair:         "#eab308",
    poor:         "#f97316",
    "very poor":  "#ef4444",
    bad:          "#ef4444",
    serious:      "#dc2626",
    failed:       "#991b1b",
  };
  return map[(label || "").toLowerCase()] || "#6b7280";
}
