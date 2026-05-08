/**
 * batch.js — batch review mode (ministry use).
 */

let _batchFile    = null;
let _batchResults = [];

// ── File handling ─────────────────────────────────────────────────────────────

function handleBatchDrop(e) {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith(".zip")) setBatchFile(f);
  else alert("Please drop a ZIP file.");
}

function handleBatchFile(f) {
  if (f) setBatchFile(f);
}

function setBatchFile(f) {
  _batchFile = f;
  document.getElementById("batch-filename").textContent = `Selected: ${f.name}  (${(f.size / 1024 / 1024).toFixed(1)} MB)`;
}

// ── Run batch ─────────────────────────────────────────────────────────────────

async function runBatch() {
  if (!_batchFile) { alert("Please select a ZIP file first."); return; }

  document.getElementById("batch-results").classList.add("hidden");
  document.getElementById("batch-error").classList.add("hidden");
  document.getElementById("batch-spinner").classList.remove("hidden");
  document.getElementById("batch-btn").disabled = true;
  document.getElementById("batch-progress").textContent = "Uploading…";

  try {
    const result = await apiBatch(_batchFile);
    _batchResults = result.results;
    renderBatchResults(result);
    document.getElementById("batch-results").classList.remove("hidden");
  } catch (err) {
    const errEl = document.getElementById("batch-error");
    errEl.textContent = "Error: " + err.message;
    errEl.classList.remove("hidden");
  } finally {
    document.getElementById("batch-spinner").classList.add("hidden");
    document.getElementById("batch-btn").disabled = false;
  }
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderBatchResults(data) {
  document.getElementById("batch-total").textContent = data.total;

  // Summary stats
  const ok     = data.results.filter(r => !r.error);
  const errors = data.results.filter(r =>  r.error);
  const avgVvci = ok.length ? (ok.reduce((s, r) => s + (r.vvci || 0), 0) / ok.length).toFixed(1) : "—";
  const avgPci  = ok.length ? (ok.reduce((s, r) => s + (r.pci  || 0), 0) / ok.length).toFixed(1) : "—";

  document.getElementById("batch-summary").innerHTML = `
    <div class="bg-green-50 rounded-xl p-3 text-center">
      <div class="text-2xl font-bold text-green-700">${avgVvci}</div>
      <div class="text-xs text-gray-500">Avg vVCI</div>
    </div>
    <div class="bg-blue-50 rounded-xl p-3 text-center">
      <div class="text-2xl font-bold text-blue-700">${avgPci}</div>
      <div class="text-xs text-gray-500">Avg PCI (est.)</div>
    </div>
    <div class="bg-gray-50 rounded-xl p-3 text-center">
      <div class="text-2xl font-bold text-gray-700">${errors.length}</div>
      <div class="text-xs text-gray-500">Errors</div>
    </div>`;

  // Table rows
  const tbody = document.getElementById("batch-tbody");
  tbody.innerHTML = data.results.map(r => {
    if (r.error) {
      return `<tr class="bg-red-50">
        <td class="px-3 py-2 text-xs truncate max-w-xs" title="${r.filename}">${r.filename}</td>
        <td colspan="5" class="px-3 py-2 text-xs text-red-600">${r.error}</td>
        <td class="px-3 py-2"><span class="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full">Error</span></td>
      </tr>`;
    }
    const gps = r.gps_used ? `${r.gps_used.lat.toFixed(4)}, ${r.gps_used.lon.toFixed(4)}` : "—";
    const seg = r.road_matched
      ? `${r.road_matched.road_name} km${r.road_matched.km_start.toFixed(1)}`
      : "—";
    const condCls = condClass(r.vvci_label, "cond-vvci");
    return `<tr class="hover:bg-gray-50">
      <td class="px-3 py-2 text-xs max-w-xs truncate" title="${r.filename}">${r.filename}</td>
      <td class="px-3 py-2 font-semibold">${(r.vvci || 0).toFixed(1)}</td>
      <td class="px-3 py-2">${r.pci != null ? r.pci.toFixed(1) : "—"}</td>
      <td class="px-3 py-2"><span class="px-2 py-0.5 rounded-full text-xs ${condCls}">${r.vvci_label}</span></td>
      <td class="px-3 py-2 text-xs">${gps}</td>
      <td class="px-3 py-2 text-xs">${seg}</td>
      <td class="px-3 py-2"><span class="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">OK</span></td>
    </tr>`;
  }).join("");

  // Map pins
  const pins = data.results
    .filter(r => r.gps_used)
    .map(r => ({ lat: r.gps_used.lat, lon: r.gps_used.lon, result: r }));
  updateBatchMapPins(pins);
}

// ── Export ────────────────────────────────────────────────────────────────────

function exportBatch() {
  if (!_batchResults.length) return;

  const headers = [
    "filename","vvci","vvci_label","pci","pci_label",
    "lat","lon","gps_source","road_name","km_start","km_end","error"
  ];

  const rows = _batchResults.map(r => [
    r.filename,
    r.vvci   ?? "",
    r.vvci_label ?? "",
    r.pci    ?? "",
    r.pci_label  ?? "",
    r.gps_used?.lat ?? "",
    r.gps_used?.lon ?? "",
    r.gps_used?.source ?? "",
    r.road_matched?.road_name ?? "",
    r.road_matched?.km_start  ?? "",
    r.road_matched?.km_end    ?? "",
    r.error  ?? "",
  ].map(v => `"${v}"`).join(","));

  const csv = [headers.join(","), ...rows].join("\n");
  downloadBlob(csv, `batch_results_${dateStamp()}.csv`, "text/csv");
}
