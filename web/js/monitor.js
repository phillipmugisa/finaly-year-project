/* monitor.js — live training + pipeline dashboard */

const MON_POLL_MS = 10000;
let monTimer = null;
let lossChart = null, maeChart = null;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function initMonitor() {
  buildCharts();
  fetchMonitorStatus();
  monTimer = setInterval(fetchMonitorStatus, MON_POLL_MS);
}

function stopMonitor() {
  if (monTimer) { clearInterval(monTimer); monTimer = null; }
}

async function fetchMonitorStatus() {
  const base = (window.API_BASE || '') ;
  try {
    const r = await fetch(`${base}/monitor/status`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderMonitor(data);
    document.getElementById('mon-error').classList.add('hidden');
  } catch (e) {
    document.getElementById('mon-error').textContent = `Could not reach API: ${e.message}`;
    document.getElementById('mon-error').classList.remove('hidden');
  }
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

function renderMonitor(d) {
  renderPipeline(d.pipeline, d.dataset_partial, d.dataset_full);
  renderFeatureExtraction(d.feature_extraction);
  renderTraining('mon-train',   d.training, 'Colab Training (features)', d.dataset_full);
  renderTraining('mon-retrain', d.retrain,  'Full Model Retrain',        d.dataset_full);
  renderWatcher(d.watcher_running);
  updateCharts(d.training, d.retrain);
  document.getElementById('mon-last-update').textContent =
    'Last updated: ' + new Date().toLocaleTimeString();
}

// ---------------------------------------------------------------------------
// Pipeline section
// ---------------------------------------------------------------------------

function renderPipeline(p, dsPartial, dsFull) {
  const el = document.getElementById('mon-pipeline');
  const statusBadge = statusPill(p.status, {
    running: 'blue', done: 'green', idle: 'gray', not_started: 'gray'
  });

  const sections = (p.sections || []).map(s => {
    const pct = s.gps_total > 0 ? Math.round(s.gps_done / s.gps_total * 100) : 0;
    const bar = s.status === 'done'
      ? `<div class="w-full bg-green-100 rounded-full h-2"><div class="bg-green-500 h-2 rounded-full w-full"></div></div>`
      : `<div class="w-full bg-gray-200 rounded-full h-2">
           <div class="bg-blue-500 h-2 rounded-full transition-all" style="width:${pct}%"></div>
         </div>`;
    const matchedStr = s.matched > 0
      ? `<span class="text-green-700 font-semibold">${s.matched.toLocaleString()} matched</span>`
      : `<span class="text-gray-400">matching…</span>`;
    const dirs = (s.image_dirs || []).map(d =>
      `<span class="text-xs text-gray-400">${shortPath(d.path)} (${d.count.toLocaleString()})</span>`
    ).join('<br/>');

    return `
      <div class="border border-gray-100 rounded-xl p-3 mb-2">
        <div class="flex items-center justify-between mb-1">
          <span class="font-medium text-sm text-gray-700">${s.year}</span>
          ${s.status === 'done'
            ? '<span class="text-xs text-green-600 font-semibold">✓ Done</span>'
            : `<span class="text-xs text-blue-600">${pct}%</span>`}
        </div>
        ${dirs ? `<div class="mb-1">${dirs}</div>` : ''}
        ${bar}
        <div class="mt-1 text-xs">${matchedStr}
          ${s.gps_total > 0 ? ` · GPS: ${s.gps_done.toLocaleString()}/${s.gps_total.toLocaleString()}` : ''}
        </div>
      </div>`;
  }).join('');

  const totalMatched = p.total_matched || 0;

  el.innerHTML = `
    <div class="flex items-center justify-between mb-3">
      <h3 class="font-semibold text-gray-700">Data Pipeline</h3>
      ${statusBadge}
    </div>
    ${sections || '<p class="text-sm text-gray-400">No pipeline activity yet.</p>'}
    <div class="mt-2 pt-2 border-t border-gray-100 flex gap-6 text-xs text-gray-500">
      <span>Total matched: <strong class="text-gray-700">${totalMatched.toLocaleString()}</strong></span>
      <span>Partial CSV: <strong class="text-gray-700">${(dsPartial.rows||0).toLocaleString()} rows</strong></span>
      <span>Full CSV: <strong class="text-gray-700">${(dsFull.rows||0).toLocaleString()} rows</strong></span>
    </div>`;
}

// ---------------------------------------------------------------------------
// Feature extraction section
// ---------------------------------------------------------------------------

function renderFeatureExtraction(fe) {
  const el = document.getElementById('mon-extract');
  if (!el) return;

  const status = (fe && fe.status) || 'not_started';
  const statusBadge = statusPill(status, {
    running: 'blue', complete: 'green', stopped: 'yellow', not_started: 'gray'
  });

  if (status === 'not_started') {
    el.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-semibold text-gray-700">Feature Extraction</h3>
        ${statusBadge}
      </div>
      <p class="text-sm text-gray-400">Not started. Run <code class="bg-gray-100 px-1 rounded text-xs">python scripts/extract_features.py --dataset outputs/dataset.csv --output outputs/features.npz</code></p>`;
    return;
  }

  const pct      = fe.pct || 0;
  const barColor = status === 'complete' ? 'bg-green-500' : 'bg-blue-500';
  const barW     = status === 'complete' ? 100 : pct;
  const etaStr   = (fe.eta && status === 'running') ? `ETA ${fe.eta}` : '';
  const spdStr   = (fe.speed && status === 'running') ? `${fe.speed.toFixed(1)} it/s` : '';
  const hint     = [etaStr, spdStr].filter(Boolean).join(' · ');

  el.innerHTML = `
    <div class="flex items-center justify-between mb-3">
      <h3 class="font-semibold text-gray-700">Feature Extraction</h3>
      ${statusBadge}
    </div>
    <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
      <span>Batch ${(fe.batches_done || 0).toLocaleString()} / ${(fe.batches_total || 0).toLocaleString()}</span>
      <span class="text-gray-400">${hint}</span>
    </div>
    <div class="w-full bg-gray-200 rounded-full h-2.5">
      <div class="${barColor} h-2.5 rounded-full transition-all" style="width:${barW}%"></div>
    </div>
    <p class="text-xs text-gray-400 mt-2">
      Extracts 1536-d EfficientNet-B3 features → <code class="bg-gray-100 px-1 rounded">outputs/features.npz</code>
      for Colab training (~400 MB, avoids uploading 200 GB of raw images).
    </p>`;
}

// ---------------------------------------------------------------------------
// Training section
// ---------------------------------------------------------------------------

function renderTraining(containerId, t, title, ds) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const statusBadge = statusPill(t.status, {
    running: 'blue', complete: 'green', stopped: 'yellow',
    starting: 'blue', not_started: 'gray', idle: 'gray'
  });

  if (t.status === 'not_started' || t.status === 'idle') {
    el.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-semibold text-gray-700">${title}</h3>
        ${statusBadge}
      </div>
      <p class="text-sm text-gray-400">Not started yet.</p>`;
    return;
  }

  const cur   = t.current_epoch !== null ? t.current_epoch + 1 : 0;
  const total = t.total_epochs || 40;
  const pct   = total > 0 ? Math.round(cur / total * 100) : 0;

  const stageLabel = t.stage === 2
    ? `<span class="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full ml-2">Stage 2</span>`
    : `<span class="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full ml-2">Stage 1</span>`;

  const metrics = t.current_epoch !== null ? (() => {
    const last = t.epochs[t.epochs.length - 1] || {};
    return `
      <div class="grid grid-cols-2 gap-2 mt-3">
        ${metricTile('Val MAE', last.val_mae?.toFixed(2), 'text-blue-700')}
        ${metricTile('Val RMSE', last.val_rmse?.toFixed(2), 'text-indigo-700')}
        ${metricTile('Defect Acc', last.defect_acc !== undefined ? (last.defect_acc * 100).toFixed(1) + '%' : '—', 'text-teal-700')}
        ${metricTile('Train Loss', last.train_loss?.toFixed(4), 'text-gray-600')}
      </div>
      ${t.best_val_mae !== null
        ? `<p class="text-xs text-green-700 mt-2">Best val MAE: <strong>${t.best_val_mae?.toFixed(2)}</strong> at epoch ${t.best_epoch}</p>`
        : ''}`;
  })() : '';

  const etaStr = (() => {
    if (t.status !== 'running' || !t.last_epoch_sec || !total) return '';
    const remaining = total - cur;
    const secs = remaining * t.last_epoch_sec;
    if (secs < 60) return `~${secs}s remaining`;
    if (secs < 3600) return `~${Math.round(secs/60)}m remaining`;
    return `~${(secs/3600).toFixed(1)}h remaining`;
  })();

  el.innerHTML = `
    <div class="flex items-center justify-between mb-3">
      <div class="flex items-center">
        <h3 class="font-semibold text-gray-700">${title}</h3>
        ${stageLabel}
      </div>
      ${statusBadge}
    </div>
    <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
      <span>Epoch ${cur} / ${total}</span>
      <span class="text-gray-400">${etaStr}</span>
    </div>
    <div class="w-full bg-gray-200 rounded-full h-2.5">
      <div class="bg-green-500 h-2.5 rounded-full transition-all" style="width:${pct}%"></div>
    </div>
    ${metrics}
    <p class="text-xs text-gray-400 mt-2">Dataset: ${(ds.rows||0).toLocaleString()} images
      · ${Object.entries(ds.splits||{}).map(([k,v])=>`${k}: ${v}`).join(', ')}</p>`;
}

// ---------------------------------------------------------------------------
// Watcher
// ---------------------------------------------------------------------------

function renderWatcher(running) {
  const el = document.getElementById('mon-watcher');
  if (!el) return;
  el.innerHTML = running
    ? `<span class="inline-flex items-center gap-1.5 text-sm text-blue-700">
         <span class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
         Auto-extract watcher active — will run feature extraction after pipeline completes
       </span>`
    : `<span class="text-sm text-gray-400">Watcher not running. Start with <code class="bg-gray-100 px-1 rounded text-xs">bash scripts/auto_extract.sh &lt;pipeline_pid&gt;</code></span>`;
}

// ---------------------------------------------------------------------------
// Charts (Chart.js loaded lazily via CDN)
// ---------------------------------------------------------------------------

function buildCharts() {
  const lossCtx = document.getElementById('mon-chart-loss');
  const maeCtx  = document.getElementById('mon-chart-mae');
  if (!lossCtx || !maeCtx || typeof Chart === 'undefined') return;

  const commonOpts = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
    scales: {
      x: { title: { display: true, text: 'Epoch', font: { size: 10 } } },
      y: { beginAtZero: false },
    },
  };

  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Train Loss (Colab features)', data: [], borderColor: '#3b82f6', tension: 0.3, pointRadius: 2 },
      { label: 'Train Loss (Full model)',     data: [], borderColor: '#8b5cf6', tension: 0.3, pointRadius: 2 },
    ]},
    options: { ...commonOpts, scales: { ...commonOpts.scales, y: { title: { display: true, text: 'Loss' } } } },
  });

  maeChart = new Chart(maeCtx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Val MAE (Colab features)', data: [], borderColor: '#10b981', tension: 0.3, pointRadius: 2 },
      { label: 'Val MAE (Full model)',     data: [], borderColor: '#f59e0b', tension: 0.3, pointRadius: 2 },
    ]},
    options: { ...commonOpts, scales: { ...commonOpts.scales, y: { title: { display: true, text: 'MAE (vVCI points)' } } } },
  });
}

function updateCharts(training, retrain) {
  if (!lossChart || !maeChart) return;

  const partial = training.epochs || [];
  const full    = retrain.epochs  || [];
  const maxLen  = Math.max(partial.length, full.length);
  const labels  = Array.from({ length: maxLen }, (_, i) => i);

  lossChart.data.labels        = labels;
  lossChart.data.datasets[0].data = partial.map(e => e.train_loss);
  lossChart.data.datasets[1].data = full.map(e => e.train_loss);
  lossChart.update('none');

  maeChart.data.labels         = labels;
  maeChart.data.datasets[0].data = partial.map(e => e.val_mae);
  maeChart.data.datasets[1].data = full.map(e => e.val_mae);
  maeChart.update('none');
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function statusPill(status, colorMap) {
  const c = colorMap[status] || 'gray';
  const label = status.replace(/_/g, ' ');
  const dot = status === 'running' || status === 'starting'
    ? `<span class="w-2 h-2 rounded-full bg-${c}-500 animate-pulse mr-1.5"></span>`
    : `<span class="w-2 h-2 rounded-full bg-${c}-400 mr-1.5"></span>`;
  return `<span class="inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-full bg-${c}-100 text-${c}-700">${dot}${label}</span>`;
}

function metricTile(label, value, cls) {
  return `
    <div class="bg-gray-50 rounded-lg p-2 text-center">
      <p class="text-xs text-gray-400">${label}</p>
      <p class="font-bold text-sm ${cls}">${value ?? '—'}</p>
    </div>`;
}

function shortPath(p) {
  const parts = p.split('/');
  return parts.slice(-2).join('/');
}
