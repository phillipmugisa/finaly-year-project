/* visualizer.js — Project breakdown + CNN-explainer model architecture */

// ── Phase data ──────────────────────────────────────────────────────────────

const VIZ_PHASES = [
  {
    n: 1, title: "Data Pipeline", status: "complete", color: "green",
    icon: "🗄️",
    desc: "GPS extraction, image-to-segment matching, multi-year dataset building. 132,544 images matched across Jinja + Hoima (2021-22, 2023-24, 2025-26).",
    tasks: [
      { text: "Road-code-constrained GPS matching (104-zone collision fix)", done: true },
      { text: "EXIF + OCR unified GPS extractor (8-worker parallel OCR)", done: true },
      { text: "Multi-year temporal train/val/test split (most recent year → test)", done: true },
      { text: "Pothole count → grade binning [0,1,3,6,11]", done: true },
      { text: "Dataset quality report + GPS cache (.gps.json sidecars)", done: true },
      { text: "132,544 images matched — Jinja + Hoima, 3 survey years (2021-22 → 2025-26)", done: true },
    ]
  },
  {
    n: 2, title: "Model Development", status: "in-progress", color: "blue",
    icon: "🧠",
    desc: "EfficientNet-B3 + deeper defect head + vVCI/PCI heads. 132k features extracted; CPU training done (MAE 16.6); Colab PCI pre-training on RDD2022 is next.",
    tasks: [
      { text: "Multi-task architecture (6 defect heads + vVCI + PCI) + HeadsModel for Colab", done: true },
      { text: "Two-stage trainer: freeze backbone → unfreeze top 3 blocks", done: true },
      { text: "Deeper DefectHead: shared 1536→256 bottleneck + BN + ReLU before per-defect classifiers", done: true },
      { text: "Analytical vVCI consistency loss: aligns vVCI head with MoWT formula via soft grades", done: true },
      { text: "Segment-level feature aggregation: mean-pool images per 1km segment to align with labels", done: true },
      { text: "Random 70/15/15 resplit by segment — 408 training segments, defect acc 44.6%", done: true },
      { text: "Feature extraction: 132,544 images → features.npz (~400 MB)", done: true },
      { text: "CPU baseline training: test MAE 16.6 vVCI points, defect acc 44.6%", done: true },
      { text: "Model export: TorchScript + ONNX + metadata JSON", done: true },
      { text: "Colab training: HeadsModel on Uganda features + RDD2022 PCI pre-training (pending)", done: false },
    ]
  },
  {
    n: 3, title: "Inference API", status: "complete", color: "green",
    icon: "⚡",
    desc: "FastAPI server: single image, batch ZIP, and GPS segment lookup.",
    tasks: [
      { text: "POST /predict — image → vVCI + PCI + defect grades", done: true },
      { text: "POST /predict-batch — ZIP archive processing", done: true },
      { text: "GET /nearest-segment — GPS → road segment lookup", done: true },
      { text: "API key auth + CORS middleware", done: true },
      { text: "GET /monitor/status — live training/pipeline status", done: true },
    ]
  },
  {
    n: 4, title: "Web Application", status: "complete", color: "green",
    icon: "📱",
    desc: "Single-page app for field survey (smartphone) and ministry batch review.",
    tasks: [
      { text: "Field survey: drag-drop + live camera + GPS override", done: true },
      { text: "Batch review: ZIP upload + results table + CSV export", done: true },
      { text: "Leaflet map with condition colour-coded pins", done: true },
      { text: "Session log + CSV export for field engineers", done: true },
      { text: "Monitor dashboard: pipeline + training progress", done: true },
    ]
  },
  {
    n: 5, title: "Evaluation & Deployment", status: "not-started", color: "gray",
    icon: "🚀",
    desc: "Cross-road generalisation, cost analysis, Docker deployment, research paper.",
    tasks: [
      { text: "Cross-road evaluation: train Jinja → test Hoima", done: false },
      { text: "Multi-year consistency comparison (2021–22 vs 2025–26)", done: false },
      { text: "Cost analysis: UGX smartphone vs Mobicap vs manual survey", done: false },
      { text: "MC Dropout prediction confidence intervals", done: false },
      { text: "Docker Compose: FastAPI + web app in container", done: false },
    ]
  },
];

// ── Model architecture data ──────────────────────────────────────────────────

const BB_BLOCKS = [
  { id: "input", label: "Input",    sub: "224×224×3",  spatial: 224, ch: 3,    type: "input",
    desc: "RGB pavement image resized to 224×224. For Mobicap images the 80 px blue GPS header is cropped first. Normalised to ImageNet mean/std.",
    params: 0, color: "#94a3b8" },
  { id: "stem",  label: "Stem",     sub: "112×112×40", spatial: 112, ch: 40,   type: "conv",
    desc: "Conv2d(3→40, 3×3, stride 2) + BatchNorm + SiLU. Halves spatial resolution, lifts channels from 3 to 40.",
    params: 1080, color: "#38bdf8" },
  { id: "b1",    label: "Block 1",  sub: "112×112×24", spatial: 112, ch: 24,   type: "mbconv",
    desc: "1× MBConv1 — depthwise separable conv, stride 1, Squeeze-Excite(ratio=0.25). Reduces channels (projection).",
    params: 1448, color: "#818cf8" },
  { id: "b2",    label: "Block 2",  sub: "56×56×32",   spatial: 56,  ch: 32,   type: "mbconv",
    desc: "2× MBConv6 — 3×3 depthwise, expansion 6, stride 2. First downsampling stage.",
    params: 14706, color: "#818cf8" },
  { id: "b3",    label: "Block 3",  sub: "28×28×48",   spatial: 28,  ch: 48,   type: "mbconv",
    desc: "3× MBConv6 — 5×5 depthwise, expansion 6, stride 2. Second downsampling.",
    params: 55208, color: "#6366f1" },
  { id: "b4",    label: "Block 4",  sub: "14×14×96",   spatial: 14,  ch: 96,   type: "mbconv",
    desc: "3× MBConv6 — 3×3 depthwise, expansion 6, stride 2. Third downsampling.",
    params: 110992, color: "#6366f1" },
  { id: "b5",    label: "Block 5",  sub: "14×14×136",  spatial: 14,  ch: 136,  type: "mbconv",
    desc: "4× MBConv6 — 5×5 depthwise, expansion 6, stride 1. Deepens representation without downsampling.",
    params: 452592, color: "#7c3aed" },
  { id: "b6",    label: "Block 6",  sub: "7×7×232",    spatial: 7,   ch: 232,  type: "mbconv",
    desc: "5× MBConv6 — 5×5 depthwise, expansion 6, stride 2. Final downsampling to 7×7.",
    params: 1239504, color: "#7c3aed" },
  { id: "b7",    label: "Block 7",  sub: "7×7×384",    spatial: 7,   ch: 384,  type: "mbconv",
    desc: "2× MBConv6 — 3×3 depthwise, expansion 6, stride 1. Deepens to 384 channels at 7×7.",
    params: 2111568, color: "#6d28d9" },
  { id: "gap",   label: "GAP",      sub: "1×1536",     spatial: 1,   ch: 1536, type: "gap",
    desc: "Conv2d(384→1536, 1×1) + BN + SiLU, then Global Average Pooling collapses 7×7 → scalar per channel. Produces the 1536-d feature vector passed to all three heads.",
    params: 590592, color: "#a855f7" },
];

const HEADS_DATA = [
  {
    id: "defect", label: "Defect Head", sub: "Bottleneck + 6 × Linear(256→5)",
    color: "#f97316", bg: "#fff7ed", border: "#fed7aa",
    desc: "Shared 1536→256 bottleneck (BatchNorm + ReLU) followed by six independent classifiers, one per visible defect. The shared layer lets correlated defects (e.g. cracking + ravelling) build a common representation before branching. Predicts grade logits 1–5 (stored 0–4 internally).",
    params: 401694,
    layers: ["Dropout(0.3)", "Linear(1536 → 256)", "BatchNorm1d + ReLU", "── per defect ──", "Dropout(0.15)", "Linear(256 → 5)", "× 6 defects"],
    output: "6 × (B, 5) logits",
    outputs: ["All Cracking","Wide Cracking","Ravelling","Bleeding","Drainage","Potholes"],
  },
  {
    id: "vvci", label: "vVCI Head", sub: "MLP → sigmoid × 100",
    color: "#10b981", bg: "#f0fdf4", border: "#86efac",
    desc: "Two-layer MLP predicting Visual Condition Index. Output sigmoid×100 constrains range to [0,100]. Trained on vVCI computed from 6 image-observable defects (34.5% of full VCI weight).",
    params: 394497,
    layers: ["Dropout(0.3)", "Linear(1536→256)", "BatchNorm1d + ReLU", "Dropout(0.15)", "Linear(256→1)", "Sigmoid × 100"],
    output: "(B, 1) in [0, 100]",
  },
  {
    id: "pci", label: "PCI Head", sub: "MLP → sigmoid × 100",
    color: "#8b5cf6", bg: "#faf5ff", border: "#c4b5fd",
    desc: "Identical architecture to vVCI head, separate weights. Trained on formula-derived PCI from ground-truth defect grades. RDD2022 Japan pre-training (~47k images) runs on Colab T4 via direct FigShare download before Uganda feature training — no local GPU required.",
    params: 394497,
    layers: ["Dropout(0.3)", "Linear(1536→256)", "BatchNorm1d + ReLU", "Dropout(0.15)", "Linear(256→1)", "Sigmoid × 100"],
    output: "(B, 1) in [0, 100]",
  },
];

// ── State ────────────────────────────────────────────────────────────────────

let vizCanvas, vizCtx;
let vizLayout = { blocks: [], heads: [], connections: [] };
let vizAnim = { running: false, t: 0, rafId: null, particles: [] };
let vizHover = null;
let vizSelected = null;
let phaseExpanded = new Set();

// ── Init ─────────────────────────────────────────────────────────────────────

function initVisualizer() {
  renderPhases();
  initVizCanvas();
}

function switchVizTab(tab) {
  document.querySelectorAll('.viz-sub-panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.viz-sub-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`viz-panel-${tab}`).classList.remove('hidden');
  document.getElementById(`viz-btn-${tab}`).classList.add('active');
  if (tab === 'model') { initVizCanvas(); }
}

// ── Project phases ────────────────────────────────────────────────────────────

function renderPhases() {
  const container = document.getElementById('viz-phases');
  container.innerHTML = VIZ_PHASES.map((ph, i) => {
    const done  = ph.tasks.filter(t => t.done).length;
    const total = ph.tasks.length;
    const pct   = Math.round(done / total * 100);
    const statusLabel = { complete: 'Complete', 'in-progress': 'In Progress', 'not-started': 'Not Started' }[ph.status];
    const statusCls   = { complete: 'bg-green-100 text-green-700', 'in-progress': 'bg-blue-100 text-blue-700', 'not-started': 'bg-gray-100 text-gray-500' }[ph.status];
    const barCls      = { complete: 'bg-green-500', 'in-progress': 'bg-blue-500', 'not-started': 'bg-gray-300' }[ph.status];
    const connCls     = i < VIZ_PHASES.length - 1 ? 'after:content-[\"\"] after:absolute after:left-5 after:top-full after:w-0.5 after:h-6 after:bg-gray-200' : '';

    const taskList = ph.tasks.map(t => `
      <li class="flex items-start gap-2 text-sm ${t.done ? 'text-gray-700' : 'text-gray-400'}">
        <span class="mt-0.5 flex-shrink-0 ${t.done ? 'text-green-500' : 'text-gray-300'}">${t.done ? '✓' : '○'}</span>
        ${t.text}
      </li>`).join('');

    return `
      <div class="relative ${connCls}">
        <div class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden transition-all">
          <button class="w-full text-left p-4 flex items-center gap-4"
                  onclick="togglePhase(${ph.n})">
            <div class="w-10 h-10 rounded-full flex items-center justify-center text-lg flex-shrink-0
                        ${ph.status==='complete' ? 'bg-green-100' : ph.status==='in-progress' ? 'bg-blue-100' : 'bg-gray-100'}">
              ${ph.icon}
            </div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="text-xs font-bold text-gray-400 uppercase tracking-wide">Phase ${ph.n}</span>
                <span class="font-semibold text-gray-800">${ph.title}</span>
                <span class="text-xs px-2 py-0.5 rounded-full font-medium ${statusCls}">${statusLabel}</span>
              </div>
              <p class="text-xs text-gray-500 mt-0.5 truncate">${ph.desc}</p>
            </div>
            <div class="flex items-center gap-3 flex-shrink-0">
              <div class="text-right">
                <div class="text-sm font-bold ${ph.status==='complete' ? 'text-green-600' : 'text-gray-700'}">${done}/${total}</div>
                <div class="w-20 bg-gray-100 rounded-full h-1.5 mt-1">
                  <div class="${barCls} h-1.5 rounded-full transition-all" style="width:${pct}%"></div>
                </div>
              </div>
              <span class="text-gray-400 text-sm" id="viz-phase-caret-${ph.n}">▸</span>
            </div>
          </button>
          <div id="viz-phase-tasks-${ph.n}" class="hidden border-t border-gray-50 px-4 pb-4 pt-3">
            <ul class="space-y-2">${taskList}</ul>
          </div>
        </div>
        ${i < VIZ_PHASES.length - 1 ? '<div class="w-0.5 h-5 bg-gray-200 ml-5"></div>' : ''}
      </div>`;
  }).join('');
}

function togglePhase(n) {
  const tasks = document.getElementById(`viz-phase-tasks-${n}`);
  const caret = document.getElementById(`viz-phase-caret-${n}`);
  const open  = !tasks.classList.contains('hidden');
  tasks.classList.toggle('hidden', open);
  caret.textContent = open ? '▸' : '▾';
}

// ── Canvas setup ─────────────────────────────────────────────────────────────

function initVizCanvas() {
  vizCanvas = document.getElementById('viz-model-canvas');
  if (!vizCanvas) return;
  vizCtx = vizCanvas.getContext('2d');
  resizeVizCanvas();
  computeVizLayout();
  drawViz();
  vizCanvas.addEventListener('click',     onVizClick);
  vizCanvas.addEventListener('mousemove', onVizHover);
  vizCanvas.addEventListener('mouseleave', () => { vizHover = null; drawViz(); });
  window.addEventListener('resize', () => {
    resizeVizCanvas(); computeVizLayout(); drawViz();
  });
}

function resizeVizCanvas() {
  const wrap = vizCanvas.parentElement;
  vizCanvas.width  = wrap.clientWidth;
  vizCanvas.height = Math.max(540, Math.round(wrap.clientWidth * 0.62));
}

// ── Layout computation ────────────────────────────────────────────────────────

function computeVizLayout() {
  const W = vizCanvas.width, H = vizCanvas.height;
  const PAD = 24, DEPTH = 7;

  // Backbone strip: y-band
  const BB_Y  = Math.round(H * 0.24);
  const BB_H_AREA = Math.round(H * 0.22);

  // Log-scale block dimensions
  const logS  = b => Math.log2(Math.max(b.spatial, 1));
  const logC  = b => Math.log2(b.ch);
  const maxLS = logS(BB_BLOCKS[0]), minLS = logS(BB_BLOCKS[BB_BLOCKS.length-1]);
  const maxLC = logC(BB_BLOCKS[BB_BLOCKS.length-1]), minLC = logC(BB_BLOCKS[0]);

  const bw = b => lerp(18, 60, (logS(b) - minLS) / (maxLS - minLS));
  const bh = b => lerp(28, Math.round(BB_H_AREA * 0.85), (logC(b) - minLC) / (maxLC - minLC));

  const totalBlockW = BB_BLOCKS.reduce((s, b) => s + bw(b), 0);
  const gapW = (W - PAD*2 - totalBlockW) / (BB_BLOCKS.length - 1);

  let cx = PAD;
  const blocks = BB_BLOCKS.map(b => {
    const w = bw(b), h = bh(b);
    const x = cx, y = BB_Y + (BB_H_AREA - h) / 2;
    cx += w + gapW;
    return { ...b, x, y, w, h, d: DEPTH,
             cx: x + w/2, cy: y + h/2 };
  });

  // Feature vector bar
  const FV_Y = Math.round(H * 0.52);
  const FV_W = Math.round(W * 0.28), FV_H = 34;
  const FV_X = (W - FV_W) / 2;
  const fv = { x: FV_X, y: FV_Y, w: FV_W, h: FV_H, cx: W/2, cy: FV_Y + FV_H/2 };

  // Heads
  const HEAD_Y = Math.round(H * 0.70);
  const HEAD_W = Math.round(W * 0.25), HEAD_H = Math.round(H * 0.13);
  const headXs = [W*0.14, W*0.5, W*0.86].map(cx => cx - HEAD_W/2);
  const heads = HEADS_DATA.map((h, i) => ({
    ...h, x: headXs[i], y: HEAD_Y, w: HEAD_W, h: HEAD_H,
    cx: headXs[i] + HEAD_W/2, cy: HEAD_Y + HEAD_H/2,
  }));

  // Output nodes
  const OUT_Y = Math.round(H * 0.90);
  const outputs = [
    { label: "6 × Grade Logits", sub: "(B, 5) per defect", color: HEADS_DATA[0].color, cx: heads[0].cx, cy: OUT_Y },
    { label: "vVCI Score",       sub: "0 – 100",           color: HEADS_DATA[1].color, cx: heads[1].cx, cy: OUT_Y },
    { label: "PCI Score",        sub: "0 – 100",           color: HEADS_DATA[2].color, cx: heads[2].cx, cy: OUT_Y },
  ];

  // GAP → fv connection (from last backbone block)
  const lastBB = blocks[blocks.length - 1];

  vizLayout = {
    W, H, blocks, fv, heads, outputs, lastBB,
    gapBlock: lastBB,
  };
}

function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

// ── Drawing ───────────────────────────────────────────────────────────────────

function drawViz() {
  if (!vizCtx) return;
  const ctx = vizCtx;
  const { W, H, blocks, fv, heads, outputs, lastBB } = vizLayout;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, W, H);

  // Section labels
  drawSectionLabel(ctx, 'EfficientNet-B3 Backbone  (pretrained ImageNet)', 24, blocks[0].y - 20, '#64748b');
  drawSectionLabel(ctx, 'Feature Vector  1536-d', fv.x, fv.y - 16, '#64748b');
  drawSectionLabel(ctx, 'Task Heads', W * 0.5, heads[0].y - 16, '#64748b', 'center');

  // Connections: backbone arrows
  for (let i = 0; i < blocks.length - 1; i++) {
    const a = blocks[i], b = blocks[i+1];
    drawArrow(ctx, a.x + a.w, a.cy, b.x, b.cy, '#cbd5e1', false);
  }

  // Connection: last backbone → feature vector
  drawCurve(ctx, lastBB.cx, lastBB.y + lastBB.h, fv.cx, fv.y, '#94a3b8');

  // Connection: feature vector → each head
  for (const h of heads) {
    drawCurve(ctx, fv.cx, fv.y + fv.h, h.cx, h.y, h.color + '88');
  }

  // Connection: head → output
  for (let i = 0; i < heads.length; i++) {
    const hd = heads[i], out = outputs[i];
    ctx.strokeStyle = hd.color + 'aa';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(hd.cx, hd.y + hd.h);
    ctx.lineTo(out.cx, out.cy - 14);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw backbone blocks
  for (const b of blocks) {
    const hover    = vizHover === b.id;
    const selected = vizSelected === b.id;
    drawBlock3D(ctx, b, hover, selected);
  }

  // Draw feature vector
  drawFV(ctx, fv);

  // Draw heads
  for (const h of heads) {
    const hover    = vizHover === h.id;
    const selected = vizSelected === h.id;
    drawHead(ctx, h, hover, selected);
  }

  // Draw output pills
  for (const out of outputs) {
    drawOutputPill(ctx, out);
  }

  // Animate particles if running
  if (vizAnim.running) {
    for (const p of vizAnim.particles) drawParticle(ctx, p);
  }

  // Legend
  drawLegend(ctx, W, H);
}

function drawBlock3D(ctx, b, hover, selected) {
  const { x, y, w, h, d, color } = b;
  const alpha = hover ? 1 : 0.88;

  // Right face
  ctx.fillStyle = shadeHex(color, -30);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(x+w, y); ctx.lineTo(x+w+d, y-d);
  ctx.lineTo(x+w+d, y+h-d); ctx.lineTo(x+w, y+h);
  ctx.closePath(); ctx.fill();

  // Top face
  ctx.fillStyle = shadeHex(color, 30);
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x+d, y-d);
  ctx.lineTo(x+w+d, y-d); ctx.lineTo(x+w, y);
  ctx.closePath(); ctx.fill();

  // Front face
  ctx.fillStyle = color;
  if (selected) {
    ctx.shadowColor = color; ctx.shadowBlur = 16;
  }
  roundRect(ctx, x, y, w, h, 3);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Border
  ctx.strokeStyle = selected ? '#1e293b' : (hover ? '#475569' : 'rgba(0,0,0,0.15)');
  ctx.lineWidth   = selected ? 2 : 1;
  roundRect(ctx, x, y, w, h, 3);
  ctx.stroke();

  ctx.globalAlpha = 1;

  // Channel label (small, inside block if tall enough)
  if (h > 42 && w > 14) {
    ctx.save();
    ctx.translate(x + w/2, y + h/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = `bold ${Math.min(10, w-2)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(b.ch, 0, 0);
    ctx.restore();
  }

  // Block label below
  ctx.fillStyle = '#64748b';
  ctx.font = `${Math.min(9, Math.max(7, w-4))}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(b.label, x + w/2, b.y + b.h + d + 3);
  ctx.fillStyle = '#94a3b8';
  ctx.font = `${Math.min(8, Math.max(6, w-4))}px sans-serif`;
  ctx.fillText(b.sub, x + w/2, b.y + b.h + d + 13);
}

function drawFV(ctx, fv) {
  // Gradient bar representing 1536 features
  const grad = ctx.createLinearGradient(fv.x, 0, fv.x + fv.w, 0);
  grad.addColorStop(0,   '#38bdf8');
  grad.addColorStop(0.4, '#818cf8');
  grad.addColorStop(0.7, '#a855f7');
  grad.addColorStop(1,   '#8b5cf6');

  ctx.shadowColor = '#818cf8'; ctx.shadowBlur = 8;
  ctx.fillStyle = grad;
  roundRect(ctx, fv.x, fv.y, fv.w, fv.h, 6);
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 1.5;
  roundRect(ctx, fv.x, fv.y, fv.w, fv.h, 6);
  ctx.stroke();

  ctx.fillStyle = 'white';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Feature Vector  ·  1536-d', fv.cx, fv.cy);
}

function drawHead(ctx, h, hover, selected) {
  const { x, y, w, hh: hSize, color, bg, border } = { ...h, hh: h.h };

  ctx.shadowColor = selected ? color : (hover ? color + '66' : 'rgba(0,0,0,0.08)');
  ctx.shadowBlur  = selected ? 16 : (hover ? 10 : 4);
  ctx.fillStyle = hover || selected ? lightenHex(bg || '#f1f5f9', 0) : (bg || '#f1f5f9');
  roundRect(ctx, x, y, w, hSize, 10);
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.strokeStyle = selected ? color : (hover ? color + 'cc' : (border || '#e2e8f0'));
  ctx.lineWidth   = selected ? 2.5 : (hover ? 2 : 1.5);
  roundRect(ctx, x, y, w, hSize, 10);
  ctx.stroke();

  // Colour accent bar at top
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, 4, [10, 10, 0, 0]) : roundRect(ctx, x, y, w, 4, 0);
  ctx.fill();

  ctx.fillStyle = '#1e293b';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(h.label, h.cx, y + hSize * 0.4);

  ctx.fillStyle = '#64748b';
  ctx.font = '10px sans-serif';
  ctx.fillText(h.sub, h.cx, y + hSize * 0.65);

  ctx.fillStyle = color;
  ctx.font = 'bold 10px sans-serif';
  ctx.fillText(fmtParams(h.params) + ' params', h.cx, y + hSize * 0.85);
}

function drawOutputPill(ctx, out) {
  const r = 14, w = 130;
  const x = out.cx - w/2, y = out.cy - r;

  ctx.fillStyle = out.color + '22';
  ctx.strokeStyle = out.color + 'bb';
  ctx.lineWidth = 1.5;
  roundRect(ctx, x, y, w, r*2, r);
  ctx.fill(); ctx.stroke();

  ctx.fillStyle = out.color;
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(out.label, out.cx, out.cy - 3);

  ctx.fillStyle = '#94a3b8';
  ctx.font = '9px sans-serif';
  ctx.fillText(out.sub, out.cx, out.cy + 7);
}

function drawLegend(ctx, W, H) {
  const items = [
    { color: '#38bdf8', label: 'Conv' },
    { color: '#6366f1', label: 'MBConv block' },
    { color: '#a855f7', label: 'GAP / Feature' },
    { color: '#f97316', label: 'Defect Head' },
    { color: '#10b981', label: 'vVCI Head' },
    { color: '#8b5cf6', label: 'PCI Head' },
  ];
  let lx = 16;
  const ly = H - 20;
  ctx.font = '9px sans-serif';
  ctx.textBaseline = 'middle';
  for (const it of items) {
    ctx.fillStyle = it.color;
    ctx.fillRect(lx, ly - 5, 10, 10);
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText(it.label, lx + 13, ly);
    lx += ctx.measureText(it.label).width + 26;
  }
}

function drawSectionLabel(ctx, text, x, y, color, align = 'left') {
  ctx.fillStyle = color;
  ctx.font = '10px sans-serif';
  ctx.textAlign = align;
  ctx.textBaseline = 'bottom';
  ctx.fillText(text, x, y);
}

function drawArrow(ctx, x1, y1, x2, y2, color, dashed = false) {
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2 - 6, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  // Arrowhead
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2-7, y2-4);
  ctx.lineTo(x2-7, y2+4);
  ctx.closePath(); ctx.fill();
}

function drawCurve(ctx, x1, y1, x2, y2, color) {
  const my = (y1 + y2) / 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(x1, my, x2, my, x2, y2);
  ctx.stroke();
}

function drawParticle(ctx, p) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
  ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0');
  ctx.shadowColor = p.color;
  ctx.shadowBlur  = 8;
  ctx.fill();
  ctx.shadowBlur = 0;
}

// ── Animation ─────────────────────────────────────────────────────────────────

function startVizAnimation() {
  if (vizAnim.running) { stopVizAnimation(); return; }
  vizAnim.running = true;
  vizAnim.t = 0;
  vizAnim.particles = [];

  const { blocks, fv, heads } = vizLayout;

  // Build path waypoints: input → backbone → fv → each head
  const mainPath = [
    { x: blocks[0].cx, y: blocks[0].cy },
    ...blocks.map(b => ({ x: b.x + b.w, y: b.cy })),
    { x: fv.cx, y: fv.cy },
  ];
  const headPaths = heads.map(h => [
    { x: fv.cx, y: fv.cy + fv.h/2 },
    { x: h.cx,  y: h.cy },
  ]);

  function spawnParticle(path, color, delay) {
    vizAnim.particles.push({ path, color, t: -delay, speed: 0.008, r: 4, alpha: 0, x: path[0].x, y: path[0].y });
  }

  spawnParticle(mainPath, '#38bdf8', 0);
  spawnParticle(mainPath, '#818cf8', 0.15);
  spawnParticle(mainPath, '#a855f7', 0.30);
  headPaths.forEach((p, i) => {
    spawnParticle(p, HEADS_DATA[i].color, 0.55 + i * 0.1);
    spawnParticle(p, HEADS_DATA[i].color, 0.75 + i * 0.1);
  });

  document.getElementById('viz-anim-btn').textContent = '⏹ Stop';
  animLoop();
}

function stopVizAnimation() {
  vizAnim.running = false;
  if (vizAnim.rafId) cancelAnimationFrame(vizAnim.rafId);
  vizAnim.particles = [];
  document.getElementById('viz-anim-btn').textContent = '▶ Animate Forward Pass';
  drawViz();
}

function animLoop() {
  if (!vizAnim.running) return;
  vizAnim.t += 0.012;

  for (const p of vizAnim.particles) {
    p.t += p.speed;
    if (p.t < 0) continue;
    const tClamped = Math.max(0, Math.min(1, p.t));
    const pt = pathLerp(p.path, tClamped);
    p.x = pt.x; p.y = pt.y;
    p.alpha = p.t > 0.9 ? (1 - p.t) / 0.1 : Math.min(1, p.t / 0.05);
    if (p.t > 1) p.t = 0;
  }

  drawViz();
  vizAnim.rafId = requestAnimationFrame(animLoop);
}

function pathLerp(pts, t) {
  if (pts.length < 2) return pts[0];
  const seg  = (pts.length - 1) * t;
  const i    = Math.min(Math.floor(seg), pts.length - 2);
  const frac = seg - i;
  return { x: lerp(pts[i].x, pts[i+1].x, frac), y: lerp(pts[i].y, pts[i+1].y, frac) };
}

// ── Click & hover ──────────────────────────────────────────────────────────────

function onVizClick(e) {
  const { x, y } = canvasXY(e);
  const hit = hitTest(x, y);
  vizSelected = (hit && vizSelected !== hit) ? hit : null;
  drawViz();
  showVizDetail(vizSelected);
}

function onVizHover(e) {
  const { x, y } = canvasXY(e);
  const hit = hitTest(x, y);
  if (hit !== vizHover) {
    vizHover = hit;
    vizCanvas.style.cursor = hit ? 'pointer' : 'default';
    drawViz();
  }
}

function canvasXY(e) {
  const r = vizCanvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function hitTest(mx, my) {
  const { blocks, heads } = vizLayout;
  for (const b of [...blocks].reverse()) {
    if (mx >= b.x && mx <= b.x + b.w + b.d && my >= b.y - b.d && my <= b.y + b.h)
      return b.id;
  }
  for (const h of heads) {
    if (mx >= h.x && mx <= h.x + h.w && my >= h.y && my <= h.y + h.h)
      return h.id;
  }
  return null;
}

function showVizDetail(id) {
  const panel = document.getElementById('viz-detail-panel');
  if (!id) { panel.innerHTML = defaultDetail(); return; }

  const block = BB_BLOCKS.find(b => b.id === id);
  const head  = HEADS_DATA.find(h => h.id === id);
  const data  = block || head;
  if (!data) return;

  const color  = data.color || '#6366f1';
  const layers = data.layers ? `
    <div class="mt-3">
      <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Layers</p>
      <div class="space-y-1">
        ${data.layers.map(l => `
          <div class="bg-gray-50 rounded-lg px-3 py-1.5 text-xs font-mono text-gray-700">${l}</div>
        `).join('')}
      </div>
    </div>` : '';

  const outputs = data.outputs ? `
    <div class="mt-3">
      <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Outputs</p>
      <ul class="space-y-1">
        ${data.outputs.map((o, i) => `
          <li class="text-xs text-gray-600 flex items-center gap-2">
            <span class="w-4 h-4 rounded-full text-white text-center flex-shrink-0 flex items-center justify-center" style="background:${color};font-size:9px">${i+1}</span>
            ${o}
          </li>`).join('')}
      </ul>
    </div>` : '';

  const paramStr = data.params > 0 ? fmtParams(data.params) : '—';
  const outputStr = data.output || (block ? `(B, ${block.ch}, ${block.spatial}, ${block.spatial})` : '');

  panel.innerHTML = `
    <div class="border-l-4 pl-3 mb-3" style="border-color:${color}">
      <h4 class="font-bold text-gray-800 text-sm">${data.label}</h4>
      <p class="text-xs text-gray-500">${data.sub || ''}</p>
    </div>
    <p class="text-xs text-gray-600 leading-relaxed">${data.desc}</p>
    <div class="grid grid-cols-2 gap-2 mt-3">
      <div class="bg-gray-50 rounded-lg p-2 text-center">
        <p class="text-xs text-gray-400">Parameters</p>
        <p class="font-bold text-sm text-gray-800">${paramStr}</p>
      </div>
      <div class="bg-gray-50 rounded-lg p-2 text-center">
        <p class="text-xs text-gray-400">Output shape</p>
        <p class="font-bold text-xs text-gray-800">${outputStr}</p>
      </div>
    </div>
    ${layers}
    ${outputs}`;
}

function defaultDetail() {
  return `<p class="text-sm text-gray-400 text-center mt-8">Click any block<br/>to see details</p>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y); ctx.arcTo(x+w, y, x+w, y+r, r);
  ctx.lineTo(x + w, y + h - r); ctx.arcTo(x+w, y+h, x+w-r, y+h, r);
  ctx.lineTo(x + r, y + h); ctx.arcTo(x, y+h, x, y+h-r, r);
  ctx.lineTo(x, y + r); ctx.arcTo(x, y, x+r, y, r);
  ctx.closePath();
}

function shadeHex(hex, amount) {
  const n = parseInt(hex.replace('#',''), 16);
  const r = Math.max(0, Math.min(255, (n >> 16) + amount));
  const g = Math.max(0, Math.min(255, ((n >> 8) & 0xff) + amount));
  const b = Math.max(0, Math.min(255, (n & 0xff) + amount));
  return `#${((1<<24)|(r<<16)|(g<<8)|b).toString(16).slice(1)}`;
}

function lightenHex(hex, _) { return hex; }

function fmtParams(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}
