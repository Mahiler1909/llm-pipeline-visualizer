/**
 * Application entry point.
 * Wires up UI, loads model, and manages the pipeline.
 */

import * as models from './models.js';
import * as pipeline from './pipeline.js';
import * as config from './config.js';
import * as viz from './viz.js';
import * as attention from './attention.js';
import * as samplingPanel from './sampling-panel.js';
import { getTokenColor, cosineSimilarity } from './utils.js';

// ─── DOM Elements ───

const $ = (id) => document.getElementById(id);

const loadingOverlay = $('loading-overlay');
const loadingTitle = $('loading-title');
const loadingBar = $('loading-bar');
const loadingText = $('loading-text');

const modelName = $('model-name');
const badgeLayers = $('badge-layers');
const badgeDim = $('badge-dim');
const badgeVocab = $('badge-vocab');
const badgeParams = $('badge-params');
const footerModel = $('footer-model');

const sidebarClose = $('sidebar-close');
const sidebarOpen = $('sidebar-open');
const sidebar = $('sidebar');
const modelSelect = $('model-select');
const tempSlider = $('temp-slider');
const tempValue = $('temp-value');
const topkSlider = $('topk-slider');
const topkValue = $('topk-value');
const toppSlider = $('topp-slider');
const toppValue = $('topp-value');

const welcomeState = $('welcome-state');

const queryInput = $('query-input');
const generateBtn = $('generate-btn');
const moreBtn = $('more-btn');
const resetBtn = $('reset-btn');
const tokenCount = $('token-count');
const inputTokens = $('input-tokens');

const networkCanvas = $('network-canvas');
const tooltipEl = $('tooltip');

const embeddingModal = $('embedding-modal');
const modalTokenText = $('modal-token-text');
const modalTokenId = $('modal-token-id');
const modalRealBadge = $('modal-real-badge');
const modalHeatmap = $('modal-heatmap');
const modalClose = $('modal-close');
const modalInfoText = $('modal-info-text');
const heatmapTooltip = $('heatmap-tooltip');

const simBtn = $('sim-btn');
const similarityModal = $('similarity-modal');
const simCanvas = $('sim-canvas');
const simClose = $('sim-close');
const simBadge = $('sim-badge');
const simInfoText = $('sim-info-text');

const layerModal = $('layer-modal');
const layerTitle = $('layer-title');
const layerBadge = $('layer-badge');
const layerClose = $('layer-close');
const ldAttn = $('ld-attn');
const ldFfn = $('ld-ffn');
const layerProgress = $('layer-progress');
const layerProgressText = $('layer-progress-text');
const layerProgressBar = $('layer-progress-bar');
const layerAttn = $('layer-attn');
const headSelect = $('head-select');
const attnHeatmap = $('attn-heatmap');
const layerInfoText = $('layer-info-text');

const distCanvas = $('dist-canvas');
const greedyToggle = $('greedy-toggle');

const zoomInBtn = $('zoom-in');
const zoomOutBtn = $('zoom-out');
const zoomResetBtn = $('zoom-reset');

const autoBtn = $('auto-btn');

// Info panel elements
const infoPanel = $('info-panel');
const infoIcon = $('info-icon');
const infoTitle = $('info-title');
const infoBody = $('info-body');
const legend = $('legend');

// ─── Initialize ───

let isProcessing = false;
let isAutoGenerating = false;
let autoGenAbort = false;
let hasGenerated = false;

async function init() {
  console.log('[app] init start');

  try {
    viz.init(networkCanvas, tooltipEl, showEmbeddingModal);
    samplingPanel.init(distCanvas, tooltipEl);
    console.log('[app] viz initialized');
  } catch (err) {
    console.error('[app] viz init error:', err);
  }

  setupEvents();
  console.log('[app] events setup');

  await loadSelectedModel();
  console.log('[app] init complete');
}

async function loadSelectedModel() {
  const modelId = modelSelect.value;
  console.log('[app] loading model:', modelId);

  loadingOverlay.hidden = false;
  loadingOverlay.style.display = '';
  loadingBar.style.width = '0%';

  try {
    await models.loadModel(modelId, (p) => {
      console.log('[app] progress:', p.phase, p.message);
      loadingTitle.textContent = `Cargando ${models.MODEL_CONFIGS[modelId].name}...`;
      loadingText.textContent = p.message;
      if (p.progress != null) {
        loadingBar.style.width = p.progress + '%';
      }
    });

    console.log('[app] model loaded, updating UI');
    updateModelInfo(modelId);
    console.log('[app] model info updated');
  } catch (err) {
    console.error('[app] model load error:', err);
    loadingText.textContent = 'Error: ' + err.message;
    await new Promise(r => setTimeout(r, 3000));
  }

  // ALWAYS hide overlay
  console.log('[app] hiding overlay');
  loadingOverlay.hidden = true;
  loadingOverlay.style.display = 'none';

  if (!hasGenerated) {
    // First load: show welcome state
    if (welcomeState) welcomeState.hidden = false;
  } else if (queryInput.value.trim()) {
    // Model change: auto-regenerate with current text
    await new Promise(r => setTimeout(r, 200));
    try { await runPipeline(); } catch (err) { console.error('[app] auto-generate error:', err); }
  }
}

function updateModelInfo(modelId) {
  const cfg = models.MODEL_CONFIGS[modelId];
  if (!cfg) return;
  modelName.textContent = cfg.name;
  badgeLayers.textContent = `${cfg.layers} capas`;
  badgeDim.textContent = `${cfg.hidden_dim}d`;
  badgeVocab.textContent = `${cfg.vocab_size.toLocaleString()} vocab`;
  badgeParams.textContent = `${cfg.params} params`;
  footerModel.textContent = cfg.name;
}

// ─── Pipeline ───

async function runPipeline() {
  const text = queryInput.value.trim();
  if (!text || isProcessing) return;

  isProcessing = true;
  generateBtn.disabled = true;
  generateBtn.textContent = '...';
  console.log('[app] pipeline start:', text);

  try {
    const result = await pipeline.run(text);
    console.log('[app] pipeline done, tokens:', result.tokens.length, 'preds:', result.predictions.length);

    // Hide welcome state on first generation
    if (welcomeState && !welcomeState.hidden) {
      welcomeState.hidden = true;
    }
    hasGenerated = true;

    renderInputTokens(result.tokens);
    tokenCount.textContent = `${result.tokens.length} tokens`;

    viz.build(result.tokens, result.modelConfig, result.predictions);
    samplingPanel.update(pipeline.getDistribution());
    moreBtn.disabled = false;
    autoBtn.disabled = false;
    simBtn.disabled = result.tokens.length < 2;

    // Real attention arcs (cheap recompute; only if weights already downloaded)
    refreshAttention();
  } catch (err) {
    console.error('[app] pipeline error:', err);
  } finally {
    isProcessing = false;
    generateBtn.disabled = false;
    generateBtn.textContent = 'Generar';
  }
}

function renderInputTokens(tokens) {
  inputTokens.innerHTML = '';
  tokens.forEach((t, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.style.backgroundColor = getTokenColor(i);
    chip.style.color = '#000';
    chip.textContent = t.text.trim() || '⎵';
    chip.title = `ID: ${t.id}`;
    inputTokens.appendChild(chip);
  });
}

// ─── Embedding Modal ───

let currentEmbVec = null;
let currentEmbDims = 0;
let currentEmbGridCols = 0;
let currentEmbTokenText = '';
let heatmapListenersAdded = false;
let embModalRequestId = 0;

async function showEmbeddingModal(tokenId, tokenText) {
  const cfg = models.getConfig(models.getLoadedModelId());
  if (!cfg) return;
  const dims = cfg.hidden_dim;
  const requestId = ++embModalRequestId;

  modalTokenText.textContent = tokenText;
  modalTokenId.textContent = `ID: ${tokenId}`;
  modalRealBadge.hidden = true;
  modalInfoText.textContent = 'Descargando el vector real del modelo desde HuggingFace...';

  // Loading state on the heatmap canvas
  currentEmbVec = null;
  const hctx = modalHeatmap.getContext('2d');
  hctx.clearRect(0, 0, modalHeatmap.width, modalHeatmap.height);
  hctx.fillStyle = '#1a1a2e';
  hctx.fillRect(0, 0, modalHeatmap.width, modalHeatmap.height);
  hctx.fillStyle = '#8b949e';
  hctx.font = '14px monospace';
  hctx.textAlign = 'center';
  hctx.fillText('Cargando vector real...', modalHeatmap.width / 2, modalHeatmap.height / 2);
  embeddingModal.hidden = false;

  const { vector, isReal } = await models.getEmbeddingVectorAsync(tokenId);
  // Discard if the user opened another token meanwhile
  if (requestId !== embModalRequestId || embeddingModal.hidden) return;

  modalRealBadge.hidden = false;
  modalRealBadge.textContent = isReal ? 'REAL' : 'SIMULADO';
  modalRealBadge.className = 'modal__badge ' + (isReal ? 'modal__badge--real' : 'modal__badge--sim');
  modalInfoText.textContent = isReal
    ? `Este es el vector REAL de ${dims} numeros (fila ${tokenId} de la matriz wte.weight de GPT-2) que representa el "significado" del token. Tokens con significados similares tienen vectores parecidos.`
    : `Vector simulado de ${dims} dimensiones (no se pudo descargar el real desde HuggingFace). La idea es la misma: cada token se representa con ${dims} numeros.`;

  currentEmbVec = vector;
  currentEmbDims = dims;
  currentEmbGridCols = Math.ceil(Math.sqrt(dims));
  currentEmbTokenText = tokenText;

  drawEmbeddingHeatmap(vector, dims, -1);

  if (!heatmapListenersAdded) {
    modalHeatmap.addEventListener('mousemove', handleHeatmapHover);
    modalHeatmap.addEventListener('mouseleave', handleHeatmapLeave);
    heatmapListenersAdded = true;
  }
}

function handleHeatmapHover(e) {
  if (!currentEmbVec) return;

  const rect = modalHeatmap.getBoundingClientRect();
  const scaleX = modalHeatmap.width / rect.width;
  const scaleY = modalHeatmap.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  const cols = currentEmbGridCols;
  const size = modalHeatmap.width;
  const cellW = size / cols;
  const rows = Math.ceil(currentEmbDims / cols);
  const cellH = size / rows;

  const col = Math.floor(mx / cellW);
  const row = Math.floor(my / cellH);
  const dimIdx = row * cols + col;

  if (dimIdx < 0 || dimIdx >= currentEmbDims) {
    handleHeatmapLeave();
    return;
  }

  const val = currentEmbVec[dimIdx];
  const absVal = Math.abs(val);

  // Color for the value
  let valColor;
  if (absVal < 0.02) valColor = '#8b949e';
  else if (val > 0) valColor = '#f87171';
  else valColor = '#60a5fa';

  // Bar visualization: map value to a visual bar
  const barPct = Math.min(absVal / 0.3, 1) * 100;
  const barDir = val >= 0 ? 'rojo / positivo' : 'azul / negativo';

  heatmapTooltip.innerHTML =
    `<b>Dim ${dimIdx}</b> / ${currentEmbDims}<br>` +
    `<span style="color:${valColor}; font-size:1.1em; font-weight:700">${val >= 0 ? '+' : ''}${val.toFixed(4)}</span><br>` +
    `<span style="color:#8b949e">` +
    (absVal < 0.02
      ? `Cerca de cero: esta dimension no<br>distingue mucho a "${currentEmbTokenText}"`
      : `Intensidad: ${barPct.toFixed(0)}% (${barDir})<br>` +
        `Cuanto mas ${val > 0 ? 'rojo' : 'azul'}, mas contribuye<br>esta dimension al significado`) +
    `</span>`;

  heatmapTooltip.style.left = (e.clientX + 16) + 'px';
  heatmapTooltip.style.top = (e.clientY - 14) + 'px';
  heatmapTooltip.hidden = false;

  drawEmbeddingHeatmap(currentEmbVec, currentEmbDims, dimIdx);
}

function handleHeatmapLeave() {
  heatmapTooltip.hidden = true;
  if (currentEmbVec) {
    drawEmbeddingHeatmap(currentEmbVec, currentEmbDims, -1);
  }
}

function drawEmbeddingHeatmap(vec, dims, highlightIdx) {
  const ctx = modalHeatmap.getContext('2d');
  const size = modalHeatmap.width;
  const cols = Math.ceil(Math.sqrt(dims));
  const rows = Math.ceil(dims / cols);
  const cellW = size / cols;
  const cellH = size / rows;

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < dims; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const val = vec[i];
    let r, g, b;
    if (val >= 0) {
      const t = Math.min(val / 0.3, 1);
      r = 255;
      g = Math.round(255 * (1 - t * 0.7));
      b = Math.round(255 * (1 - t * 0.8));
    } else {
      const t = Math.min(-val / 0.3, 1);
      r = Math.round(255 * (1 - t * 0.8));
      g = Math.round(255 * (1 - t * 0.6));
      b = 255;
    }
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(col * cellW, row * cellH, cellW - 0.5, cellH - 0.5);
  }

  // Highlight hovered cell
  if (highlightIdx >= 0 && highlightIdx < dims) {
    const hCol = highlightIdx % cols;
    const hRow = Math.floor(highlightIdx / cols);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(hCol * cellW - 0.5, hRow * cellH - 0.5, cellW + 0.5, cellH + 0.5);
  }
}

// ─── Similarity Modal (cosine similarity between input embeddings) ───

const SIM_MARGIN = 78; // space for token labels on top/left axes
let simData = null;    // { labels: [], matrix: Float32Array(n*n), n }
let simListenersAdded = false;
let simRequestId = 0;

async function showSimilarityModal() {
  const tokens = pipeline.getLastTokens();
  if (!tokens || tokens.length < 2) return;
  const requestId = ++simRequestId;

  // Deduplicate by token ID (repeated tokens have identical embeddings)
  const unique = [];
  const seen = new Set();
  for (const t of tokens) {
    if (!seen.has(t.id)) {
      seen.add(t.id);
      unique.push(t);
    }
  }

  simData = null;
  simBadge.hidden = true;
  const ctx = simCanvas.getContext('2d');
  ctx.clearRect(0, 0, simCanvas.width, simCanvas.height);
  ctx.fillStyle = '#8b949e';
  ctx.font = '14px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`Descargando ${unique.length} vectores reales...`, simCanvas.width / 2, simCanvas.height / 2);
  similarityModal.hidden = false;

  const results = await Promise.all(unique.map(t => models.getEmbeddingVectorAsync(t.id)));
  if (requestId !== simRequestId || similarityModal.hidden) return;

  const allReal = results.every(r => r.isReal);
  simBadge.hidden = false;
  simBadge.textContent = allReal ? 'REAL' : 'SIMULADO';
  simBadge.className = 'modal__badge ' + (allReal ? 'modal__badge--real' : 'modal__badge--sim');
  simInfoText.textContent = allReal
    ? 'Cada celda compara dos tokens del prompt usando sus vectores REALES: rojo = significados cercanos (similitud alta), azul = opuestos, oscuro = sin relacion. La diagonal siempre es 1.'
    : 'No se pudieron descargar los vectores reales: con vectores simulados la similitud NO refleja significados. Revisa tu conexion e intenta de nuevo.';

  const n = unique.length;
  const matrix = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const sim = i === j ? 1 : cosineSimilarity(results[i].vector, results[j].vector);
      matrix[i * n + j] = sim;
      matrix[j * n + i] = sim;
    }
  }

  simData = { labels: unique.map(t => t.text.trim() || '⎵'), matrix, n };
  drawSimilarityMatrix(-1, -1);

  if (!simListenersAdded) {
    simCanvas.addEventListener('mousemove', handleSimHover);
    simCanvas.addEventListener('mouseleave', () => {
      heatmapTooltip.hidden = true;
      if (simData) drawSimilarityMatrix(-1, -1);
    });
    simListenersAdded = true;
  }
}

function simColor(sim) {
  // Diverging scale: blue (-1) → dark (0) → red (+1)
  const t = Math.min(Math.abs(sim), 1);
  if (sim >= 0) {
    return `rgb(${Math.round(26 + t * 229)},${Math.round(26 + t * 60)},${Math.round(46 + t * 30)})`;
  }
  return `rgb(${Math.round(26 + t * 30)},${Math.round(26 + t * 70)},${Math.round(46 + t * 209)})`;
}

function drawSimilarityMatrix(hi, hj) {
  if (!simData) return;
  const { labels, matrix, n } = simData;
  const ctx = simCanvas.getContext('2d');
  const size = simCanvas.width;
  const cell = (size - SIM_MARGIN) / n;

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = simColor(matrix[i * n + j]);
      ctx.fillRect(SIM_MARGIN + j * cell, SIM_MARGIN + i * cell, cell - 1, cell - 1);
    }
  }

  // Value inside cells when they are big enough
  if (cell >= 34) {
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i * n + j];
        ctx.fillStyle = Math.abs(v) > 0.55 ? '#0d1117' : '#c9d1d9';
        ctx.fillText(v.toFixed(2), SIM_MARGIN + j * cell + cell / 2, SIM_MARGIN + i * cell + cell / 2);
      }
    }
  }

  // Axis labels
  ctx.font = '10px monospace';
  const maxLabel = 9;
  for (let i = 0; i < n; i++) {
    const text = labels[i].length > maxLabel ? labels[i].slice(0, maxLabel) + '…' : labels[i];
    const active = i === hi || i === hj;
    ctx.fillStyle = active ? '#fbbf24' : '#8b949e';
    // Left axis (rows)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, SIM_MARGIN - 6, SIM_MARGIN + i * cell + cell / 2);
    // Top axis (columns), rotated
    ctx.save();
    ctx.translate(SIM_MARGIN + i * cell + cell / 2, SIM_MARGIN - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'left';
    ctx.fillText(text, 0, 0);
    ctx.restore();
  }

  // Highlight hovered cell
  if (hi >= 0 && hj >= 0) {
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.strokeRect(SIM_MARGIN + hj * cell - 0.5, SIM_MARGIN + hi * cell - 0.5, cell, cell);
  }
}

function handleSimHover(e) {
  if (!simData) return;
  const rect = simCanvas.getBoundingClientRect();
  const scale = simCanvas.width / rect.width;
  const mx = (e.clientX - rect.left) * scale;
  const my = (e.clientY - rect.top) * scale;
  const cell = (simCanvas.width - SIM_MARGIN) / simData.n;
  const j = Math.floor((mx - SIM_MARGIN) / cell);
  const i = Math.floor((my - SIM_MARGIN) / cell);

  if (i < 0 || j < 0 || i >= simData.n || j >= simData.n) {
    heatmapTooltip.hidden = true;
    drawSimilarityMatrix(-1, -1);
    return;
  }

  const sim = simData.matrix[i * simData.n + j];
  let desc;
  if (i === j) desc = 'El mismo token: similitud perfecta';
  else if (sim > 0.4) desc = 'Significados muy relacionados';
  else if (sim > 0.25) desc = 'Algo relacionados';
  else if (sim > -0.1) desc = 'Poca relacion semantica';
  else desc = 'Direcciones opuestas en el espacio';

  heatmapTooltip.innerHTML =
    `<b>"${simData.labels[i]}"</b> vs <b>"${simData.labels[j]}"</b><br>` +
    `<span style="color:${sim > 0.25 ? '#f87171' : '#60a5fa'}; font-size:1.1em; font-weight:700">${sim.toFixed(3)}</span><br>` +
    `<span style="color:#8b949e">${desc}</span>`;
  heatmapTooltip.style.left = (e.clientX + 16) + 'px';
  heatmapTooltip.style.top = (e.clientY - 14) + 'px';
  heatmapTooltip.hidden = false;

  drawSimilarityMatrix(i, j);
}

// ─── Layer Modal (transformer internals + real layer-0 attention) ───

const ATTN_MARGIN = 78;
let attnResult = null;   // result of attention.computeAttention for current tokens
let attnLabels = [];
let attnRequestId = 0;
let currentHeadIdx = -1; // -1 = average of all heads
let attnListenersAdded = false;

/**
 * Recompute real layer-0 attention for the current tokens (only if the
 * weights are already downloaded) and feed the hover arcs in the canvas.
 */
async function refreshAttention() {
  const modelId = models.getLoadedModelId();
  const cfg = models.getConfig(modelId);
  const tokens = pipeline.getLastTokens();
  if (!cfg || !tokens || tokens.length === 0 || !attention.isLoaded(modelId)) return;

  const requestId = ++attnRequestId;
  try {
    const result = await attention.computeAttention(modelId, tokens.map(t => t.id), cfg.heads);
    if (requestId !== attnRequestId) return;
    attnResult = result;
    attnLabels = tokens.map(t => t.text.trim() || '⎵');
    viz.setAttention({ seqLen: result.seqLen, avg: (i, j) => result.avg(i, j) });
    if (!layerModal.hidden && !layerAttn.hidden) drawAttnHeatmap(currentHeadIdx, -1, -1);
  } catch (err) {
    console.warn('[app] no se pudo calcular la atencion:', err);
  }
}

async function showLayerModal(layerIdx) {
  const modelId = models.getLoadedModelId();
  const cfg = models.getConfig(modelId);
  if (!cfg) return;

  layerTitle.textContent = `Interior de la capa ${layerIdx + 1} de ${cfg.layers}`;
  ldAttn.innerHTML = `Multi-Head<br>Attention (${cfg.heads} cabezas)`;
  ldFfn.innerHTML = `FFN<br>${cfg.hidden_dim}&rarr;${cfg.ffn_dim}&rarr;${cfg.hidden_dim}`;
  layerProgress.hidden = true;
  layerAttn.hidden = true;
  layerBadge.hidden = true;
  layerModal.hidden = false;

  if (layerIdx > 0) {
    layerInfoText.textContent =
      `Todas las capas comparten esta estructura: la atencion mezcla informacion entre tokens y el FFN la procesa. ` +
      `Esta capa opera sobre la salida de la capa ${layerIdx}. Por limites del modelo ONNX, ` +
      `los valores REALES de atencion solo se calculan para la capa 1 (haz click en la columna L1).`;
    return;
  }

  layerInfoText.textContent =
    'La matriz muestra cuanto "mira" cada token (fila) a cada token anterior (columna). ' +
    'Es la atencion REAL de la capa 1, calculada en tu navegador con los pesos originales de GPT-2. ' +
    'El triangulo superior esta vacio por la mascara causal: un token no puede mirar al futuro.';

  // Lazy one-time download of the layer-0 weights (~7MB for GPT-2)
  if (!attention.isLoaded(modelId)) {
    layerProgress.hidden = false;
    layerProgressBar.style.width = '0%';
    layerProgressText.textContent = 'Descargando pesos reales de la capa 1...';
    try {
      await attention.loadLayer0(modelId, ({ loaded, total }) => {
        layerProgressBar.style.width = ((loaded / total) * 100).toFixed(1) + '%';
        layerProgressText.textContent =
          `Descargando pesos reales de la capa 1... ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`;
      });
    } catch (err) {
      console.warn('[app] descarga de pesos fallo:', err);
      layerProgress.hidden = true;
      layerBadge.hidden = false;
      layerBadge.textContent = 'SIN DATOS';
      layerBadge.className = 'modal__badge modal__badge--sim';
      layerInfoText.textContent =
        'No se pudieron descargar los pesos reales desde HuggingFace (revisa tu conexion). ' +
        'El diagrama de arriba muestra la estructura de la capa de todas formas.';
      return;
    }
    layerProgressText.textContent = 'Calculando atencion...';
  }

  await refreshAttention();
  layerProgress.hidden = true;
  if (layerModal.hidden) return;

  if (!attnResult) {
    layerBadge.hidden = false;
    layerBadge.textContent = 'SIN DATOS';
    layerBadge.className = 'modal__badge modal__badge--sim';
    layerInfoText.textContent = 'Genera un prompt primero para calcular la atencion sobre sus tokens.';
    return;
  }

  layerBadge.hidden = false;
  layerBadge.textContent = 'REAL';
  layerBadge.className = 'modal__badge modal__badge--real';

  populateHeadSelect(cfg.heads);
  layerAttn.hidden = false;
  drawAttnHeatmap(currentHeadIdx, -1, -1);

  if (!attnListenersAdded) {
    attnHeatmap.addEventListener('mousemove', handleAttnHover);
    attnHeatmap.addEventListener('mouseleave', () => {
      heatmapTooltip.hidden = true;
      if (attnResult) drawAttnHeatmap(currentHeadIdx, -1, -1);
    });
    attnListenersAdded = true;
  }
}

function populateHeadSelect(heads) {
  if (headSelect.options.length !== heads + 1) {
    headSelect.innerHTML = '';
    const avg = document.createElement('option');
    avg.value = '-1';
    avg.textContent = 'Promedio de todas';
    headSelect.appendChild(avg);
    for (let h = 0; h < heads; h++) {
      const o = document.createElement('option');
      o.value = String(h);
      o.textContent = `Head ${h + 1}`;
      headSelect.appendChild(o);
    }
    currentHeadIdx = -1;
  }
  headSelect.value = String(currentHeadIdx);
}

function attnColor(w) {
  const t = Math.sqrt(Math.min(Math.max(w, 0), 1));
  return `rgb(${Math.round(26 + t * 141)},${Math.round(26 + t * 113)},${Math.round(46 + t * 204)})`;
}

function drawAttnHeatmap(headIdx, hi, hj) {
  if (!attnResult) return;
  const n = attnResult.seqLen;
  const ctx = attnHeatmap.getContext('2d');
  const size = attnHeatmap.width;
  const cell = (size - ATTN_MARGIN) / n;

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j > i) {
        ctx.fillStyle = '#11141c'; // causal mask: future is unreachable
      } else {
        const w = headIdx < 0 ? attnResult.avg(i, j) : attnResult.get(headIdx, i, j);
        ctx.fillStyle = attnColor(w);
      }
      ctx.fillRect(ATTN_MARGIN + j * cell, ATTN_MARGIN + i * cell, cell - 1, cell - 1);
    }
  }

  // Values inside cells when readable
  if (cell >= 34) {
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        const w = headIdx < 0 ? attnResult.avg(i, j) : attnResult.get(headIdx, i, j);
        ctx.fillStyle = w > 0.45 ? '#0d1117' : '#c9d1d9';
        ctx.fillText(w.toFixed(2), ATTN_MARGIN + j * cell + cell / 2, ATTN_MARGIN + i * cell + cell / 2);
      }
    }
  }

  // Axis labels
  ctx.font = '10px monospace';
  const maxLabel = 9;
  for (let i = 0; i < n; i++) {
    const text = attnLabels[i].length > maxLabel ? attnLabels[i].slice(0, maxLabel) + '…' : attnLabels[i];
    const active = i === hi || i === hj;
    ctx.fillStyle = active ? '#fbbf24' : '#8b949e';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, ATTN_MARGIN - 6, ATTN_MARGIN + i * cell + cell / 2);
    ctx.save();
    ctx.translate(ATTN_MARGIN + i * cell + cell / 2, ATTN_MARGIN - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'left';
    ctx.fillText(text, 0, 0);
    ctx.restore();
  }

  if (hi >= 0 && hj >= 0) {
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.strokeRect(ATTN_MARGIN + hj * cell - 0.5, ATTN_MARGIN + hi * cell - 0.5, cell, cell);
  }
}

function handleAttnHover(e) {
  if (!attnResult) return;
  const rect = attnHeatmap.getBoundingClientRect();
  const scale = attnHeatmap.width / rect.width;
  const mx = (e.clientX - rect.left) * scale;
  const my = (e.clientY - rect.top) * scale;
  const n = attnResult.seqLen;
  const cell = (attnHeatmap.width - ATTN_MARGIN) / n;
  const j = Math.floor((mx - ATTN_MARGIN) / cell);
  const i = Math.floor((my - ATTN_MARGIN) / cell);

  if (i < 0 || j < 0 || i >= n || j >= n) {
    heatmapTooltip.hidden = true;
    drawAttnHeatmap(currentHeadIdx, -1, -1);
    return;
  }

  if (j > i) {
    heatmapTooltip.innerHTML =
      `<b>Mascara causal</b><br>` +
      `<span style="color:#8b949e">"${attnLabels[i]}" no puede mirar a "${attnLabels[j]}":<br>esta en el futuro de la secuencia</span>`;
  } else {
    const w = currentHeadIdx < 0 ? attnResult.avg(i, j) : attnResult.get(currentHeadIdx, i, j);
    heatmapTooltip.innerHTML =
      `<b>"${attnLabels[i]}"</b> atiende a <b>"${attnLabels[j]}"</b><br>` +
      `<span style="color:#a78bfa; font-size:1.1em; font-weight:700">${(w * 100).toFixed(1)}%</span><br>` +
      `<span style="color:#8b949e">${currentHeadIdx < 0 ? 'Promedio de las cabezas' : 'Head ' + (currentHeadIdx + 1)} &middot; cada fila suma 100%</span>`;
  }
  heatmapTooltip.style.left = (e.clientX + 16) + 'px';
  heatmapTooltip.style.top = (e.clientY - 14) + 'px';
  heatmapTooltip.hidden = false;

  drawAttnHeatmap(currentHeadIdx, i, j);
}

// ─── Info Panel (contextual education) ───

const INFO_CARDS = {
  token: {
    icon: '\ud83d\udcdd',
    title: 'Tokens',
    getBasic(cfg) {
      return `
        <p class="info-panel__text">
          El modelo no lee palabras, lee <strong>tokens</strong>: fragmentos de texto con un ID numerico.
        </p>
        <p class="info-panel__text">
          "the" = 1 token, pero "tokenization" = ["token", "ization"]. Palabras comunes se mantienen enteras; las raras se dividen.
        </p>
        <div class="info-panel__formula">"The capital of" → [464, 3139, 286]</div>`;
    },
    getAdvanced(cfg) {
      const vocab = cfg ? cfg.vocab_size.toLocaleString() : '50,257';
      return `
        <p class="info-panel__text">
          GPT-2 usa <strong>BPE (Byte-Pair Encoding)</strong>: parte de los 256 bytes posibles y
          <strong>fusiona iterativamente el par de simbolos mas frecuente</strong> del corpus de entrenamiento,
          unas 50.000 veces. Resultado: vocabulario de ${vocab} tokens.
        </p>
        <p class="info-panel__text">
          Por eso una palabra rara o inventada se parte en varios pedazos (sus pares nunca fueron frecuentes),
          y el espacio inicial forma parte del token: " the" y "the" son tokens distintos.
        </p>
        <div class="info-panel__formula">vocab = 256 bytes
+ ~50.000 fusiones BPE
+ &lt;|endoftext|&gt;</div>`;
    },
    experiment: 'Escribe una palabra inventada como "tokenizometro" y mira en cuantos pedazos la parte el tokenizer (chips de colores abajo).',
  },
  embedding: {
    icon: '\ud83d\udcca',
    title: 'Embeddings',
    getBasic(cfg) {
      const dim = cfg ? cfg.hidden_dim : 768;
      return `
        <p class="info-panel__text">
          Cada token se convierte en una <strong>lista de ${dim} numeros</strong> que representa su significado.
        </p>
        <p class="info-panel__text">
          Palabras similares tienen numeros parecidos. Asi el modelo sabe que "gato" y "perro" son mas cercanos que "gato" y "avion".
        </p>
        <div class="info-panel__detail">
          <span>Click en un nodo verde para ver el vector REAL del modelo</span>
        </div>`;
    },
    getAdvanced(cfg) {
      const dim = cfg ? cfg.hidden_dim : 768;
      const vocab = cfg ? cfg.vocab_size : 50257;
      return `
        <p class="info-panel__text">
          Los embeddings forman un <strong>espacio vectorial de ${dim} dimensiones</strong> donde la
          <em>direccion</em> codifica significado. La cercania se mide con <strong>similitud coseno</strong>:
        </p>
        <div class="info-panel__formula">cos(\u03b8) = A\u00b7B / (|A|\u00b7|B|)
+1 = mismo sentido
 0 = sin relacion
\u22121 = opuestos</div>
        <p class="info-panel__text">
          La matriz de embeddings (wte) es ${vocab.toLocaleString()}\u00d7${dim} \u2248
          <strong>${Math.round(vocab * dim / 1e6)}M parametros</strong>, y se reutiliza al final del pipeline
          para convertir el vector de salida en logits (<em>weight tying</em>).
        </p>`;
    },
    experiment: 'Genera "The king and the queen" y pulsa el boton \u229e Similitud: \u00bfque par de tokens tiene la similitud mas alta?',
  },
  transformer: {
    icon: '\u26a1',
    title: 'Transformer',
    getBasic(cfg) {
      const layers = cfg ? cfg.layers : 12;
      return `
        <p class="info-panel__text">
          Aqui ocurre la "comprension". Cada token <strong>mira a todos los anteriores</strong> para entender el contexto
          (<strong>atencion</strong>) y luego procesa esa informacion.
        </p>
        <p class="info-panel__text">
          En "capital of <em>Spain</em>", el modelo conecta "capital" con "Spain" para deducir que se habla de Madrid. Se repite en <strong>${layers} capas</strong>, cada vez entendiendo relaciones mas complejas.
        </p>
        <div class="info-panel__detail">
          <span>Click en un nodo morado para ver el interior de la capa (L1 muestra atencion REAL)</span>
        </div>`;
    },
    getAdvanced(cfg) {
      const heads = cfg ? cfg.heads : 12;
      const dim = cfg ? cfg.hidden_dim : 768;
      const ffn = cfg ? cfg.ffn_dim : 3072;
      return `
        <div class="info-panel__formula">Attention(Q,K,V) =
softmax(Q\u00b7K\u1d40/\u221ad<sub>k</sub>)\u00b7V</div>
        <p class="info-panel__text">
          Cada token genera tres vectores: <strong>Q</strong>uery (que busco), <strong>K</strong>ey (que ofrezco)
          y <strong>V</strong>alue (que comunico). El producto Q\u00b7K mide cuanto "encaja" cada par de tokens,
          como un buscador comparando tu consulta con cada documento.
        </p>
        <p class="info-panel__text">
          Esto pasa en <strong>${heads} cabezas en paralelo</strong> (cada una de ${dim / heads} dims) que aprenden
          relaciones distintas: sintaxis, posiciones, referencias... Despues, un <strong>FFN</strong>
          (${dim}\u2192${ffn}\u2192${dim}) procesa cada posicion por separado. Las conexiones residuales (\u2295)
          y LayerNorm estabilizan el entrenamiento.
        </p>`;
    },
    experiment: 'Haz click en la columna L1 para ver la matriz de atencion real y cambia de cabeza: cada una mira a tokens distintos.',
  },
  logit: {
    icon: '\ud83c\udfaf',
    title: 'Logits',
    getBasic(cfg) {
      const vocab = cfg ? cfg.vocab_size.toLocaleString() : '50,257';
      return `
        <p class="info-panel__text">
          El modelo asigna un <strong>puntaje a cada palabra</strong> del vocabulario (${vocab} palabras). A mayor puntaje, mas probable es que sea la siguiente.
        </p>
        <p class="info-panel__text">
          Estos puntajes se convierten en <strong>probabilidades</strong> (0-100%) con softmax. La <strong>temperatura</strong> los ajusta: baja = pocas opciones claras, alta = muchas opciones parejas.
        </p>`;
    },
    getAdvanced(cfg) {
      return `
        <div class="info-panel__formula">p<sub>i</sub> = e^(z<sub>i</sub>/T) / Σ<sub>j</sub> e^(z<sub>j</sub>/T)</div>
        <p class="info-panel__text">
          La temperatura <strong>T divide los logits antes del softmax</strong>: con T&lt;1 las diferencias se
          amplifican (distribucion afilada, casi determinista); con T&gt;1 se atenuan (distribucion plana, mas azar).
          T cercana a 0 equivale a greedy: siempre gana el logit mayor.
        </p>
        <p class="info-panel__text">
          Los logits salen de multiplicar el vector final del ultimo token por la matriz de embeddings
          transpuesta (<em>weight tying</em>): literalmente se mide que token del vocabulario "se parece mas"
          al vector que produjo el transformer.
        </p>`;
    },
    experiment: 'Mueve temperature a 0.1 y mira la DISTRIBUCION EN VIVO del sidebar: una sola barra domina. Subela a 2.0 y compara.',
  },
  sampling: {
    icon: '\ud83c\udfb2',
    title: 'Sampling',
    getBasic(cfg) {
      return `
        <p class="info-panel__text">
          El modelo <strong>no siempre elige la palabra mas probable</strong>. Elige al azar entre las mejores opciones, ponderando por probabilidad. Por eso cada generacion es diferente.
        </p>
        <p class="info-panel__text">
          <strong>Top-K</strong> limita a los ${config.get('topK')} mejores candidatos.
          <strong>Top-P</strong> filtra los que sumen hasta ${(config.get('topP') * 100).toFixed(0)}% de probabilidad.
          La &#9733; marca el token elegido.
        </p>`;
    },
    getAdvanced(cfg) {
      return `
        <div class="info-panel__formula">logits / T
  &rarr; top-k  &rarr; softmax
  &rarr; top-p  &rarr; renormalizar
  &rarr; sample</div>
        <p class="info-panel__text">
          <strong>Greedy</strong> (siempre el mas probable) es determinista pero cae en bucles repetitivos.
          El muestreo ponderado introduce variedad controlada.
        </p>
        <p class="info-panel__text">
          <strong>Top-p (nucleus)</strong> es adaptativo: si el modelo esta muy seguro, el 90% acumulado lo
          cubren 2-3 tokens (pocas opciones); si esta indeciso, entran muchos mas. Por eso suele funcionar
          mejor que un top-k fijo.
        </p>`;
    },
    experiment: 'Activa Greedy en el sidebar y usa Auto-generar dos veces con el mismo prompt: el texto sera identico. Desactivalo y cada corrida cambiara.',
  },
};

let infoTab = 'basic';       // selected tab persists while moving between zones
let currentInfoZone = null;
let currentInfoCfg = null;
let infoHideTimer = null;
let infoPanelHover = false;

function updateInfoPanel(zone, modelCfg) {
  const card = zone ? INFO_CARDS[zone] : null;
  if (card) {
    clearTimeout(infoHideTimer);
    currentInfoZone = zone;
    currentInfoCfg = modelCfg;
    infoIcon.textContent = card.icon;
    infoTitle.textContent = card.title;
    renderInfoBody(card, modelCfg);
    infoPanel.hidden = false;
    legend.hidden = false;
  } else {
    // Delay the hide so the mouse can travel INTO the panel (tabs are clickable)
    clearTimeout(infoHideTimer);
    infoHideTimer = setTimeout(() => {
      if (!infoPanelHover) {
        infoPanel.hidden = true;
        legend.hidden = true;
      }
    }, 350);
  }
}

function renderInfoBody(card, cfg) {
  const content = infoTab === 'advanced' ? card.getAdvanced(cfg) : card.getBasic(cfg);
  infoBody.innerHTML =
    `<div class="info-panel__tabs">
      <button class="info-panel__tab${infoTab === 'basic' ? ' info-panel__tab--active' : ''}" data-tab="basic">B&aacute;sico</button>
      <button class="info-panel__tab${infoTab === 'advanced' ? ' info-panel__tab--active' : ''}" data-tab="advanced">Profundizar</button>
    </div>` +
    content +
    (card.experiment
      ? `<div class="info-panel__experiment"><span>&#129514;</span><span>${card.experiment}</span></div>`
      : '');
}

// ─── Autoregressive Generation ───

const AUTO_GEN_STEPS = 8;
const AUTO_GEN_DELAY = 1200; // ms between steps after animation

async function startAutoGenerate() {
  if (isAutoGenerating || isProcessing) return;
  if (!pipeline.hasResults()) return;

  isAutoGenerating = true;
  autoGenAbort = false;
  autoBtn.textContent = '\u23f9 Parar';
  autoBtn.classList.add('is-running');
  generateBtn.disabled = true;
  moreBtn.disabled = true;

  for (let step = 0; step < AUTO_GEN_STEPS; step++) {
    if (autoGenAbort) break;

    // 1. Get the sampled token from current predictions
    const moreResult = await pipeline.generateMore();
    if (!moreResult || autoGenAbort) break;

    // 2. Animate token travel
    await viz.animateTokenTravel(moreResult.topWord);
    if (autoGenAbort) break;

    // 3. Update input with new text
    queryInput.value = moreResult.newText;

    // 4. Run pipeline with new text (re-tokenize + re-infer)
    isProcessing = true;
    try {
      const result = await pipeline.run(moreResult.newText);
      renderInputTokens(result.tokens);
      tokenCount.textContent = `${result.tokens.length} tokens`;
      viz.build(result.tokens, result.modelConfig, result.predictions);
    } catch (err) {
      console.error('[app] autoregressive step error:', err);
      break;
    } finally {
      isProcessing = false;
    }

    if (autoGenAbort) break;

    // 5. Pause so the user can see the result
    await new Promise(r => setTimeout(r, AUTO_GEN_DELAY));
  }

  stopAutoGenerate();
}

function stopAutoGenerate() {
  autoGenAbort = true;
  isAutoGenerating = false;
  autoBtn.textContent = '\u25b6 Auto-generar';
  autoBtn.classList.remove('is-running');
  generateBtn.disabled = false;
  moreBtn.disabled = !pipeline.hasResults();
}

// ─── Events ───

function setupEvents() {
  generateBtn.addEventListener('click', runPipeline);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runPipeline();
  });

  moreBtn.addEventListener('click', async () => {
    if (isProcessing || isAutoGenerating) return;
    const result = await pipeline.generateMore();
    if (result) {
      queryInput.value = result.newText;
      await runPipeline();
    }
  });

  autoBtn.addEventListener('click', () => {
    if (isAutoGenerating) {
      stopAutoGenerate();
    } else {
      startAutoGenerate();
    }
  });

  resetBtn.addEventListener('click', () => {
    if (isAutoGenerating) stopAutoGenerate();
    queryInput.value = '';
    inputTokens.innerHTML = '';
    tokenCount.textContent = '0 tokens';
    moreBtn.disabled = true;
    autoBtn.disabled = true;
    simBtn.disabled = true;
    samplingPanel.clear();
    hasGenerated = false;
    viz.clear();
    if (welcomeState) welcomeState.hidden = false;
    queryInput.focus();
  });

  // Info panel: listen for hover zone changes on the canvas
  viz.onHoverZone(updateInfoPanel);

  // Keep the panel open while the mouse is over it (the tabs are clickable)
  infoPanel.addEventListener('mouseenter', () => {
    infoPanelHover = true;
    clearTimeout(infoHideTimer);
  });
  infoPanel.addEventListener('mouseleave', () => {
    infoPanelHover = false;
    infoPanel.hidden = true;
    legend.hidden = true;
  });

  // Tab switching (Basico / Profundizar) via delegation
  infoBody.addEventListener('click', (e) => {
    const btn = e.target.closest('.info-panel__tab');
    if (!btn) return;
    infoTab = btn.dataset.tab;
    const card = INFO_CARDS[currentInfoZone];
    if (card) renderInfoBody(card, currentInfoCfg);
  });

  sidebarClose.addEventListener('click', () => {
    document.body.classList.add('sidebar-hidden');
  });

  sidebarOpen.addEventListener('click', () => {
    document.body.classList.remove('sidebar-hidden');
  });

  tempSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    tempValue.textContent = val.toFixed(2);
    config.set('temperature', val);
  });

  topkSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    topkValue.textContent = val;
    config.set('topK', val);
  });

  toppSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    toppValue.textContent = val.toFixed(2);
    config.set('topP', val);
  });

  modelSelect.addEventListener('change', async () => {
    config.set('modelId', modelSelect.value);
    await loadSelectedModel();
  });

  config.onChange((key) => {
    if ((key === 'temperature' || key === 'topK' || key === 'topP' || key === 'greedy') && pipeline.hasResults()) {
      const predictions = pipeline.recomputePredictions();
      if (predictions) {
        viz.updatePredictions(predictions);
        samplingPanel.update(pipeline.getDistribution());
      }
    }
  });

  greedyToggle.addEventListener('change', () => {
    config.set('greedy', greedyToggle.checked);
  });

  modalClose.addEventListener('click', () => { embeddingModal.hidden = true; heatmapTooltip.hidden = true; });
  embeddingModal.addEventListener('click', (e) => {
    if (e.target === embeddingModal) { embeddingModal.hidden = true; heatmapTooltip.hidden = true; }
  });

  viz.onLayerClick((layerIdx) => {
    showLayerModal(layerIdx).catch(err => console.error('[app] layer modal error:', err));
  });
  layerClose.addEventListener('click', () => { layerModal.hidden = true; heatmapTooltip.hidden = true; });
  layerModal.addEventListener('click', (e) => {
    if (e.target === layerModal) { layerModal.hidden = true; heatmapTooltip.hidden = true; }
  });
  headSelect.addEventListener('change', () => {
    currentHeadIdx = parseInt(headSelect.value, 10);
    drawAttnHeatmap(currentHeadIdx, -1, -1);
  });

  simBtn.addEventListener('click', () => {
    showSimilarityModal().catch(err => console.error('[app] similarity error:', err));
  });
  simClose.addEventListener('click', () => { similarityModal.hidden = true; heatmapTooltip.hidden = true; });
  similarityModal.addEventListener('click', (e) => {
    if (e.target === similarityModal) { similarityModal.hidden = true; heatmapTooltip.hidden = true; }
  });

  zoomInBtn.addEventListener('click', () => viz.zoomIn());
  zoomOutBtn.addEventListener('click', () => viz.zoomOut());
  zoomResetBtn.addEventListener('click', () => viz.resetView());

  // Config info icons → reuse the global tooltip (lives outside sidebar overflow)
  document.querySelectorAll('.config__info').forEach(icon => {
    icon.addEventListener('mouseenter', (e) => {
      const text = icon.getAttribute('data-tooltip');
      if (!text) return;
      tooltipEl.innerHTML = text;
      tooltipEl.style.left = (e.clientX + 14) + 'px';
      tooltipEl.style.top = (e.clientY - 10) + 'px';
      tooltipEl.hidden = false;
    });
    icon.addEventListener('mousemove', (e) => {
      tooltipEl.style.left = (e.clientX + 14) + 'px';
      tooltipEl.style.top = (e.clientY - 10) + 'px';
    });
    icon.addEventListener('mouseleave', () => {
      tooltipEl.hidden = true;
    });
  });
}

// ─── Start ───
console.log('[app] module loaded');
init().catch(err => {
  console.error('[app] FATAL init error:', err);
  loadingOverlay.hidden = true;
  loadingOverlay.style.display = 'none';
});
