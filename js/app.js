/**
 * Application entry point.
 * Wires up UI, loads model, and manages the pipeline.
 */

import * as models from './models.js';
import * as pipeline from './pipeline.js';
import * as config from './config.js';
import * as viz from './viz.js';
import { getTokenColor } from './utils.js';

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
const modalHeatmap = $('modal-heatmap');
const modalClose = $('modal-close');
const modalInfoText = $('modal-info-text');
const heatmapTooltip = $('heatmap-tooltip');

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
    moreBtn.disabled = false;
    autoBtn.disabled = false;
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

function showEmbeddingModal(tokenId, tokenText) {
  const cfg = models.getConfig(models.getLoadedModelId());
  if (!cfg) return;
  const dims = cfg.hidden_dim;

  modalTokenText.textContent = tokenText;
  modalTokenId.textContent = `ID: ${tokenId}`;
  modalInfoText.textContent = `Este vector de ${dims} numeros representa el "significado" del token en un espacio matematico. Tokens con significados similares tienen vectores parecidos.`;

  const vec = models.getEmbeddingVector(tokenId, dims);
  currentEmbVec = vec;
  currentEmbDims = dims;
  currentEmbGridCols = Math.ceil(Math.sqrt(dims));
  currentEmbTokenText = tokenText;

  drawEmbeddingHeatmap(vec, dims, -1);
  embeddingModal.hidden = false;

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

// ─── Info Panel (contextual education) ───

const INFO_CARDS = {
  token: {
    icon: '\ud83d\udcdd',
    title: 'Tokens',
    getContent(cfg) {
      return `
        <p class="info-panel__text">
          El modelo no lee palabras, lee <strong>tokens</strong>: fragmentos de texto con un ID numerico.
        </p>
        <p class="info-panel__text">
          "the" = 1 token, pero "tokenization" = ["token", "ization"]. Palabras comunes se mantienen enteras; las raras se dividen.
        </p>
        <div class="info-panel__formula">"The capital of" → [464, 3139, 286]</div>`;
    },
  },
  embedding: {
    icon: '\ud83d\udcca',
    title: 'Embeddings',
    getContent(cfg) {
      const dim = cfg ? cfg.hidden_dim : 768;
      return `
        <p class="info-panel__text">
          Cada token se convierte en una <strong>lista de ${dim} numeros</strong> que representa su significado.
        </p>
        <p class="info-panel__text">
          Palabras similares tienen numeros parecidos. Asi el modelo sabe que "gato" y "perro" son mas cercanos que "gato" y "avion".
        </p>
        <div class="info-panel__detail">
          <span>Click en un nodo verde para ver su vector</span>
        </div>`;
    },
  },
  transformer: {
    icon: '\u26a1',
    title: 'Transformer',
    getContent(cfg) {
      const heads = cfg ? cfg.heads : 12;
      const layers = cfg ? cfg.layers : 12;
      return `
        <p class="info-panel__text">
          Aqui ocurre la "comprension". Cada token <strong>mira a todos los demas</strong> para entender el contexto
          (<strong>atencion</strong>) y luego procesa esa informacion.
        </p>
        <p class="info-panel__text">
          En "capital of <em>Spain</em>", el modelo conecta "capital" con "Spain" para deducir que se habla de Madrid. Se repite en <strong>${layers} capas</strong>, cada vez entendiendo relaciones mas complejas.
        </p>`;
    },
  },
  logit: {
    icon: '\ud83c\udfaf',
    title: 'Logits',
    getContent(cfg) {
      const vocab = cfg ? cfg.vocab_size.toLocaleString() : '50,257';
      return `
        <p class="info-panel__text">
          El modelo asigna un <strong>puntaje a cada palabra</strong> del vocabulario (${vocab} palabras). A mayor puntaje, mas probable es que sea la siguiente.
        </p>
        <p class="info-panel__text">
          Estos puntajes se convierten en <strong>probabilidades</strong> (0-100%) con softmax. La <strong>temperatura</strong> los ajusta: baja = pocas opciones claras, alta = muchas opciones parejas.
        </p>`;
    },
  },
  sampling: {
    icon: '\ud83c\udfb2',
    title: 'Sampling',
    getContent(cfg) {
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
  },
};

function updateInfoPanel(zone, modelCfg) {
  const card = zone ? INFO_CARDS[zone] : null;
  if (card) {
    infoIcon.textContent = card.icon;
    infoTitle.textContent = card.title;
    infoBody.innerHTML = card.getContent(modelCfg);
    infoPanel.hidden = false;
    legend.hidden = false;
  } else {
    infoPanel.hidden = true;
    legend.hidden = true;
  }
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
    hasGenerated = false;
    viz.clear();
    if (welcomeState) welcomeState.hidden = false;
    queryInput.focus();
  });

  // Info panel: listen for hover zone changes on the canvas
  viz.onHoverZone(updateInfoPanel);

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
    if ((key === 'temperature' || key === 'topK' || key === 'topP') && pipeline.hasResults()) {
      const predictions = pipeline.recomputePredictions();
      if (predictions) {
        viz.updatePredictions(predictions);
      }
    }
  });

  modalClose.addEventListener('click', () => { embeddingModal.hidden = true; heatmapTooltip.hidden = true; });
  embeddingModal.addEventListener('click', (e) => {
    if (e.target === embeddingModal) { embeddingModal.hidden = true; heatmapTooltip.hidden = true; }
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
