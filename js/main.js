/**
 * Orquestador: estado central, boot del modelo, ciclo autorregresivo,
 * rail de progreso y barra de contexto sticky.
 */

import * as models from './models.js';
import * as pipeline from './pipeline.js';
import * as config from './config.js';
import * as presenter from './presenter.js';
import { renderProse } from './content.js';
import { esc } from './stages/shared.js';
import * as llmStage from './stages/llm.js';
import * as tokensStage from './stages/tokens.js';
import * as embeddingsStage from './stages/embeddings.js';
import * as posicionStage from './stages/posicion.js';
import * as atencionStage from './stages/atencion.js';
import * as bloqueStage from './stages/bloque.js';
import * as logitsStage from './stages/logits.js';
import * as muestreoStage from './stages/muestreo.js';
import * as bucleStage from './stages/bucle.js';
import * as entrenamientoStage from './stages/entrenamiento.js';
import * as limitesStage from './stages/limites.js';

// ─── Modelo fijo (DistilGPT-2), con escape hatch ?model= ───

const MODEL_SHORTCUTS = {
  'distilgpt2': 'onnx-community/distilgpt2-ONNX',
  'gpt2': 'onnx-community/gpt2-ONNX',
  'gpt2-medium': 'onnx-community/gpt2-medium-ONNX',
  'gpt2-large': 'onnx-community/gpt2-large-ONNX',
};

const param = new URLSearchParams(location.search).get('model');
const MODEL_ID = MODEL_SHORTCUTS[param] || MODEL_SHORTCUTS['distilgpt2'];

// ─── Estado central ───

const state = {
  modelId: MODEL_ID,
  modelConfig: models.getConfig(MODEL_ID),
  baseText: '',
  text: '',
  tokens: [],
  predictions: null,
  history: [],   // [{ word, prob, cycle }]
  cycle: 1,
  modelReady: false,
};

// ─── DOM ───

const contextBar = document.getElementById('context-bar');
const contextText = document.getElementById('context-text');
const cycleLabel = document.getElementById('cycle-label');
const modelStatus = document.getElementById('model-status');
const promptInput = document.getElementById('prompt-input');

// ─── Boot ───

renderProse(state.modelConfig);
llmStage.init(document.getElementById('w-llm'), { onNavigate: scrollToSection, modelConfig: state.modelConfig });
tokensStage.init(document.getElementById('w-tokens'));
embeddingsStage.init(document.getElementById('w-embeddings'));
posicionStage.init(document.getElementById('w-posicion'));
atencionStage.init(document.getElementById('w-atencion'), document.getElementById('st-atencion'));
bloqueStage.init(document.getElementById('w-bloque'), state.modelConfig);
logitsStage.init(document.getElementById('w-logits'));
muestreoStage.init(document.getElementById('w-muestreo'), { onResample: refreshDerived });
bucleStage.init(document.getElementById('w-bucle'), { onAccept: acceptAndRepeat });
entrenamientoStage.init(document.getElementById('w-entrenamiento'));
limitesStage.init(document.getElementById('w-limites'));
presenter.init();

// ─── Carga del modelo (tokenizer primero, ONNX en background) ───

let loadPromise = null;
let tokenizerReadyResolve;
const tokenizerReady = new Promise(r => { tokenizerReadyResolve = r; });

function ensureModelLoading() {
  if (loadPromise) return loadPromise;
  modelStatus.textContent = 'cargando tokenizer…';
  loadPromise = models.loadModel(MODEL_ID, (p) => {
    if (p.phase === 'model' || p.phase === 'ready') tokenizerReadyResolve();
    if (p.phase === 'model') {
      const pct = p.progress ? ` ${Math.round(p.progress)}%` : '';
      modelStatus.textContent = `descargando modelo${pct}`;
      logitsStage.setProgress(`Descargando el modelo (una sola vez)…${pct}`, p.progress);
    }
    if (p.phase === 'ready') {
      modelStatus.textContent = `${state.modelConfig.name} listo`;
      logitsStage.setProgress('Modelo listo, calculando…', 100);
    }
  }).then((cfg) => {
    state.modelReady = true;
    tokenizerReadyResolve();
    return cfg;
  });
  loadPromise.catch(err => {
    modelStatus.textContent = 'error al cargar el modelo';
    logitsStage.setProgress('No se pudo descargar el modelo. Revisa tu conexión y recarga la página.', 0);
    console.error('[main] error cargando modelo:', err);
  });
  return loadPromise;
}

// ─── Scroll con fallback (algunos entornos no animan el smooth scroll) ───

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth' });
  setTimeout(() => {
    // Si el smooth no llegó (animaciones deshabilitadas o canceladas
    // por re-renders/scroll anchoring), saltar directo al destino.
    if (Math.abs(el.getBoundingClientRect().top) > 120) {
      el.scrollIntoView({ behavior: 'instant' });
    }
  }, 700);
}

// ─── Ciclo del pipeline ───

let runSeq = 0;

async function runCycle(text, { scrollToTokens = false } = {}) {
  const seq = ++runSeq;
  state.text = text;
  ensureModelLoading();
  if (scrollToTokens) {
    // Antes de actualizar el DOM: los re-renders async cancelan el smooth scroll
    scrollToSection('st-tokens');
  }

  // Parte 1: etapas que solo necesitan el tokenizer + pesos por HTTP Range
  await tokenizerReady;
  if (seq !== runSeq) return;
  const { tokens } = models.tokenize(text);
  state.tokens = tokens;
  updateContextBar();
  tokensStage.update(state);
  embeddingsStage.update(state);
  posicionStage.update(state);
  atencionStage.update(state);

  // Parte 2: etapas que necesitan el forward pass del modelo completo
  try {
    await loadPromise;
  } catch {
    return; // el error ya se mostró en logits
  }
  if (seq !== runSeq) return;
  const { predictions } = await pipeline.run(text);
  if (seq !== runSeq) return;
  state.predictions = predictions;
  logitsStage.update(state);
  muestreoStage.update(state);
  bucleStage.update(state);
  limitesStage.update(state);
}

/** Re-deriva predicciones de los mismos logits (sliders, re-muestreo). */
function refreshDerived() {
  if (!pipeline.hasResults()) return;
  state.predictions = pipeline.recomputePredictions();
  logitsStage.update(state);
  muestreoStage.update(state);
  bucleStage.update(state);
}

config.onChange(refreshDerived);

/** §6: añadir el token elegido y repetir el ciclo. */
async function acceptAndRepeat() {
  const sampled = state.predictions?.find(p => p.isSampled);
  const res = await pipeline.generateMore();
  if (!res) return;
  state.history.push({ word: res.topWord, prob: sampled ? sampled.prob : 0, cycle: state.cycle });
  state.cycle += 1;
  await runCycle(res.newText, { scrollToTokens: true });
}

// ─── Barra de contexto sticky ───

function updateContextBar() {
  const genPart = state.text.slice(state.baseText.length);
  contextText.innerHTML =
    `<bdi>${esc(state.baseText)}<span class="gen">${esc(genPart)}</span><span class="gen">▌</span></bdi>`;
  cycleLabel.textContent = `ciclo ${state.cycle}`;
}

// ─── Hero ───

document.getElementById('btn-start').addEventListener('click', () => startJourney());
promptInput.addEventListener('keydown', (e) => {
  // isComposing: no disparar a mitad de composición (IME, teclas muertas)
  if (e.key === 'Enter' && !e.isComposing) startJourney();
});

document.querySelectorAll('#hero-examples .chip').forEach(chip => {
  chip.addEventListener('click', () => {
    promptInput.value = chip.dataset.prompt;
    startJourney();
  });
});

function startJourney({ scroll = true } = {}) {
  const text = promptInput.value.trim();
  if (!text) return;
  state.baseText = text;
  state.history = [];
  state.cycle = 1;
  state.predictions = null;
  contextBar.hidden = false;
  // Permalink: el prompt vive en la URL para poder compartir el recorrido
  const url = new URL(location.href);
  url.searchParams.set('p', text);
  history.replaceState(null, '', url);
  // En modo presentación el scroll lo dicta el avance de pasos, no el inicio
  runCycle(text, { scrollToTokens: scroll && !presenter.isActive() });
}

// Llegada por permalink: precargar el prompt y arrancar sin scroll
const sharedPrompt = new URLSearchParams(location.search).get('p');
if (sharedPrompt && sharedPrompt.trim()) {
  promptInput.value = sharedPrompt.trim();
  startJourney({ scroll: false });
}

// ─── Rail de progreso ───

const railDots = Array.from(document.querySelectorAll('.rail__dot'));

railDots.forEach(dot => {
  dot.addEventListener('click', () => {
    scrollToSection(dot.dataset.target);
  });
});

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const id = entry.target.id;
    railDots.forEach(dot => dot.classList.toggle('is-active', dot.dataset.target === id));
  });
}, { rootMargin: '-45% 0px -45% 0px' });

document.querySelectorAll('.stage').forEach(s => sectionObserver.observe(s));
