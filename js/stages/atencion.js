/**
 * §3 Atención — pesos reales de la capa 0, una query a la vez.
 * Descarga lazy de ~7MB cuando la sección entra al viewport.
 */

import * as attention from '../attention.js';
import { esc, tokenLabel } from './shared.js';

const TOKEN_CAP = 12;

let root, progressEl, fillEl, contentEl, stripEl, noteEl, sentenceEl, headsEl;
let queryIdx = -1;       // índice dentro de la ventana mostrada
let headIdx = -1;        // -1 = promedio
let attnResult = null;
let windowTokens = [];
let lastState = null;
let loadStarted = false;
let computeReq = 0;

export function init(rootEl, sectionEl) {
  root = rootEl;
  root.innerHTML = `
    <div class="inline-progress" id="attn-progress">
      <span id="attn-progress-text">Los pesos reales de la capa 1 (~7&nbsp;MB) se descargarán al llegar aquí.</span>
      <div class="track"><div class="fill" id="attn-progress-fill"></div></div>
    </div>
    <div id="attn-content" hidden>
      <div class="attn-strip" id="attn-strip"></div>
      <p class="widget-caption" id="attn-note"></p>
      <p class="attn-sentence" id="attn-sentence"></p>
      <div class="attn-heads" id="attn-heads"></div>
    </div>`;

  progressEl = root.querySelector('#attn-progress');
  fillEl = root.querySelector('#attn-progress-fill');
  contentEl = root.querySelector('#attn-content');
  stripEl = root.querySelector('#attn-strip');
  noteEl = root.querySelector('#attn-note');
  sentenceEl = root.querySelector('#attn-sentence');
  headsEl = root.querySelector('#attn-heads');

  // Descarga lazy al entrar al viewport
  const io = new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting) && lastState && !loadStarted) {
      io.disconnect();
      startLoad();
    }
  }, { rootMargin: '200px' });
  io.observe(sectionEl);
}

export function update(state) {
  lastState = state;
  attnResult = null; // el texto cambió: recalcular
  if (attention.isLoaded(state.modelId) || loadStarted) {
    compute();
  }
}

async function startLoad() {
  loadStarted = true;
  const textEl = root.querySelector('#attn-progress-text');
  textEl.textContent = 'Descargando pesos reales de la capa 1…';
  try {
    await attention.loadLayer0(lastState.modelId, ({ loaded, total }) => {
      if (total) fillEl.style.width = `${Math.round(loaded / total * 100)}%`;
    });
    await compute();
  } catch (err) {
    textEl.textContent = 'No se pudieron descargar los pesos de atención (¿sin conexión?). ' +
      'Esta sección necesita internet para mostrar datos reales.';
    console.warn('[atencion]', err);
  }
}

async function compute() {
  if (!lastState) return;
  const myReq = ++computeReq;
  const all = lastState.tokens;
  windowTokens = all.slice(-TOKEN_CAP);
  if (windowTokens.length < 2) {
    contentEl.hidden = false;
    progressEl.hidden = true;
    stripEl.innerHTML = '';
    sentenceEl.textContent = 'Escribe al menos dos tokens para ver la atención.';
    return;
  }
  queryIdx = windowTokens.length - 1;

  try {
    const result = await attention.computeAttention(
      lastState.modelId,
      windowTokens.map(t => t.id),
      lastState.modelConfig.heads
    );
    if (myReq !== computeReq) return;
    attnResult = result;
    progressEl.hidden = true;
    contentEl.hidden = false;
    renderHeads();
    render();
  } catch (err) {
    progressEl.hidden = false;
    root.querySelector('#attn-progress-text').textContent =
      'No se pudo calcular la atención real (¿sin conexión?).';
    console.warn('[atencion]', err);
  }
}

function weightAt(i, j) {
  return headIdx < 0 ? attnResult.avg(i, j) : attnResult.get(headIdx, i, j);
}

function render() {
  if (!attnResult) return;
  const n = windowTokens.length;

  // pesos de la fila query
  const ws = [];
  for (let j = 0; j <= queryIdx; j++) ws.push({ j, w: weightAt(queryIdx, j) });
  const top3 = new Set([...ws].sort((a, b) => b.w - a.w).slice(0, 3).map(x => x.j));

  stripEl.innerHTML = windowTokens.map((t, j) => {
    if (j > queryIdx) {
      return `<span class="attn-token" data-j="${j}" style="opacity:0.35">${esc(t.text)}</span>`;
    }
    const w = weightAt(queryIdx, j);
    const isQuery = j === queryIdx;
    const alpha = Math.min(1, w * 1.8);
    const pct = top3.has(j) && !isQuery
      ? `<span class="attn-pct">${(w * 100).toFixed(0)}%</span>` : '';
    return `<span class="attn-token ${isQuery ? 'is-query' : ''}" data-j="${j}"
      style="background:rgba(74,111,165,${alpha.toFixed(3)}); ${alpha > 0.55 ? 'color:#fff;' : ''}"
      >${esc(t.text)}${pct}</span>`;
  }).join('');

  stripEl.querySelectorAll('.attn-token').forEach(span => {
    span.addEventListener('click', () => {
      queryIdx = Number(span.dataset.j);
      render();
    });
  });

  const truncated = lastState.tokens.length > n ? ` · mostrando los últimos ${n} tokens` : '';
  noteEl.textContent = `capa 1 · ${headIdx < 0 ? 'promedio de las ' + attnResult.heads + ' cabezas' : 'cabeza ' + (headIdx + 1)}` +
    ` · clic en un token para usarlo como punto de vista${truncated}`;

  // Frase dinámica con los 2 más atendidos (sin contar el propio token)
  const others = ws.filter(x => x.j !== queryIdx).sort((a, b) => b.w - a.w);
  const q = tokenLabel(windowTokens[queryIdx].text);
  if (others.length === 0) {
    sentenceEl.innerHTML = `El primer token solo puede mirarse a sí mismo.`;
  } else {
    const parts = others.slice(0, 2).map(x =>
      `<span class="w">"${esc(tokenLabel(windowTokens[x.j].text))}"</span> (${(x.w * 100).toFixed(0)}%)`);
    sentenceEl.innerHTML =
      `Desde <span class="w">"${esc(q)}"</span>, el modelo mira sobre todo a ${parts.join(' y a ')}.` +
      ` Solo puede mirar hacia atrás: el futuro está enmascarado.`;
  }
}

function renderHeads() {
  const heads = attnResult.heads;
  let html = `<button class="head-btn ${headIdx < 0 ? 'is-active' : ''}" data-h="-1">Prom</button>`;
  for (let h = 0; h < heads; h++) {
    html += `<button class="head-btn ${headIdx === h ? 'is-active' : ''}" data-h="${h}">${h + 1}</button>`;
  }
  headsEl.innerHTML = html;
  headsEl.querySelectorAll('.head-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      headIdx = Number(btn.dataset.h);
      headsEl.querySelectorAll('.head-btn').forEach(b => b.classList.toggle('is-active', b === btn));
      render();
    });
  });
}
