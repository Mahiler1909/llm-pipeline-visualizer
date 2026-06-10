/**
 * §2 Embeddings — vector real del token + matriz de similitud coseno.
 * Usa weights.getEmbeddingRow directamente (con modelId explícito) para que
 * funcione antes de que el modelo ONNX termine de descargarse.
 */

import * as models from '../models.js';
import * as weights from '../weights.js';
import { cosineSimilarity } from '../utils.js';
import { esc, tokenLabel, pastelBg, pastelInk, divergent } from './shared.js';

const SHOW_DIMS = 48;
const SIM_CAP = 10;

let root, chipsEl, titleEl, badgeEl, svgEl, captionEl, gridEl, simCaptionEl;
let selectedIdx = -1;
let reqId = 0;

export function init(rootEl) {
  root = rootEl;
  root.innerHTML = `
    <div class="emb-chips" id="emb-chips"></div>
    <div class="emb-head">
      <span id="emb-title"></span>
      <span class="badge" id="emb-badge"></span>
    </div>
    <div class="emb-svg-wrap">
      <svg id="emb-svg" viewBox="0 0 480 130" role="img" aria-label="Valores del vector de embedding"></svg>
    </div>
    <p class="widget-caption" id="emb-caption">Pasa el cursor sobre una barra.</p>
    <div class="sim-section">
      <p class="sim-title">Similitud coseno entre los tokens del texto</p>
      <div class="sim-grid" id="sim-grid"></div>
      <p class="widget-caption" id="sim-caption">Pasa el cursor sobre una celda para comparar dos tokens.</p>
    </div>`;

  chipsEl = root.querySelector('#emb-chips');
  titleEl = root.querySelector('#emb-title');
  badgeEl = root.querySelector('#emb-badge');
  svgEl = root.querySelector('#emb-svg');
  captionEl = root.querySelector('#emb-caption');
  gridEl = root.querySelector('#sim-grid');
  simCaptionEl = root.querySelector('#sim-caption');
}

async function fetchVector(modelId, tokenId, dims) {
  try {
    const vector = await weights.getEmbeddingRow(modelId, tokenId);
    return { vector, isReal: true };
  } catch {
    return { vector: models.getEmbeddingVector(tokenId, dims), isReal: false };
  }
}

export function update(state) {
  const { tokens, modelId, modelConfig } = state;
  if (!tokens.length) return;
  selectedIdx = tokens.length - 1;

  renderChips(state);
  renderVector(state);
  renderSimilarity(state);
}

function renderChips(state) {
  chipsEl.innerHTML = state.tokens.map((t, i) =>
    `<button class="chip ${i === selectedIdx ? 'is-active' : ''}" data-i="${i}"
       style="${i === selectedIdx ? '' : `border-color:${pastelInk(i)}40`}">${esc(tokenLabel(t.text))}</button>`
  ).join('');
  chipsEl.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedIdx = Number(btn.dataset.i);
      renderChips(state);
      renderVector(state);
    });
  });
}

async function renderVector(state) {
  const myReq = ++reqId;
  const t = state.tokens[selectedIdx];
  const dims = state.modelConfig.hidden_dim;
  titleEl.textContent = `vector de "${tokenLabel(t.text)}" · ID ${t.id}`;
  badgeEl.textContent = 'cargando…';
  badgeEl.className = 'badge';

  const { vector, isReal } = await fetchVector(state.modelId, t.id, dims);
  if (myReq !== reqId) return; // llegó una petición más nueva

  badgeEl.textContent = isReal ? 'pesos reales · HuggingFace' : 'simulado (sin conexión)';
  badgeEl.className = isReal ? 'badge badge--real' : 'badge';

  const W = 480, H = 130, mid = H / 2;
  const slice = Array.from(vector.slice(0, SHOW_DIMS));
  const maxAbs = Math.max(...slice.map(Math.abs), 1e-6);
  const barW = W / SHOW_DIMS;

  svgEl.innerHTML =
    `<line x1="0" y1="${mid}" x2="${W}" y2="${mid}" stroke="var(--hairline)" stroke-width="1"/>` +
    slice.map((v, d) => {
      const h = Math.abs(v) / maxAbs * (mid - 10);
      const y = v >= 0 ? mid - h : mid;
      const color = v >= 0 ? 'var(--accent)' : 'var(--blue)';
      return `<rect class="emb-bar" data-d="${d}" data-v="${v.toFixed(4)}"
        x="${(d * barW + 1).toFixed(1)}" y="${y.toFixed(1)}"
        width="${(barW - 2).toFixed(1)}" height="${Math.max(h, 0.5).toFixed(1)}"
        fill="${color}" opacity="0.8"/>`;
    }).join('');

  captionEl.textContent = `mostrando ${SHOW_DIMS} de ${dims} dimensiones`;
  svgEl.querySelectorAll('.emb-bar').forEach(bar => {
    bar.addEventListener('mouseenter', () => {
      captionEl.textContent = `dim ${bar.dataset.d}: ${bar.dataset.v}  (de ${dims} dimensiones)`;
    });
  });
  svgEl.addEventListener('mouseleave', () => {
    captionEl.textContent = `mostrando ${SHOW_DIMS} de ${dims} dimensiones`;
  });
}

async function renderSimilarity(state) {
  // Tokens únicos (los últimos SIM_CAP)
  const seen = new Set();
  const unique = [];
  for (const t of state.tokens) {
    if (!seen.has(t.id)) { seen.add(t.id); unique.push(t); }
  }
  const toks = unique.slice(-SIM_CAP);
  if (toks.length < 2) {
    gridEl.innerHTML = '';
    simCaptionEl.textContent = 'Se necesitan al menos 2 tokens distintos.';
    return;
  }

  const dims = state.modelConfig.hidden_dim;
  const results = await Promise.all(toks.map(t => fetchVector(state.modelId, t.id, dims)));
  const vecs = results.map(r => r.vector);
  const allReal = results.every(r => r.isReal);

  const n = toks.length;
  const cellSize = n > 7 ? '34px' : '44px';
  gridEl.style.gridTemplateColumns = `auto repeat(${n}, ${cellSize})`;

  let html = '<div></div>' + toks.map(t =>
    `<div class="sim-label sim-label--col">${esc(tokenLabel(t.text))}</div>`).join('');

  for (let i = 0; i < n; i++) {
    html += `<div class="sim-label">${esc(tokenLabel(toks[i].text))}</div>`;
    for (let j = 0; j < n; j++) {
      const sim = cosineSimilarity(vecs[i], vecs[j]);
      html += `<div class="sim-cell" data-i="${i}" data-j="${j}" data-sim="${sim.toFixed(3)}"
        style="background:${divergent(sim)}"></div>`;
    }
  }
  gridEl.innerHTML = html;

  const base = `${n} tokens · ${allReal ? 'embeddings reales' : 'embeddings simulados'} · diagonal = 1 (token consigo mismo)`;
  simCaptionEl.textContent = base;
  gridEl.querySelectorAll('.sim-cell').forEach(cell => {
    cell.addEventListener('mouseenter', () => {
      const i = Number(cell.dataset.i), j = Number(cell.dataset.j);
      simCaptionEl.textContent =
        `cos("${tokenLabel(toks[i].text)}", "${tokenLabel(toks[j].text)}") = ${cell.dataset.sim}`;
    });
  });
  gridEl.addEventListener('mouseleave', () => { simCaptionEl.textContent = base; });
}
