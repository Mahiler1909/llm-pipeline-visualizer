/**
 * §3 Posición — el mismo token en posiciones distintas entra distinto:
 * entrada = wte[token] + wpe[posición], con filas REALES de ambas matrices
 * (HTTP Range sobre el safetensors original, igual que embeddings).
 */

import * as models from '../models.js';
import * as weights from '../weights.js';
import { seededRandom } from '../utils.js';
import { esc, tokenLabel } from './shared.js';

const SHOW_DIMS = 48;
const POS_CAP = 8;        // posiciones del prompt ofrecidas como chips
const EXTRA_POS = [100, 1000]; // posiciones lejanas, para ver que existen

let root, tokenChipsEl, posChipsEl, badgeEl, captionEl;
let rowEls = {}; // { wte, wpe, sum } → { svg, label }
let selectedTok = -1;
let selectedPos = 0;
let reqId = 0;
let lastState = null;

export function init(rootEl) {
  root = rootEl;
  root.innerHTML = `
    <p class="sim-title" style="margin-top:0">elige un token del texto…</p>
    <div class="emb-chips" id="pos-token-chips"></div>
    <p class="sim-title">…y una posición</p>
    <div class="emb-chips" id="pos-pos-chips"></div>
    <div class="emb-head">
      <span></span>
      <span class="badge" id="pos-badge"></span>
    </div>
    <div class="pos-rows">
      ${['wte', 'wpe', 'sum'].map(k => `
        <div class="pos-row">
          <div class="pos-row__label mono" id="pos-label-${k}"></div>
          <svg id="pos-svg-${k}" viewBox="0 0 480 64" role="img"></svg>
        </div>`).join('')}
    </div>
    <p class="widget-caption" id="pos-caption">Pasa el cursor sobre una barra.</p>`;

  tokenChipsEl = root.querySelector('#pos-token-chips');
  posChipsEl = root.querySelector('#pos-pos-chips');
  badgeEl = root.querySelector('#pos-badge');
  captionEl = root.querySelector('#pos-caption');
  rowEls = {
    wte: { svg: root.querySelector('#pos-svg-wte'), label: root.querySelector('#pos-label-wte') },
    wpe: { svg: root.querySelector('#pos-svg-wpe'), label: root.querySelector('#pos-label-wpe') },
    sum: { svg: root.querySelector('#pos-svg-sum'), label: root.querySelector('#pos-label-sum') },
  };
}

export function update(state) {
  if (!state.tokens.length) return;
  lastState = state;
  selectedTok = state.tokens.length - 1;
  selectedPos = selectedTok;
  renderChips();
  renderRows();
}

function renderChips() {
  const tokens = lastState.tokens;
  tokenChipsEl.innerHTML = tokens.map((t, i) =>
    `<button class="chip ${i === selectedTok ? 'is-active' : ''}" data-i="${i}">${esc(tokenLabel(t.text))}</button>`
  ).join('');
  tokenChipsEl.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedTok = Number(btn.dataset.i);
      selectedPos = selectedTok; // su posición real, luego se puede variar
      renderChips();
      renderRows();
    });
  });

  const positions = tokens.slice(0, POS_CAP).map((_, i) => i);
  for (const p of EXTRA_POS) if (!positions.includes(p)) positions.push(p);
  posChipsEl.innerHTML = positions.map(p =>
    `<button class="chip ${p === selectedPos ? 'is-active' : ''}" data-p="${p}">pos ${p}</button>`
  ).join('');
  posChipsEl.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedPos = Number(btn.dataset.p);
      renderChips();
      renderRows();
    });
  });
}

/** wpe simulada (determinista) cuando no hay red. */
function fakePositional(position, dims) {
  const rng = seededRandom(position * 7717 + 13);
  const vec = new Float32Array(dims);
  for (let i = 0; i < dims; i++) vec[i] = (rng() * 2 - 1) * 0.1;
  return vec;
}

async function renderRows() {
  const myReq = ++reqId;
  const t = lastState.tokens[selectedTok];
  const dims = lastState.modelConfig.hidden_dim;
  badgeEl.textContent = 'cargando…';
  badgeEl.className = 'badge';

  let wte, wpe, isReal = true;
  try {
    [wte, wpe] = await Promise.all([
      weights.getEmbeddingRow(lastState.modelId, t.id),
      weights.getPositionalRow(lastState.modelId, selectedPos),
    ]);
  } catch {
    isReal = false;
    wte = models.getEmbeddingVector(t.id, dims);
    wpe = fakePositional(selectedPos, dims);
  }
  if (myReq !== reqId) return;

  const sum = new Float32Array(SHOW_DIMS);
  for (let i = 0; i < SHOW_DIMS; i++) sum[i] = wte[i] + wpe[i];

  badgeEl.textContent = isReal ? 'pesos reales · HuggingFace' : 'simulado (sin conexión)';
  badgeEl.className = isReal ? 'badge badge--real' : 'badge';

  rowEls.wte.label.textContent = `wte["${tokenLabel(t.text)}"]  · significado (fija)`;
  rowEls.wpe.label.textContent = `wpe[${selectedPos}]  · posición`;
  rowEls.sum.label.textContent = `suma  · lo que entra al transformer`;

  drawStrip(rowEls.wte.svg, wte, 'wte');
  drawStrip(rowEls.wpe.svg, wpe, 'wpe');
  drawStrip(rowEls.sum.svg, sum, 'sum');
  captionEl.textContent =
    `mismo token, posición ${selectedPos}: la fila wpe cambia y la suma con ella · ${SHOW_DIMS} de ${dims} dims`;
}

function drawStrip(svg, vector, kind) {
  const W = 480, H = 64, mid = H / 2;
  const slice = Array.from(vector.slice(0, SHOW_DIMS));
  const maxAbs = Math.max(...slice.map(Math.abs), 1e-6);
  const barW = W / SHOW_DIMS;

  svg.innerHTML =
    `<line x1="0" y1="${mid}" x2="${W}" y2="${mid}" stroke="var(--hairline)" stroke-width="1"/>` +
    slice.map((v, d) => {
      const h = Math.abs(v) / maxAbs * (mid - 4);
      const y = v >= 0 ? mid - h : mid;
      const color = v >= 0 ? 'var(--accent)' : 'var(--blue)';
      return `<rect class="pos-bar" data-d="${d}" data-v="${v.toFixed(4)}" data-k="${kind}"
        x="${(d * barW + 1).toFixed(1)}" y="${y.toFixed(1)}"
        width="${(barW - 2).toFixed(1)}" height="${Math.max(h, 0.5).toFixed(1)}"
        fill="${color}" opacity="0.8"/>`;
    }).join('');

  svg.querySelectorAll('.pos-bar').forEach(bar => {
    bar.addEventListener('mouseenter', () => {
      captionEl.textContent = `${bar.dataset.k}, dim ${bar.dataset.d}: ${bar.dataset.v}`;
    });
  });
}
