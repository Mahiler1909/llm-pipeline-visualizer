/**
 * §4 Logits — top-15 por logit crudo + slider de temperatura.
 * El slider se construye una sola vez en init (re-renderizarlo rompería el drag).
 */

import * as pipeline from '../pipeline.js';
import * as config from '../config.js';
import { esc, tokenLabel } from './shared.js';

const TOP_N = 15;

let root, waitEl, waitTextEl, waitFillEl, contentEl, rowsEl, captionEl, tempVal;

export function init(rootEl) {
  root = rootEl;
  root.innerHTML = `
    <div class="inline-progress" id="logits-wait">
      <span id="logits-wait-text">El modelo se está descargando en segundo plano (una sola vez)…</span>
      <div class="track"><div class="fill" id="logits-wait-fill"></div></div>
    </div>
    <div id="logits-content" hidden>
      <div class="logit-rows" id="logit-rows"></div>
      <p class="widget-caption" id="logit-caption"></p>
      <div class="ctrl">
        <div class="ctrl__row"><span>Temperatura</span><span class="val" id="temp-val">1.00</span></div>
        <input type="range" id="temp-slider" min="0.1" max="2" step="0.05" value="${config.get('temperature')}">
        <div class="ctrl__ends"><span>T→0: determinista</span><span>T alta: caos</span></div>
      </div>
    </div>`;

  waitEl = root.querySelector('#logits-wait');
  waitTextEl = root.querySelector('#logits-wait-text');
  waitFillEl = root.querySelector('#logits-wait-fill');
  contentEl = root.querySelector('#logits-content');
  rowsEl = root.querySelector('#logit-rows');
  captionEl = root.querySelector('#logit-caption');
  tempVal = root.querySelector('#temp-val');

  const slider = root.querySelector('#temp-slider');
  slider.addEventListener('input', () => {
    config.set('temperature', parseFloat(slider.value));
  });
}

/** Progreso de descarga del modelo (lo alimenta main.js) */
export function setProgress(message, pct) {
  waitTextEl.textContent = message;
  if (pct != null) waitFillEl.style.width = `${Math.round(pct)}%`;
}

export function update(state) {
  if (!state.predictions) return;
  const dist = pipeline.getDistribution(TOP_N);
  if (!dist) return;

  waitEl.hidden = true;
  contentEl.hidden = false;
  tempVal.textContent = dist.temperature.toFixed(2);

  const maxProb = Math.max(...dist.items.map(it => it.prob), 1e-9);
  rowsEl.innerHTML = dist.items.map(it => `
    <div class="logit-row">
      <span class="word" title="${esc(tokenLabel(it.word))}">${esc(tokenLabel(it.word))}</span>
      <span class="logit-val">${it.rawLogit.toFixed(1)}</span>
      <div class="bar-track"><div class="bar" style="width:${(it.prob / maxProb * 100).toFixed(2)}%"></div></div>
      <span class="pct">${(it.prob * 100).toFixed(1)}%</span>
    </div>`).join('');

  captionEl.textContent =
    `los ${TOP_N} logits más altos de ${state.modelConfig.vocab_size.toLocaleString('es')} · ` +
    `columna gris = logit crudo (z) · % = probabilidad real softmax(z/T) sobre todo el vocabulario`;
}
