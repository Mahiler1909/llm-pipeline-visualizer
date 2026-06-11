/**
 * §Para terminar — Límites: demo de alucinación en vivo.
 * Corre su propio forward con models.* (sin tocar el caché de pipeline.js,
 * para no pisar los logits del viaje principal).
 */

import * as models from '../models.js';
import { esc, tokenLabel } from './shared.js';

// Elegidos probando contra DistilGPT-2 real: contrastan hechos frecuentes
// (alta confianza) con invenciones servidas por la misma mecánica.
const PRESETS = [
  'Barack Obama was the president of the',
  'Steve Jobs was the founder of',
  'The 2030 World Cup was won by',
  'Albert Einstein was born in',
];

const TOP_N = 5;

let root, waitEl, contentEl, chipsEl, inputEl, promptEl, rowsEl, captionEl;
let ready = false;
let busy = false;

export function init(rootEl) {
  root = rootEl;
  root.innerHTML = `
    <div class="bucle-wait" id="hallu-wait">Esperando al modelo (se activa al completar el viaje)…</div>
    <div id="hallu-content" hidden>
      <div class="emb-chips" id="hallu-chips">
        ${PRESETS.map(p => `<button class="chip" data-p="${esc(p)}">${esc(p)}</button>`).join('')}
      </div>
      <input type="text" class="hero__input mono hallu-input" id="hallu-input"
             placeholder="…o escribe el tuyo (en inglés, sin terminar la frase)" autocomplete="off" spellcheck="false">
      <p class="sim-title" id="hallu-prompt" hidden></p>
      <div class="hallu-rows" id="hallu-rows"></div>
      <p class="widget-caption" id="hallu-caption">Elige un prompt: hechos futuros, discutibles o capitales con trampa.</p>
    </div>`;

  waitEl = root.querySelector('#hallu-wait');
  contentEl = root.querySelector('#hallu-content');
  chipsEl = root.querySelector('#hallu-chips');
  inputEl = root.querySelector('#hallu-input');
  promptEl = root.querySelector('#hallu-prompt');
  rowsEl = root.querySelector('#hallu-rows');
  captionEl = root.querySelector('#hallu-caption');

  chipsEl.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => run(btn.dataset.p));
  });
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing && inputEl.value.trim()) run(inputEl.value.trim());
  });
}

export function update(state) {
  if (!state.modelReady || ready) return;
  ready = true;
  waitEl.hidden = true;
  contentEl.hidden = false;
}

async function run(prompt) {
  if (!ready || busy) return;
  busy = true;
  promptEl.hidden = false;
  promptEl.textContent = `"${prompt}" + …`;
  rowsEl.innerHTML = '';
  captionEl.textContent = 'calculando…';

  try {
    const { encoded } = models.tokenize(prompt);
    const { logits } = await models.forward(encoded);

    // softmax real (T=1) sobre todo el vocabulario
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > maxVal) maxVal = logits[i];
    let sum = 0;
    for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - maxVal);

    const indexed = [];
    for (let i = 0; i < logits.length; i++) indexed.push({ logit: logits[i], idx: i });
    indexed.sort((a, b) => b.logit - a.logit);
    const top = indexed.slice(0, TOP_N).map(it => ({
      word: models.decodeToken(it.idx),
      prob: Math.exp(it.logit - maxVal) / sum,
    }));

    const maxProb = top[0].prob;
    rowsEl.innerHTML = top.map((t, i) => `
      <div class="hallu-row ${i === 0 ? 'is-top' : ''}">
        <span class="word mono">${esc(tokenLabel(t.word))}</span>
        <div class="bar-track"><div class="bar" style="width:${(t.prob / maxProb * 100).toFixed(1)}%"></div></div>
        <span class="pct mono">${(t.prob * 100).toFixed(1)}%</span>
      </div>`).join('');

    captionEl.textContent =
      `el modelo afirma "${tokenLabel(top[0].word)}" con ${(top[0].prob * 100).toFixed(1)}% — ` +
      `la mecánica es la misma sea cierto o inventado`;
  } catch (err) {
    captionEl.textContent = 'No se pudo calcular. Intenta de nuevo.';
    console.error('[limites] forward fallo:', err);
  } finally {
    busy = false;
  }
}
