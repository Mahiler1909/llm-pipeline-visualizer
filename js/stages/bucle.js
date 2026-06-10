/**
 * §6 El bucle — la salida se vuelve entrada (generación autorregresiva).
 */

import { getClosing } from '../content.js';
import { esc, tokenLabel, fmtPct } from './shared.js';

let root, waitEl, contentEl, textEl, historyEl, btnEl, closingEl;
let onAccept = null;

export function init(rootEl, callbacks) {
  root = rootEl;
  onAccept = callbacks.onAccept;
  root.innerHTML = `
    <div class="bucle-wait" id="bucle-wait">Esperando la predicción del modelo…</div>
    <div id="bucle-content" hidden>
      <p class="bucle-text" id="bucle-text"></p>
      <div class="bucle-history" id="bucle-history"></div>
      <div class="bucle-actions">
        <button class="btn btn--primary" id="btn-accept"></button>
      </div>
      <p class="widget-caption" id="bucle-closing" style="margin-top:2rem">${esc(getClosing())}</p>
    </div>`;

  waitEl = root.querySelector('#bucle-wait');
  contentEl = root.querySelector('#bucle-content');
  textEl = root.querySelector('#bucle-text');
  historyEl = root.querySelector('#bucle-history');
  btnEl = root.querySelector('#btn-accept');

  btnEl.addEventListener('click', () => {
    btnEl.disabled = true;
    onAccept?.();
  });
}

export function update(state) {
  if (!state.predictions) return;
  waitEl.hidden = true;
  contentEl.hidden = false;
  btnEl.disabled = false;

  const sampled = state.predictions.find(p => p.isSampled) || state.predictions[0];
  const genPart = state.text.slice(state.baseText.length);

  textEl.innerHTML =
    esc(state.baseText) +
    `<span class="gen">${esc(genPart)}</span>` +
    `<span class="new">${esc(sampled.word)}</span>`;

  btnEl.textContent = `Añadir "${tokenLabel(sampled.word)}" y repetir el ciclo ↑`;

  if (state.history.length) {
    historyEl.innerHTML =
      `<div class="bucle-history__title">tokens generados en esta sesión</div>` +
      state.history.map(h =>
        `"${esc(tokenLabel(h.word))}" · ${fmtPct(h.prob)} · ciclo ${h.cycle}`
      ).join('<br>');
  } else {
    historyEl.innerHTML = '';
  }
}
