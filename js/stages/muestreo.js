/**
 * §5 Muestreo — la ruleta de candidatos + top-k / top-p / greedy.
 * Los sliders se construyen una sola vez en init.
 */

import * as pipeline from '../pipeline.js';
import * as config from '../config.js';
import { esc, tokenLabel, pastelBg, fmtPct } from './shared.js';

let root, waitEl, contentEl, ruletaEl, captionEl, cutListEl;
let topkVal, toppVal;
let onResample = null;
let flashNext = false;

export function init(rootEl, callbacks) {
  root = rootEl;
  onResample = callbacks.onResample;
  root.innerHTML = `
    <div class="bucle-wait" id="mues-wait">Esperando los logits del paso anterior…</div>
    <div id="mues-content" hidden>
      <div class="ruleta" id="ruleta"></div>
      <p class="widget-caption" id="mues-caption"></p>
      <div class="cut-list" id="cut-list"></div>
      <div class="ctrl">
        <div class="ctrl__row"><span>Top-K (máx. candidatos)</span><span class="val" id="topk-val">${config.get('topK')}</span></div>
        <input type="range" id="topk-slider" min="1" max="50" step="1" value="${config.get('topK')}">
        <div class="ctrl__row" style="margin-top:1.2rem"><span>Top-P (núcleo de probabilidad)</span><span class="val" id="topp-val">${config.get('topP').toFixed(2)}</span></div>
        <input type="range" id="topp-slider" min="0.1" max="1" step="0.05" value="${config.get('topP')}">
      </div>
      <div class="sample-actions">
        <button class="btn" id="btn-sample">🎲 Muestrear de nuevo</button>
        <label class="toggle">
          <input type="checkbox" id="greedy-toggle" ${config.get('greedy') ? 'checked' : ''}>
          <span>Greedy: siempre el más probable</span>
        </label>
      </div>
    </div>`;

  waitEl = root.querySelector('#mues-wait');
  contentEl = root.querySelector('#mues-content');
  ruletaEl = root.querySelector('#ruleta');
  captionEl = root.querySelector('#mues-caption');
  cutListEl = root.querySelector('#cut-list');
  topkVal = root.querySelector('#topk-val');
  toppVal = root.querySelector('#topp-val');

  const topkSlider = root.querySelector('#topk-slider');
  topkSlider.addEventListener('input', () => {
    config.set('topK', parseInt(topkSlider.value, 10));
  });

  const toppSlider = root.querySelector('#topp-slider');
  toppSlider.addEventListener('input', () => {
    config.set('topP', parseFloat(toppSlider.value));
  });

  root.querySelector('#greedy-toggle').addEventListener('change', (e) => {
    config.set('greedy', e.target.checked);
  });

  root.querySelector('#btn-sample').addEventListener('click', () => {
    flashNext = true;
    onResample?.();
  });
}

export function update(state) {
  if (!state.predictions) return;
  waitEl.hidden = true;
  contentEl.hidden = false;
  topkVal.textContent = config.get('topK');
  toppVal.textContent = config.get('topP').toFixed(2);

  const nucleus = state.predictions.filter(p => p.inNucleus);
  ruletaEl.innerHTML = nucleus.map((p, i) => `
    <div class="ruleta__seg ${p.isSampled ? 'is-sampled' : ''} ${p.isSampled && flashNext ? 'flash' : ''}"
         style="width:${(p.nucleusProb * 100).toFixed(2)}%; background:${pastelBg(i)}"
         title="${esc(tokenLabel(p.word))} · ${fmtPct(p.nucleusProb)}">
      ${p.nucleusProb > 0.07 ? esc(tokenLabel(p.word)) : ''}
    </div>`).join('');
  flashNext = false;

  const sampled = state.predictions.find(p => p.isSampled);
  captionEl.textContent = sampled
    ? `★ ganó "${tokenLabel(sampled.word)}" con ${fmtPct(sampled.nucleusProb)} del núcleo · ${nucleus.length} candidato${nucleus.length === 1 ? '' : 's'} en juego`
    : '';

  // Candidatos eliminados (de los 15 más probables)
  const dist = pipeline.getDistribution(15);
  const cut = dist.items.filter(it => it.cutByTopK || !it.inNucleus);
  cutListEl.innerHTML = cut.length
    ? 'eliminados: ' + cut.map(it =>
        `<span class="cut" title="${it.cutByTopK ? 'eliminado por top-k' : 'fuera del núcleo top-p'}">${esc(tokenLabel(it.word))}<span class="cut-reason"> ${it.cutByTopK ? 'k' : 'p'}</span></span>`
      ).join('')
      + ` <span style="font-size:11px">(k = corte top-k · p = fuera del núcleo)</span>`
    : 'con esta configuración ningún candidato del top-15 queda eliminado';
}
