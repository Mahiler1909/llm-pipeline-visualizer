/**
 * §Antes de empezar — ¿Qué es un LLM?
 * Widget A: mapa clicable del pipeline (navega a cada sección).
 * Widget B: escala logarítmica de tamaños de modelos.
 */

import { esc } from './shared.js';

const PHASES = [
  { target: 'st-tokens', num: '1', name: 'Tokens', sub: 'texto → IDs' },
  { target: 'st-embeddings', num: '2', name: 'Embeddings', sub: 'IDs → vectores' },
  { target: 'st-posicion', num: '3', name: 'Posición', sub: '+ orden' },
  { target: 'st-atencion', num: '4', name: 'Atención', sub: 'entra el contexto' },
  { target: 'st-bloque', num: '5', name: 'Bloque transformer', sub: 'atención + FFN' },
  { target: 'st-logits', num: '6', name: 'Logits', sub: 'puntaje por token' },
  { target: 'st-muestreo', num: '7', name: 'Muestreo', sub: 'elegir uno' },
  { target: 'st-bucle', num: '↺', name: 'El bucle', sub: 'y vuelta a empezar' },
];

// Tamaños en parámetros (escala log10). Frontera: estimación pública.
const SIZES = [
  { name: 'DistilGPT-2', params: 82e6, label: '82M', here: true },
  { name: 'GPT-2 XL', params: 1.5e9, label: '1.5B' },
  { name: 'GPT-3', params: 175e9, label: '175B' },
  { name: 'Llama 3', params: 405e9, label: '405B' },
  { name: 'Frontera (est.)', params: 2e12, label: '~2T' },
];

let captionEl;

export function init(rootEl, { onNavigate, modelConfig }) {
  const blockLabel = `× ${modelConfig.layers}`;

  const LOG_MIN = 7.5, LOG_MAX = 12.4;
  const width = (p) =>
    Math.round(((Math.log10(p) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100);

  rootEl.innerHTML = `
    <div class="pmap" id="pmap">
      ${PHASES.map(ph => `
        <button class="pmap__item" data-target="${ph.target}">
          <span class="pmap__num mono">${ph.num}</span>
          <span class="pmap__name">${esc(ph.name)}${ph.target === 'st-bloque' ? ` <span class="mono pmap__mult">${blockLabel}</span>` : ''}</span>
          <span class="pmap__sub mono">${esc(ph.sub)}</span>
        </button>`).join('')}
    </div>
    <p class="widget-caption">el camino que vas a recorrer · clic en una etapa para saltar</p>
    <div class="scale-section">
      <p class="sim-title">¿Qué tan "large"? · parámetros (escala logarítmica)</p>
      ${SIZES.map(s => `
        <div class="scale-row ${s.here ? 'scale-row--here' : ''}" data-name="${esc(s.name)}" data-params="${s.params}">
          <span class="scale-name">${esc(s.name)}${s.here ? ' ·&nbsp;<b>este</b>' : ''}</span>
          <div class="scale-track"><div class="scale-fill" style="width:${width(s.params)}%"></div></div>
          <span class="scale-label mono">${s.label}</span>
        </div>`).join('')}
      <p class="widget-caption" id="llm-caption">Pasa el cursor sobre un modelo.</p>
    </div>`;

  captionEl = rootEl.querySelector('#llm-caption');

  rootEl.querySelectorAll('.pmap__item').forEach(btn => {
    btn.addEventListener('click', () => onNavigate?.(btn.dataset.target));
  });

  const base = SIZES[0].params;
  rootEl.querySelectorAll('.scale-row').forEach(row => {
    row.addEventListener('mouseenter', () => {
      const p = Number(row.dataset.params);
      const ratio = p / base;
      captionEl.textContent = ratio === 1
        ? `${row.dataset.name}: el modelo que corre en esta página`
        : `${row.dataset.name}: ~${Math.round(ratio).toLocaleString('es')}× el modelo de esta página`;
    });
  });
  rootEl.querySelector('.scale-section').addEventListener('mouseleave', () => {
    captionEl.textContent = 'Pasa el cursor sobre un modelo.';
  });
}

export function update() { /* estático: nada que actualizar por ciclo */ }
