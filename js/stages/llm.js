/**
 * §Definición — ¿Qué es un LLM?
 * Widget A: la función en grande (P(siguiente token | contexto)).
 * Widget B: escala logarítmica de tamaños de modelos.
 */

import { esc } from './shared.js';

// Tamaños en parámetros (escala log10). Frontera: estimación pública.
const SIZES = [
  { name: 'DistilGPT-2', params: 82e6, label: '82M', here: true },
  { name: 'GPT-2 XL', params: 1.5e9, label: '1.5B' },
  { name: 'GPT-3', params: 175e9, label: '175B' },
  { name: 'Llama 3', params: 405e9, label: '405B' },
  { name: 'Frontera (est.)', params: 2e12, label: '~2T' },
];

let captionEl;

export function init(rootEl) {
  const LOG_MIN = 7.5, LOG_MAX = 12.4;
  const width = (p) =>
    Math.round(((Math.log10(p) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100);

  rootEl.innerHTML = `
    <div class="bigformula mono">P( siguiente token | contexto )</div>
    <p class="widget-caption" style="text-align:center">
      "The cat sat on the …" → ¿mat? ¿sofa? ¿moon? — un puntaje para cada opción
    </p>
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
