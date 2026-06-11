/**
 * §El pipeline — las piezas, juntas: mapa clicable de la cadena completa.
 * Cada etapa navega a su sección (conceptos ya vistos o pasos por venir).
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

export function init(rootEl, { onNavigate, modelConfig }) {
  const blockLabel = `× ${modelConfig.layers}`;

  rootEl.innerHTML = `
    <div class="pmap" id="pmap">
      ${PHASES.map(ph => `
        <button class="pmap__item" data-target="${ph.target}">
          <span class="pmap__num mono">${ph.num}</span>
          <span class="pmap__name">${esc(ph.name)}${ph.target === 'st-bloque' ? ` <span class="mono pmap__mult">${blockLabel}</span>` : ''}</span>
          <span class="pmap__sub mono">${esc(ph.sub)}</span>
        </button>`).join('')}
    </div>
    <p class="widget-caption">la cadena completa de cada predicción · clic en una etapa para saltar</p>`;

  rootEl.querySelectorAll('.pmap__item').forEach(btn => {
    btn.addEventListener('click', () => onNavigate?.(btn.dataset.target));
  });
}

export function update() { /* estático: nada que actualizar por ciclo */ }
