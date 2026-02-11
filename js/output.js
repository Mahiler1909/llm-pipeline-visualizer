/**
 * Output step: computes probability distribution over next tokens
 * using softmax with temperature, and renders the results.
 */

import { softmax, seededRandom, hash, fmt, delay, reveal, clearElement } from './utils.js';
import * as config from './config.js';

// Candidate vocabulary for output predictions
const CANDIDATES = {
  es: [
    'mesa', 'silla', 'alfombra', 'cama', 'cocina', 'jardín', 'ventana',
    'puerta', 'suelo', 'techo', 'sofá', 'balcón', 'pared', 'esquina',
    'calle', 'parque', 'sol', 'luna', 'noche', 'mañana',
  ],
  en: [
    'mat', 'floor', 'chair', 'table', 'bed', 'roof', 'fence',
    'garden', 'road', 'hill', 'wall', 'door', 'step', 'bridge',
    'river', 'field', 'tree', 'house', 'rock', 'cloud',
  ],
};

// Context-aware word associations for more realistic results
const CONTEXT_BOOST = {
  'gato': { 'mesa': 2.5, 'alfombra': 2.2, 'sofá': 2.8, 'silla': 2.0, 'cama': 1.8, 'ventana': 1.5 },
  'sentó': { 'silla': 2.5, 'sofá': 2.8, 'suelo': 1.5, 'mesa': 1.8, 'cama': 1.6, 'banco': 1.4 },
  'fox': { 'fence': 2.5, 'hill': 2.0, 'field': 1.8, 'road': 1.5, 'wall': 1.3 },
  'quick': { 'fox': 3.0, 'step': 1.5, 'road': 1.2 },
  'brown': { 'fox': 3.5, 'door': 1.2, 'fence': 1.5, 'floor': 1.3 },
  'inteligencia': { 'artificial': 3.5, 'humana': 2.5 },
  'artificial': { 'crear': 2.0, 'aprender': 2.2, 'pensar': 1.8 },
  'puede': { 'crear': 2.0, 'aprender': 2.5, 'pensar': 2.0, 'hacer': 2.2 },
};

/** Stored state for re-rendering on temperature change */
let lastRenderData = null;

/**
 * Compute logits for candidate words based on context.
 */
function computeLogits(transformerOutput) {
  const tokens = transformerOutput.tokens;
  const hiddenState = transformerOutput.hiddenState;
  const lastHidden = hiddenState[hiddenState.length - 1];

  // Detect language
  const allText = tokens.map(t => t.text.toLowerCase()).join(' ');
  const isSpanish = /[áéíóúñ]/.test(allText) || ['el', 'la', 'en', 'de', 'que', 'se'].some(w => allText.includes(w));
  const candidates = isSpanish ? CANDIDATES.es : CANDIDATES.en;

  // Compute base logits from hidden state similarity
  const logits = candidates.map(word => {
    const rng = seededRandom(hash(word));
    let base = 0;
    for (let d = 0; d < lastHidden.length; d++) {
      base += lastHidden[d] * (rng() * 2 - 1);
    }

    // Add context boost
    let boost = 0;
    for (const token of tokens) {
      const lower = token.text.toLowerCase();
      if (CONTEXT_BOOST[lower] && CONTEXT_BOOST[lower][word]) {
        boost += CONTEXT_BOOST[lower][word];
      }
    }

    return base + boost;
  });

  return { candidates, logits };
}

/**
 * Render the output bars and result.
 */
function renderOutput(container, candidates, logits, temperature) {
  clearElement(container);

  const wrapper = document.createElement('div');
  wrapper.className = 'output-container';
  container.appendChild(wrapper);

  // Formula display
  const formula = document.createElement('div');
  formula.className = 'output-formula visible';
  formula.innerHTML = `P(word) = exp(logit / <em>T=${fmt(temperature, 1)}</em>) / &Sigma; exp(logits / <em>T</em>)`;
  wrapper.appendChild(formula);

  // Compute probabilities
  const probs = softmax(logits, temperature);

  // Sort by probability
  const sorted = candidates
    .map((word, i) => ({ word, prob: probs[i], logit: logits[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 12);

  const maxProb = sorted[0].prob;

  // Bars
  const barsContainer = document.createElement('div');
  barsContainer.className = 'output-bars';
  wrapper.appendChild(barsContainer);

  sorted.forEach((item, idx) => {
    const bar = document.createElement('div');
    bar.className = 'output-bar visible';
    if (idx === 0) bar.classList.add('winner');

    bar.innerHTML = `
      <span class="output-bar__word">${item.word}</span>
      <div class="output-bar__track">
        <div class="output-bar__fill" style="width: 0%"></div>
      </div>
      <span class="output-bar__pct">${fmt(item.prob * 100, 1)}%</span>
    `;
    barsContainer.appendChild(bar);

    // Animate bar fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const fill = bar.querySelector('.output-bar__fill');
        fill.style.width = `${(item.prob / maxProb) * 100}%`;
      });
    });
  });

  // Result
  const result = document.createElement('div');
  result.className = 'output-result';
  result.innerHTML = `
    <span class="output-result__label">Siguiente palabra:</span>
    <span class="output-result__text">"${sorted[0].word}" (${fmt(sorted[0].prob * 100, 1)}%)</span>
  `;
  wrapper.appendChild(result);
  requestAnimationFrame(() => {
    result.classList.add('visible');
  });
}

/**
 * Run the output step.
 * @param {Object} transformerOutput - Output from transformer step.
 * @param {HTMLElement} container - DOM element to render into.
 */
export async function run(transformerOutput, container) {
  clearElement(container);

  const { candidates, logits } = computeLogits(transformerOutput);

  // Store for re-rendering on temperature change
  lastRenderData = { container, candidates, logits };

  const temperature = config.get('temperature');
  renderOutput(container, candidates, logits, temperature);

  await delay(100);
}

/**
 * Re-render output with new temperature (called on config change).
 */
export function rerender() {
  if (!lastRenderData) return;
  const { container, candidates, logits } = lastRenderData;
  const temperature = config.get('temperature');
  renderOutput(container, candidates, logits, temperature);
}

/** Clear stored state (e.g., when starting a new pipeline run). */
export function reset() {
  lastRenderData = null;
}
