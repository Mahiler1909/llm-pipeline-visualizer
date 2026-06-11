/**
 * §1 Tokens — el texto se corta en piezas con ID.
 */

import * as models from '../models.js';
import { esc, tokenLabel, pastelBg, pastelInk } from './shared.js';

let root, flowEl, idsEl, captionEl, playOutEl;
let tokenizerReady = false;

export function init(rootEl) {
  root = rootEl;
  root.innerHTML = `
    <div class="token-flow" id="tok-flow"></div>
    <span class="token-arrow">↓ para el modelo, la frase es solo esto:</span>
    <div class="token-ids mono" id="tok-ids"></div>
    <p class="widget-caption" id="tok-caption">Pasa el cursor sobre un token para ver su ID.</p>
    <div class="token-play">
      <label for="tok-play-input">Prueba otra palabra o frase (tokenizer real, en vivo):</label>
      <input id="tok-play-input" type="text" placeholder="tokenizometro" spellcheck="false" autocomplete="off">
      <div class="token-play__out mono" id="tok-play-out"></div>
    </div>`;

  flowEl = root.querySelector('#tok-flow');
  idsEl = root.querySelector('#tok-ids');
  captionEl = root.querySelector('#tok-caption');
  playOutEl = root.querySelector('#tok-play-out');

  const playInput = root.querySelector('#tok-play-input');
  playInput.addEventListener('input', () => {
    const text = playInput.value;
    if (!text.trim() || !tokenizerReady) {
      playOutEl.innerHTML = '';
      return;
    }
    try {
      const { tokens } = models.tokenize(text);
      playOutEl.innerHTML =
        tokens.map((t, i) =>
          `<span class="token-piece" style="background:${pastelBg(i)}; border-bottom-color:${pastelInk(i)}">${esc(tokenLabel(t.text))}</span>`
        ).join('') +
        ` <span style="color:var(--muted)">→ ${tokens.length} token${tokens.length === 1 ? '' : 's'}</span>`;
    } catch { /* tokenizer aún no listo */ }
  });
}

export function update(state) {
  tokenizerReady = true;
  const { tokens } = state;

  flowEl.innerHTML = tokens.map((t, i) =>
    `<span class="token-piece" data-i="${i}"
       style="background:${pastelBg(i)}; border-bottom-color:${pastelInk(i)}">${esc(t.text)}</span>`
  ).join('');

  idsEl.innerHTML = '[' + tokens.map((t, i) =>
    `<span class="tid" style="background:${pastelBg(i)}; color:${pastelInk(i)}">${t.id}</span>`
  ).join('') + ']';

  flowEl.querySelectorAll('.token-piece').forEach(span => {
    span.addEventListener('mouseenter', () => {
      const t = tokens[Number(span.dataset.i)];
      const spaceNote = t.text.startsWith(' ') ? ' · el espacio inicial es parte del token' : '';
      captionEl.textContent = `"${tokenLabel(t.text)}" → ID ${t.id}${spaceNote}`;
    });
  });
  flowEl.addEventListener('mouseleave', () => {
    captionEl.textContent = `${tokens.length} tokens · vocabulario de ${state.modelConfig.vocab_size.toLocaleString('es')}`;
  });
  captionEl.textContent = `${tokens.length} tokens · vocabulario de ${state.modelConfig.vocab_size.toLocaleString('es')}`;
}
