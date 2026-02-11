/**
 * Tokenization step: splits text into tokens and assigns IDs.
 * Simulates a simplified BPE-like tokenization.
 */

import { hash, delay, reveal, clearElement } from './utils.js';

// Common vocabulary with pre-assigned IDs for consistency
const VOCAB = {
  'el': 102, 'la': 103, 'los': 104, 'las': 105, 'un': 106, 'una': 107,
  'de': 108, 'en': 109, 'que': 110, 'y': 111, 'a': 112, 'es': 113,
  'se': 114, 'no': 115, 'por': 116, 'con': 117, 'para': 118, 'como': 119,
  'su': 120, 'del': 121, 'al': 122, 'lo': 123, 'más': 124, 'pero': 125,
  'the': 200, 'a': 201, 'is': 202, 'of': 203, 'and': 204, 'to': 205,
  'in': 206, 'it': 207, 'that': 208, 'was': 209, 'for': 210, 'on': 211,
  'are': 212, 'with': 213, 'this': 214, 'be': 215, 'not': 216,
  'gato': 4521, 'perro': 4522, 'casa': 4523, 'mundo': 4524, 'tiempo': 4525,
  'vida': 4526, 'hombre': 4527, 'mujer': 4528, 'agua': 4529, 'día': 4530,
  'sentó': 5001, 'comió': 5002, 'corrió': 5003, 'habló': 5004, 'miró': 5005,
  'cat': 6001, 'dog': 6002, 'house': 6003, 'world': 6004, 'time': 6005,
  'sat': 6010, 'ate': 6011, 'ran': 6012, 'quick': 6013, 'brown': 6014,
  'fox': 6015, 'jumps': 6016, 'over': 6017, 'lazy': 6018,
  'inteligencia': 7001, 'artificial': 7002, 'puede': 7003, 'ser': 7004,
  'muy': 7005, 'grande': 7006, 'pequeño': 7007, 'bueno': 7008, 'malo': 7009,
  'hacer': 7010, 'crear': 7011, 'pensar': 7012, 'aprender': 7013,
};

/**
 * Tokenize a text string into token objects.
 * Returns: [{ text, id }]
 */
function tokenize(text) {
  // Split by spaces and punctuation, keeping punctuation as separate tokens
  const parts = text
    .trim()
    .split(/(\s+|(?=[.,!?;:])|(?<=[.,!?;:]))/)
    .filter(p => p && p.trim().length > 0);

  return parts.map(word => {
    const lower = word.toLowerCase();
    const id = VOCAB[lower] || (1000 + hash(lower) % 9000);
    return { text: word, id };
  });
}

/**
 * Run the tokenization step.
 * @param {string} text - The input query text.
 * @param {HTMLElement} container - DOM element to render into.
 * @returns {Array} - Array of token objects { text, id }.
 */
export async function run(text, container) {
  clearElement(container);

  const tokens = tokenize(text);

  const grid = document.createElement('div');
  grid.className = 'tokenizer-grid';
  container.appendChild(grid);

  // Create all token cards first (hidden)
  const cards = tokens.map(token => {
    const card = document.createElement('div');
    card.className = 'token-card';
    card.innerHTML = `
      <span class="token-card__text">"${token.text}"</span>
      <span class="token-card__arrow">&darr;</span>
      <span class="token-card__id">[${token.id}]</span>
    `;
    grid.appendChild(card);
    return card;
  });

  // Reveal one by one with staggered animation
  for (const card of cards) {
    await reveal(card, 'visible');
    await delay(120);
  }

  return tokens;
}
