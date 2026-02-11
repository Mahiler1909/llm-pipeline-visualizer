/**
 * Embeddings step: converts tokens into vector representations.
 * Uses deterministic hashing to produce consistent vectors.
 */

import { seededRandom, valueToColor, fmt, delay, reveal, clearElement } from './utils.js';

const DIMS = 8;
const DIM_LABELS = ['Sem.', 'Pos.', 'Ctx.', 'Synt.', 'Tone', 'Freq.', 'Rel.', 'Abs.'];

/**
 * Generate an embedding vector for a token.
 * @param {number} tokenId - The token's numeric ID.
 * @returns {number[]} - Array of DIMS floats in [-1, 1].
 */
function generateEmbedding(tokenId) {
  const rng = seededRandom(tokenId);
  const vec = [];
  for (let i = 0; i < DIMS; i++) {
    vec.push(rng() * 2 - 1);
  }
  return vec;
}

/**
 * Run the embeddings step.
 * @param {Array} tokens - Array of { text, id } from tokenizer.
 * @param {HTMLElement} container - DOM element to render into.
 * @returns {Array} - Array of { text, id, embedding } objects.
 */
export async function run(tokens, container) {
  clearElement(container);

  const wrapper = document.createElement('div');
  wrapper.className = 'embeddings-container';
  container.appendChild(wrapper);

  const table = document.createElement('div');
  table.className = 'embeddings-table';
  wrapper.appendChild(table);

  // Header row with dimension labels
  const header = document.createElement('div');
  header.className = 'embeddings-header';
  DIM_LABELS.forEach(label => {
    const dim = document.createElement('span');
    dim.className = 'embeddings-header__dim';
    dim.textContent = label;
    header.appendChild(dim);
  });
  table.appendChild(header);

  // Generate embeddings and build rows
  const results = [];
  const rows = [];

  for (const token of tokens) {
    const embedding = generateEmbedding(token.id);
    results.push({ ...token, embedding });

    const row = document.createElement('div');
    row.className = 'embedding-row';

    const label = document.createElement('span');
    label.className = 'embedding-row__label';
    label.textContent = token.text;
    row.appendChild(label);

    const cells = document.createElement('div');
    cells.className = 'embedding-row__cells';

    const cellElements = [];
    embedding.forEach(val => {
      const cell = document.createElement('div');
      cell.className = 'embedding-cell';
      cell.style.backgroundColor = valueToColor(val);
      cell.textContent = fmt(val);
      cell.title = `${fmt(val, 4)}`;
      cells.appendChild(cell);
      cellElements.push(cell);
    });

    row.appendChild(cells);
    table.appendChild(row);
    rows.push({ row, cellElements });
  }

  // Animate rows and cells
  for (const { row, cellElements } of rows) {
    await reveal(row, 'visible');
    for (const cell of cellElements) {
      await reveal(cell, 'visible');
      await delay(30);
    }
    await delay(80);
  }

  return results;
}
