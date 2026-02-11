/**
 * Transformer step: visualizes attention mechanism and neural network processing.
 * Renders an SVG network diagram and attention weight matrix.
 */

import { softmax, fmt, delay, reveal, clearElement } from './utils.js';

const LAYER_NAMES = ['Input', 'Attention', 'Feed-Forward', 'Output'];

/**
 * Compute attention weights between tokens using dot-product of embeddings.
 */
function computeAttention(tokenData) {
  const n = tokenData.length;
  const matrix = [];

  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      // Dot product of embeddings
      let dot = 0;
      const a = tokenData[i].embedding;
      const b = tokenData[j].embedding;
      for (let d = 0; d < a.length; d++) {
        dot += a[d] * b[d];
      }
      row.push(dot);
    }
    // Apply softmax to get attention weights
    matrix.push(softmax(row));
  }

  return matrix;
}

/**
 * Build SVG network visualization.
 */
function buildNetworkSVG(tokenData, attentionMatrix) {
  const n = tokenData.length;
  const svgNS = 'http://www.w3.org/2000/svg';
  const width = 800;
  const height = Math.max(220, n * 45 + 40);
  const layerCount = 4;
  const layerSpacing = width / (layerCount + 1);

  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('class', 'transformer-network');
  svg.style.width = '100%';
  svg.style.height = `${height}px`;

  // Glow filter
  const defs = document.createElementNS(svgNS, 'defs');
  const filter = document.createElementNS(svgNS, 'filter');
  filter.setAttribute('id', 'glow');
  const blur = document.createElementNS(svgNS, 'feGaussianBlur');
  blur.setAttribute('stdDeviation', '3');
  blur.setAttribute('result', 'coloredBlur');
  filter.appendChild(blur);
  const merge = document.createElementNS(svgNS, 'feMerge');
  const mergeNode1 = document.createElementNS(svgNS, 'feMergeNode');
  mergeNode1.setAttribute('in', 'coloredBlur');
  merge.appendChild(mergeNode1);
  const mergeNode2 = document.createElementNS(svgNS, 'feMergeNode');
  mergeNode2.setAttribute('in', 'SourceGraphic');
  merge.appendChild(mergeNode2);
  filter.appendChild(merge);
  defs.appendChild(filter);
  svg.appendChild(defs);

  // Layer labels
  LAYER_NAMES.forEach((name, i) => {
    const text = document.createElementNS(svgNS, 'text');
    text.setAttribute('x', layerSpacing * (i + 1));
    text.setAttribute('y', 16);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#94a3b8');
    text.setAttribute('font-size', '11');
    text.setAttribute('font-family', 'system-ui');
    text.textContent = name;
    svg.appendChild(text);
  });

  // Compute node positions per layer
  const layers = [];
  const nodeCountPerLayer = [n, n, Math.max(3, Math.ceil(n * 1.5)), n];

  for (let l = 0; l < layerCount; l++) {
    const count = nodeCountPerLayer[l];
    const x = layerSpacing * (l + 1);
    const nodeSpacing = (height - 50) / (count + 1);
    const nodes = [];
    for (let j = 0; j < count; j++) {
      nodes.push({ x, y: 30 + nodeSpacing * (j + 1) });
    }
    layers.push(nodes);
  }

  // Draw connections (links)
  const links = [];
  for (let l = 0; l < layerCount - 1; l++) {
    const from = layers[l];
    const to = layers[l + 1];
    for (let i = 0; i < from.length; i++) {
      for (let j = 0; j < to.length; j++) {
        const line = document.createElementNS(svgNS, 'line');
        line.setAttribute('x1', from[i].x);
        line.setAttribute('y1', from[i].y);
        line.setAttribute('x2', to[j].x);
        line.setAttribute('y2', to[j].y);
        line.setAttribute('stroke', 'rgba(244, 114, 182, 0.08)');
        line.setAttribute('stroke-width', '1');
        line.setAttribute('class', 'link');
        line.dataset.layer = l;

        // For attention layer, use attention weights for opacity
        if (l === 0 && i < attentionMatrix.length && j < attentionMatrix.length) {
          const weight = attentionMatrix[i][j];
          line.dataset.weight = weight;
        }

        svg.appendChild(line);
        links.push(line);
      }
    }
  }

  // Draw nodes
  const allNodes = [];
  const colors = ['#00d4ff', '#f472b6', '#fbbf24', '#34d399'];

  for (let l = 0; l < layerCount; l++) {
    for (let j = 0; j < layers[l].length; j++) {
      const circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', layers[l][j].x);
      circle.setAttribute('cy', layers[l][j].y);
      circle.setAttribute('r', '6');
      circle.setAttribute('fill', colors[l]);
      circle.setAttribute('opacity', '0.2');
      circle.setAttribute('class', 'node');
      circle.dataset.layer = l;
      svg.appendChild(circle);
      allNodes.push(circle);

      // Token labels for input layer
      if (l === 0 && j < tokenData.length) {
        const label = document.createElementNS(svgNS, 'text');
        label.setAttribute('x', layers[l][j].x - 20);
        label.setAttribute('y', layers[l][j].y + 4);
        label.setAttribute('text-anchor', 'end');
        label.setAttribute('fill', '#94a3b8');
        label.setAttribute('font-size', '10');
        label.setAttribute('font-family', 'monospace');
        label.textContent = tokenData[j].text;
        svg.appendChild(label);
      }
    }
  }

  return { svg, links, nodes: allNodes };
}

/**
 * Build attention matrix visualization.
 */
function buildAttentionMatrix(tokenData, attentionMatrix) {
  const section = document.createElement('div');
  section.className = 'attention-section';

  const label = document.createElement('div');
  label.className = 'attention-label';
  label.textContent = 'Attention Weights Matrix';
  section.appendChild(label);

  const matrix = document.createElement('div');
  matrix.className = 'attention-matrix';

  // Header row
  const headerRow = document.createElement('div');
  headerRow.className = 'attention-matrix__row';
  const emptyLabel = document.createElement('span');
  emptyLabel.className = 'attention-matrix__label';
  headerRow.appendChild(emptyLabel);
  tokenData.forEach(t => {
    const cell = document.createElement('div');
    cell.className = 'attention-matrix__cell';
    cell.style.fontSize = '0.55rem';
    cell.style.color = '#94a3b8';
    cell.textContent = t.text.slice(0, 5);
    headerRow.appendChild(cell);
  });
  matrix.appendChild(headerRow);

  // Data rows
  for (let i = 0; i < tokenData.length; i++) {
    const row = document.createElement('div');
    row.className = 'attention-matrix__row';

    const label = document.createElement('span');
    label.className = 'attention-matrix__label';
    label.textContent = tokenData[i].text;
    row.appendChild(label);

    for (let j = 0; j < tokenData.length; j++) {
      const weight = attentionMatrix[i][j];
      const cell = document.createElement('div');
      cell.className = 'attention-matrix__cell';
      const intensity = Math.round(weight * 255);
      cell.style.backgroundColor = `rgba(244, 114, 182, ${weight * 0.8})`;
      cell.textContent = fmt(weight);
      row.appendChild(cell);
    }

    matrix.appendChild(row);
  }

  section.appendChild(matrix);
  return section;
}

/**
 * Run the transformer step.
 * @param {Array} tokenData - Array of { text, id, embedding } from embeddings step.
 * @param {HTMLElement} container - DOM element to render into.
 * @returns {Object} - { tokens: tokenData, attention: matrix, hiddenState: array }
 */
export async function run(tokenData, container) {
  clearElement(container);

  const viz = document.createElement('div');
  viz.className = 'transformer-viz';
  container.appendChild(viz);

  const attentionMatrix = computeAttention(tokenData);
  const { svg, links, nodes } = buildNetworkSVG(tokenData, attentionMatrix);
  viz.appendChild(svg);

  // Animate nodes layer by layer
  const layerCount = 4;
  for (let l = 0; l < layerCount; l++) {
    const layerNodes = nodes.filter(n => n.dataset.layer == l);
    const layerLinks = links.filter(link => link.dataset.layer == l);

    // Activate nodes
    for (const node of layerNodes) {
      node.setAttribute('opacity', '0.9');
      node.classList.add('active');
      await delay(40);
    }

    // Activate connections from this layer
    for (const link of layerLinks) {
      const weight = parseFloat(link.dataset.weight || 0.3);
      const opacity = 0.1 + weight * 0.7;
      const strokeWidth = 0.5 + weight * 2.5;
      link.setAttribute('stroke', `rgba(244, 114, 182, ${opacity})`);
      link.setAttribute('stroke-width', strokeWidth);
      link.classList.add('active');
    }

    await delay(200);
  }

  // Show attention matrix
  const matrixViz = buildAttentionMatrix(tokenData, attentionMatrix);
  viz.appendChild(matrixViz);
  await delay(100);
  await reveal(matrixViz, 'visible');

  // Compute a simple "hidden state" by weighting embeddings with attention
  const hiddenState = tokenData.map((_, i) => {
    const dim = tokenData[0].embedding.length;
    const weighted = new Array(dim).fill(0);
    for (let j = 0; j < tokenData.length; j++) {
      for (let d = 0; d < dim; d++) {
        weighted[d] += attentionMatrix[i][j] * tokenData[j].embedding[d];
      }
    }
    return weighted;
  });

  return { tokens: tokenData, attention: attentionMatrix, hiddenState };
}
