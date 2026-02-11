/**
 * Network visualization on Canvas with zoom/pan.
 * Renders the full pipeline: Tokens → Embeddings → L1..LN → Logits → Output
 */

import { getTokenColor, lerp, clamp, seededRandom, fmt } from './utils.js';
import * as config from './config.js';

// Layout constants
const NODE_RADIUS = 6;
const COLUMN_GAP = 70;
const ROW_GAP = 36;
const LABEL_Y_OFFSET = 20;
const TOP_PADDING = 50;
const LEFT_PADDING = 100;

// Colors
const COLORS = {
  token: '#4ea8de',
  embedding: '#34d399',
  transformer: '#a78bfa',
  logit: '#f472b6',
  output: '#34d399',
  connection: 'rgba(167, 139, 250, 0.06)',
  connectionActive: 'rgba(167, 139, 250, 0.15)',
  bg: '#0d1117',
  text: '#8b949e',
  textBright: '#e6edf3',
};

/**
 * State
 */
let canvas = null;
let ctx = null;
let columns = []; // [{label, color, nodes: [{x,y,label,id,...}]}]
let outputBars = []; // [{word, prob, logit, x, y}]
let animProgress = 0;
let animTarget = 1;
let animFrame = null;
let glowTime = 0; // continuous timer for glow effect
let isAnimating = false;

// Model config stored for tooltips
let currentModelCfg = null;

// Zoom/pan state
let zoom = 1;
let panX = 0;
let panY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let panStartX = 0;
let panStartY = 0;

// Hover state
let hoveredNode = null;
let tooltip = null;

// Embedding modal callback
let onEmbeddingClick = null;

// Hover zone callback
let onHoverZoneChange = null;
let currentHoverZone = null;

export function init(canvasEl, tooltipEl, embeddingClickCb) {
  canvas = canvasEl;
  ctx = canvas.getContext('2d');
  tooltip = tooltipEl;
  onEmbeddingClick = embeddingClickCb;

  // Resize canvas to container
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Mouse events for zoom/pan
  canvas.addEventListener('wheel', handleWheel, { passive: false });
  canvas.addEventListener('mousedown', handleMouseDown);
  canvas.addEventListener('mousemove', handleMouseMove);
  canvas.addEventListener('mouseup', handleMouseUp);
  canvas.addEventListener('mouseleave', handleMouseLeave);
  canvas.addEventListener('click', handleClick);

  // Start continuous glow loop
  startGlowLoop();
}

function resizeCanvas() {
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = parent.clientWidth * dpr;
  canvas.height = parent.clientHeight * dpr;
  canvas.style.width = parent.clientWidth + 'px';
  canvas.style.height = parent.clientHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

/**
 * Continuous glow animation loop (runs always after first build).
 */
function startGlowLoop() {
  function tick() {
    glowTime += 0.015;
    if (columns.length > 0 && !isAnimating) {
      draw();
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/**
 * Build the visualization data from pipeline results.
 */
export function build(tokens, modelConfig, predictions) {
  columns = [];
  outputBars = [];
  animProgress = 0;
  animTarget = 1;
  currentModelCfg = modelConfig;

  const numTokens = tokens.length;
  const numLayers = modelConfig.layers;
  const nodesPerLayer = numTokens;

  const totalCols = 2 + numLayers + 1;
  const startY = TOP_PADDING + 30;

  let x = LEFT_PADDING;

  // 1. Tokens column
  const tokenNodes = tokens.map((t, i) => ({
    x, y: startY + i * ROW_GAP,
    label: t.text.trim() || '⎵',
    id: t.id,
    type: 'token',
    index: i,
  }));
  columns.push({ label: 'TOKENS', color: COLORS.token, nodes: tokenNodes, type: 'token' });
  x += COLUMN_GAP;

  // 2. Embeddings column
  const embedNodes = tokens.map((t, i) => ({
    x, y: startY + i * ROW_GAP,
    label: '',
    id: t.id,
    type: 'embedding',
    index: i,
    tokenText: t.text.trim(),
  }));
  columns.push({ label: 'EMBEDDINGS', color: COLORS.embedding, nodes: embedNodes, type: 'embedding' });
  x += COLUMN_GAP;

  // 3. Transformer layers
  for (let l = 0; l < numLayers; l++) {
    const layerNodes = [];
    for (let i = 0; i < nodesPerLayer; i++) {
      layerNodes.push({
        x, y: startY + i * ROW_GAP,
        label: '',
        type: 'transformer',
        layer: l,
        index: i,
      });
    }
    columns.push({
      label: `L${l + 1}`,
      color: COLORS.transformer,
      nodes: layerNodes,
      type: 'transformer',
    });
    x += COLUMN_GAP;
  }

  // 4. Logits column
  const numPreds = predictions.length;
  const logitStartY = startY;
  const MIN_LOGIT_GAP = 22; // minimum spacing so labels never overlap
  const actualLogitGap = Math.max(MIN_LOGIT_GAP, Math.min(ROW_GAP, ((numTokens - 1) * ROW_GAP) / Math.max(numPreds - 1, 1)));

  const logitNodes = predictions.map((p, i) => ({
    x, y: logitStartY + i * actualLogitGap,
    label: fmt(p.logit, 2),
    type: 'logit',
    index: i,
    word: p.word,
    prob: p.prob,
    logit: p.logit,
  }));
  columns.push({ label: 'LOGITS', color: COLORS.logit, nodes: logitNodes, type: 'logit' });
  x += COLUMN_GAP + 10;

  // 5. Output bars (with nucleus + sampling info)
  predictions.forEach((p, i) => {
    outputBars.push({
      x,
      y: logitStartY + i * actualLogitGap,
      word: p.word,
      prob: p.prob,
      logit: p.logit,
      inNucleus: p.inNucleus,
      isSampled: p.isSampled,
      nucleusProb: p.nucleusProb || 0,
      isWinner: i === 0,
    });
  });

  // Auto-fit zoom (increased width for percentages)
  const totalWidth = x + 280;
  const totalNodeHeight = Math.max(
    (numTokens - 1) * ROW_GAP + TOP_PADDING + 80,
    (numPreds - 1) * actualLogitGap + TOP_PADDING + 80
  );
  const canvasW = canvas.clientWidth;
  const canvasH = canvas.clientHeight;
  zoom = Math.min(canvasW / totalWidth, canvasH / totalNodeHeight, 1.5);
  zoom = Math.max(zoom, 0.3);
  panX = (canvasW - totalWidth * zoom) / 2;
  panY = (canvasH - totalNodeHeight * zoom) / 2;
  panY = Math.max(panY, 10);

  // Start animation (slower)
  startAnimation();
}

function startAnimation() {
  animProgress = 0;
  animTarget = 1;
  isAnimating = true;
  if (animFrame) cancelAnimationFrame(animFrame);
  animate();
}

function animate() {
  // Slower animation: 0.008 per frame (was 0.02)
  animProgress += 0.008;
  if (animProgress >= animTarget) {
    animProgress = animTarget;
    isAnimating = false;
    draw();
    return;
  }
  draw();
  animFrame = requestAnimationFrame(animate);
}

/**
 * Main draw function.
 */
function draw() {
  if (!ctx) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  const totalCols = columns.length;
  const progress = animProgress;

  // Draw connections with glow effect
  for (let c = 0; c < totalCols - 1; c++) {
    const colProgress = clamp((progress * totalCols - c) / 1, 0, 1);
    if (colProgress <= 0) continue;

    const from = columns[c];
    const to = columns[c + 1];

    let baseOpacity = 0.04;
    let lineColor = COLORS.transformer;

    if (from.type === 'token') {
      lineColor = COLORS.token;
      baseOpacity = 0.15;
    } else if (from.type === 'embedding') {
      lineColor = COLORS.embedding;
      baseOpacity = 0.08;
    } else if (to.type === 'logit') {
      lineColor = COLORS.logit;
      baseOpacity = 0.08;
    }

    // Glow pulse: a wave that travels across layers
    const glowPhase = (glowTime * 0.8 - c * 0.12) % 1.0;
    const glowIntensity = Math.max(0, Math.sin(glowPhase * Math.PI * 2)) * 0.25;

    for (const fromNode of from.nodes) {
      for (const toNode of to.nodes) {
        if (from.type === 'token' && fromNode.index !== toNode.index) continue;

        let opacity = baseOpacity;
        if (from.type === 'transformer' && fromNode.index === toNode.index) {
          opacity = 0.15;
        }
        if (to.type === 'logit') {
          opacity = 0.06 + toNode.prob * 0.3;
        }

        // Add glow pulse to self-connections and token connections
        let extraGlow = 0;
        if (from.type === 'transformer' && fromNode.index === toNode.index) {
          extraGlow = glowIntensity;
        } else if (from.type === 'token') {
          extraGlow = glowIntensity * 0.5;
        } else if (from.type === 'embedding') {
          extraGlow = glowIntensity * 0.3;
        }

        const alpha = (opacity + extraGlow) * colProgress;
        const lineWidth = (from.type === 'token' || to.type === 'logit')
          ? 1.5 + extraGlow * 3
          : 0.5 + extraGlow * 2;

        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.strokeStyle = lineColor;
        ctx.globalAlpha = Math.min(alpha, 0.8);
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Extra bright glow line on top for strong pulses
        if (extraGlow > 0.1 && (fromNode.index === toNode.index || from.type === 'token')) {
          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.strokeStyle = '#fff';
          ctx.globalAlpha = extraGlow * 0.3 * colProgress;
          ctx.lineWidth = lineWidth * 0.5;
          ctx.stroke();
        }
      }
    }
  }

  ctx.globalAlpha = 1;

  // Draw nodes
  for (let c = 0; c < totalCols; c++) {
    const col = columns[c];
    const colProgress = clamp((progress * (totalCols + 2) - c) / 1.5, 0, 1);
    if (colProgress <= 0) continue;

    // Node glow pulse per column
    const nodeGlowPhase = (glowTime * 0.8 - c * 0.12) % 1.0;
    const nodeGlow = Math.max(0, Math.sin(nodeGlowPhase * Math.PI * 2)) * 0.3;

    for (const node of col.nodes) {
      const r = NODE_RADIUS * colProgress;

      // Outer glow (enhanced with pulse)
      if (colProgress > 0.5) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 4 + nodeGlow * 4, 0, Math.PI * 2);
        ctx.fillStyle = col.color;
        ctx.globalAlpha = (0.15 + nodeGlow * 0.15) * colProgress;
        ctx.fill();
      }

      // Node circle
      ctx.globalAlpha = (0.7 + nodeGlow * 0.2) * colProgress;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = col.color;

      if (col.type === 'token') {
        ctx.fillStyle = getTokenColor(node.index);
        ctx.globalAlpha = 0.9 * colProgress;
      }

      if (col.type === 'logit' && node.index === 0) {
        ctx.fillStyle = COLORS.output;
        ctx.globalAlpha = 0.9 * colProgress;
      }

      ctx.fill();

      // Hovered node highlight
      if (hoveredNode && hoveredNode.x === node.x && hoveredNode.y === node.y) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 8, 0, Math.PI * 2);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;

      // Token text labels (left of node)
      if (col.type === 'token' && node.label) {
        ctx.font = '11px monospace';
        ctx.fillStyle = getTokenColor(node.index);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = colProgress;
        ctx.fillText(node.label, node.x - NODE_RADIUS - 8, node.y);
      }

      // Logit value labels (left of logit node)
      if (col.type === 'logit') {
        ctx.font = '9px monospace';
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = colProgress * 0.7;
        ctx.fillText(node.label, node.x - NODE_RADIUS - 6, node.y);
      }
    }

    // Column labels at bottom
    if (col.nodes.length > 0) {
      const bottomY = Math.max(...col.nodes.map(n => n.y)) + 40;
      ctx.font = 'bold 8px system-ui';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.globalAlpha = 0.6 * colProgress;
      ctx.fillText(col.label, col.nodes[0].x, bottomY);
    }
  }

  ctx.globalAlpha = 1;

  // Draw output bars with nucleus + sampling visualization
  const barsProgress = clamp((progress * (columns.length + 4) - columns.length) / 2, 0, 1);
  if (barsProgress > 0) {
    const maxBarWidth = 120;
    const barHeight = 16;
    const maxProb = outputBars.length > 0 ? Math.max(...outputBars.map(b => b.prob)) : 1;

    // Find the nucleus boundary for visual separator
    let lastNucleusIdx = -1;
    for (let i = 0; i < outputBars.length; i++) {
      if (outputBars[i].inNucleus) lastNucleusIdx = i;
    }

    for (let bi = 0; bi < outputBars.length; bi++) {
      const bar = outputBars[bi];
      const inNucleus = bar.inNucleus;
      const isSampled = bar.isSampled;
      const barWidth = (bar.prob / maxProb) * maxBarWidth * barsProgress;

      // Sampled token: bright glow ring
      if (isSampled) {
        ctx.beginPath();
        ctx.arc(bar.x, bar.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.globalAlpha = (0.2 + Math.sin(glowTime * 3) * 0.1) * barsProgress;
        ctx.fill();
      }

      // Dot
      let dotColor;
      if (isSampled) dotColor = '#fbbf24';
      else if (inNucleus) dotColor = COLORS.logit;
      else dotColor = COLORS.text;

      ctx.beginPath();
      ctx.arc(bar.x, bar.y, isSampled ? 6 : 5, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.globalAlpha = (inNucleus ? 0.9 : 0.3) * barsProgress;
      ctx.fill();

      // Sampled star
      if (isSampled) {
        ctx.font = '10px system-ui';
        ctx.fillStyle = '#fbbf24';
        ctx.globalAlpha = barsProgress;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('\u2605', bar.x, bar.y);
      }

      // Bar track
      ctx.globalAlpha = (inNucleus ? 0.1 : 0.04) * barsProgress;
      ctx.fillStyle = '#fff';
      ctx.fillRect(bar.x + 14, bar.y - barHeight / 2, maxBarWidth, barHeight);

      // Bar fill
      let barColor;
      if (isSampled) barColor = '#fbbf24';
      else if (inNucleus) barColor = COLORS.logit;
      else barColor = '#484f58';

      ctx.globalAlpha = (inNucleus ? 0.6 : 0.2) * barsProgress;
      ctx.fillStyle = barColor;
      ctx.fillRect(bar.x + 14, bar.y - barHeight / 2, barWidth, barHeight);

      // Percentage text
      const pctText = `${(bar.prob * 100).toFixed(1)}%`;
      ctx.globalAlpha = (inNucleus ? 1 : 0.35) * barsProgress;
      ctx.font = isSampled ? 'bold 9px monospace' : '9px monospace';
      ctx.fillStyle = isSampled ? '#fbbf24' : (inNucleus ? COLORS.textBright : COLORS.text);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(pctText, bar.x + 16 + barWidth + 4, bar.y);

      // Word label
      const pctWidth = ctx.measureText(pctText).width;
      ctx.font = isSampled ? 'bold 11px monospace' : '11px monospace';
      ctx.fillStyle = isSampled ? '#fbbf24' : (inNucleus ? COLORS.textBright : COLORS.text);
      ctx.globalAlpha = (inNucleus ? 1 : 0.35) * barsProgress;
      ctx.fillText(bar.word, bar.x + 16 + barWidth + pctWidth + 10, bar.y);

      // Draw nucleus separator line after last nucleus item
      if (bi === lastNucleusIdx && lastNucleusIdx < outputBars.length - 1) {
        const sepY = bar.y + (outputBars[bi + 1].y - bar.y) / 2;
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(bar.x - 5, sepY);
        ctx.lineTo(bar.x + maxBarWidth + 100, sepY);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.3)';
        ctx.globalAlpha = barsProgress;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);

        // Nucleus label
        ctx.font = '8px monospace';
        ctx.fillStyle = '#fbbf24';
        ctx.globalAlpha = 0.5 * barsProgress;
        ctx.textAlign = 'left';
        ctx.fillText('nucleus (top-p)', bar.x + maxBarWidth + 105, sepY);
      }
    }

    // SAMPLING label at bottom
    if (outputBars.length > 0) {
      const bottomY = Math.max(...outputBars.map(b => b.y)) + 40;
      ctx.font = 'bold 8px system-ui';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.globalAlpha = 0.6 * barsProgress;
      ctx.fillText('SAMPLING', outputBars[0].x + 60, bottomY);
    }
  }

  // Draw travel particle (autoregressive animation)
  if (travelParticle) {
    const p = travelParticle;
    // Trail
    ctx.beginPath();
    ctx.moveTo(p.startX, p.startY);
    ctx.lineTo(p.x, p.y);
    ctx.strokeStyle = p.color;
    ctx.globalAlpha = 0.3 * (1 - p.progress);
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glow
    ctx.beginPath();
    ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.15;
    ctx.fill();

    // Particle
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.9;
    ctx.fill();

    // Word label
    ctx.font = 'bold 10px monospace';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha = 1;
    ctx.fillText(p.word.trim(), p.x, p.y);
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}

/**
 * Update output bars when temperature/top-k changes (no full rebuild).
 */
export function updatePredictions(predictions) {
  if (columns.length === 0) return;

  // Update logit nodes
  const logitCol = columns.find(c => c.type === 'logit');
  if (logitCol) {
    predictions.forEach((p, i) => {
      if (i < logitCol.nodes.length) {
        logitCol.nodes[i].label = fmt(p.logit, 2);
        logitCol.nodes[i].word = p.word;
        logitCol.nodes[i].prob = p.prob;
        logitCol.nodes[i].logit = p.logit;
      }
    });
  }

  // Update output bars (including nucleus + sampling)
  outputBars.forEach((bar, i) => {
    if (i < predictions.length) {
      bar.word = predictions[i].word;
      bar.prob = predictions[i].prob;
      bar.logit = predictions[i].logit;
      bar.inNucleus = predictions[i].inNucleus;
      bar.isSampled = predictions[i].isSampled;
      bar.nucleusProb = predictions[i].nucleusProb || 0;
      bar.isWinner = i === 0;
    }
  });

  draw();
}

// ─── Zoom / Pan ───

function handleWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const prevZoom = zoom;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  zoom = clamp(zoom * delta, 0.15, 4);

  panX = mx - (mx - panX) * (zoom / prevZoom);
  panY = my - (my - panY) * (zoom / prevZoom);

  draw();
}

function handleMouseDown(e) {
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  panStartX = panX;
  panStartY = panY;
  canvas.style.cursor = 'grabbing';
}

function handleMouseMove(e) {
  if (isDragging) {
    panX = panStartX + (e.clientX - dragStartX);
    panY = panStartY + (e.clientY - dragStartY);
    draw();
    return;
  }

  // Hit detection for hover
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left - panX) / zoom;
  const my = (e.clientY - rect.top - panY) / zoom;

  let found = null;
  // Also check output bars for hover
  for (const bar of outputBars) {
    const dx = mx - bar.x;
    const dy = my - bar.y;
    if (dx * dx + dy * dy < 64) {
      found = { ...bar, type: 'output' };
      break;
    }
  }

  if (!found) {
    for (const col of columns) {
      for (const node of col.nodes) {
        const dx = mx - node.x;
        const dy = my - node.y;
        if (dx * dx + dy * dy < (NODE_RADIUS + 4) * (NODE_RADIUS + 4)) {
          found = node;
          break;
        }
      }
      if (found) break;
    }
  }

  // Detect hover zone from mouse position (even without a specific node)
  let newZone = null;
  if (found) {
    newZone = found.type === 'output' ? 'sampling' : found.type;
  } else if (columns.length > 0) {
    // Zone detection by x position in column ranges
    const halfGap = COLUMN_GAP / 2;
    for (const col of columns) {
      if (col.nodes.length > 0) {
        const colX = col.nodes[0].x;
        if (mx >= colX - halfGap && mx <= colX + halfGap) {
          newZone = col.type;
          break;
        }
      }
    }
    // Check output bars area
    if (!newZone && outputBars.length > 0) {
      const barX = outputBars[0].x;
      if (mx >= barX - halfGap) {
        newZone = 'sampling';
      }
    }
  }

  if (newZone !== currentHoverZone) {
    currentHoverZone = newZone;
    if (onHoverZoneChange) {
      onHoverZoneChange(newZone, currentModelCfg);
    }
  }

  if (found !== hoveredNode) {
    hoveredNode = found;
    canvas.style.cursor = found ? 'pointer' : 'grab';

    if (found && tooltip) {
      const text = getTooltipText(found);
      tooltip.innerHTML = text;
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY - 10) + 'px';
      tooltip.hidden = false;
    } else if (tooltip) {
      tooltip.hidden = true;
    }

    draw();
  } else if (found && tooltip) {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
  }
}

/**
 * Generate detailed tooltip text for each node type.
 */
function getTooltipText(node) {
  const cfg = currentModelCfg;
  const dim = cfg ? cfg.hidden_dim : 768;
  const heads = cfg ? cfg.heads : 12;
  const ffn = cfg ? cfg.ffn_dim : 3072;

  switch (node.type) {
    case 'token':
      return `<b>Token:</b> "${node.label}"<br>` +
             `<b>ID:</b> ${node.id}<br>` +
             `<b>Posicion:</b> ${node.index + 1} en la secuencia<br>` +
             `<span style="color:#8b949e">El tokenizer BPE convierte texto en IDs numericos</span>`;

    case 'embedding':
      return `<b>Embedding:</b> "${node.tokenText}"<br>` +
             `<b>Dimension:</b> ${dim}d (vector de ${dim} numeros)<br>` +
             `<b>Token ID:</b> ${node.id}<br>` +
             `<span style="color:#34d399">Click para ver el heatmap del vector</span>`;

    case 'transformer':
      return `<b>Capa ${node.layer + 1}</b> de ${cfg ? cfg.layers : '?'}<br>` +
             `<b>Posicion:</b> token ${node.index + 1}<br>` +
             `<b>Atencion:</b> ${heads} cabezas<br>` +
             `<b>FFN:</b> ${dim}→${ffn}→${dim}<br>` +
             `<span style="color:#8b949e">Self-attention + feed-forward network</span>`;

    case 'logit':
      return `<b>${node.word}</b><br>` +
             `<b>Logit:</b> ${fmt(node.logit, 4)}<br>` +
             `<b>Prob:</b> ${(node.prob * 100).toFixed(2)}%<br>` +
             `<b>Rank:</b> #${node.index + 1}<br>` +
             `<span style="color:#8b949e">Temp: ${config.get('temperature').toFixed(2)}</span>`;

    case 'output':
      return `<b>${node.word}</b> ${node.isSampled ? '<span style="color:#fbbf24">\u2605 SAMPLEADO</span>' : ''}<br>` +
             `<b>Probabilidad:</b> ${(node.prob * 100).toFixed(2)}%<br>` +
             (node.inNucleus ? `<b>Prob. nucleus:</b> ${(node.nucleusProb * 100).toFixed(2)}%<br>` : '') +
             `<b>Logit raw:</b> ${fmt(node.logit, 4)}<br>` +
             `<b>Nucleus:</b> ${node.inNucleus ? '<span style="color:#34d399">Si</span>' : '<span style="color:#f87171">No (filtrado por top-p)</span>'}<br>` +
             `<span style="color:#8b949e">softmax(logit / T) → top-p filtering → sample</span>`;

    default:
      return '';
  }
}

function handleMouseUp() {
  isDragging = false;
  canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
}

function handleMouseLeave() {
  isDragging = false;
  hoveredNode = null;
  if (tooltip) tooltip.hidden = true;
  canvas.style.cursor = 'grab';
  if (currentHoverZone !== null) {
    currentHoverZone = null;
    if (onHoverZoneChange) onHoverZoneChange(null, currentModelCfg);
  }
}

function handleClick(e) {
  if (!hoveredNode) return;
  if (hoveredNode.type === 'embedding' && onEmbeddingClick) {
    onEmbeddingClick(hoveredNode.id, hoveredNode.tokenText);
  }
}

// ─── Token Travel Animation (autoregressive) ───

let travelParticle = null; // {x, y, targetX, targetY, startX, startY, progress, word, color}

/**
 * Animate a token "traveling" from the output/sampling area back to the token column.
 * Returns a promise that resolves when the animation completes.
 */
export function animateTokenTravel(word) {
  return new Promise((resolve) => {
    if (columns.length === 0 || outputBars.length === 0) {
      resolve();
      return;
    }

    // Find the sampled bar position
    const sampled = outputBars.find(b => b.isSampled) || outputBars[0];
    const tokenCol = columns[0];
    const lastTokenNode = tokenCol.nodes[tokenCol.nodes.length - 1];

    travelParticle = {
      startX: sampled.x,
      startY: sampled.y,
      targetX: lastTokenNode.x,
      targetY: lastTokenNode.y + ROW_GAP,
      x: sampled.x,
      y: sampled.y,
      progress: 0,
      word: word,
      color: '#fbbf24',
    };

    function tick() {
      travelParticle.progress += 0.025;
      if (travelParticle.progress >= 1) {
        travelParticle = null;
        draw();
        resolve();
        return;
      }

      // Eased progress
      const t = travelParticle.progress;
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      travelParticle.x = lerp(travelParticle.startX, travelParticle.targetX, ease);
      travelParticle.y = lerp(travelParticle.startY, travelParticle.targetY, ease);

      draw();
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

// Hover zone listener
export function onHoverZone(cb) {
  onHoverZoneChange = cb;
}

export function getModelConfig() {
  return currentModelCfg;
}

// Zoom controls
export function zoomIn() {
  zoom = clamp(zoom * 1.3, 0.15, 4);
  draw();
}

export function zoomOut() {
  zoom = clamp(zoom * 0.7, 0.15, 4);
  draw();
}

export function clear() {
  columns = [];
  outputBars = [];
  travelParticle = null;
  hoveredNode = null;
  currentModelCfg = null;
  animProgress = 0;
  if (ctx) {
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  }
}

export function resetView() {
  if (columns.length > 0) {
    const allNodes = columns.flatMap(c => c.nodes);
    const allX = [...allNodes.map(n => n.x), ...outputBars.map(b => b.x + 280)];
    const allY = [...allNodes.map(n => n.y), ...outputBars.map(b => b.y)];
    const minX = Math.min(...allX) - 80;
    const maxX = Math.max(...allX) + 80;
    const minY = Math.min(...allY) - 60;
    const maxY = Math.max(...allY) + 60;
    const contentW = maxX - minX;
    const contentH = maxY - minY;
    const canvasW = canvas.clientWidth;
    const canvasH = canvas.clientHeight;
    zoom = Math.min(canvasW / contentW, canvasH / contentH, 1.5);
    zoom = Math.max(zoom, 0.3);
    panX = (canvasW - contentW * zoom) / 2 - minX * zoom;
    panY = (canvasH - contentH * zoom) / 2 - minY * zoom;
  }
  draw();
}
