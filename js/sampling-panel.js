/**
 * Live softmax distribution panel (sidebar).
 * Vertical bars for the top-n candidates that animate smoothly when the
 * temperature/top-k/top-p sliders move, with visual markers for the
 * top-k cut, the top-p nucleus area and the sampled token (★).
 */

const ANIM_MS = 250;
const PAD_TOP = 34;     // room for the ★ + word of the sampled token
const PAD_BOTTOM = 30;  // room for rank labels
const PAD_X = 6;

let canvas = null;
let ctx = null;
let tooltip = null;

let dist = null;            // last distribution from pipeline.getDistribution()
let displayProbs = null;    // animated bar heights (probs)
let fromProbs = null;
let animStart = 0;
let animFrame = null;
let hoverIdx = -1;

export function init(canvasEl, tooltipEl) {
  canvas = canvasEl;
  ctx = canvas.getContext('2d');
  tooltip = tooltipEl;
  canvas.addEventListener('mousemove', handleHover);
  canvas.addEventListener('mouseleave', () => {
    hoverIdx = -1;
    if (tooltip) tooltip.hidden = true;
    draw();
  });
}

export function update(newDist) {
  if (!canvas || !newDist) return;
  const n = newDist.items.length;

  // Match previous heights by tokenId so bars slide instead of jumping
  const prevByToken = new Map();
  if (dist && displayProbs) {
    dist.items.forEach((item, i) => prevByToken.set(item.tokenId, displayProbs[i]));
  }
  fromProbs = newDist.items.map(item => prevByToken.get(item.tokenId) ?? 0);

  dist = newDist;
  animStart = performance.now();
  if (animFrame) cancelAnimationFrame(animFrame);
  tick();
}

export function clear() {
  dist = null;
  displayProbs = null;
  if (animFrame) cancelAnimationFrame(animFrame);
  animFrame = null;
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function tick() {
  const t = Math.min((performance.now() - animStart) / ANIM_MS, 1);
  const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic
  displayProbs = dist.items.map((item, i) => fromProbs[i] + (item.prob - fromProbs[i]) * ease);
  draw();
  if (t < 1) {
    animFrame = requestAnimationFrame(tick);
  } else {
    animFrame = null;
  }
}

function barColor(item) {
  if (item.isSampled) return '#fbbf24';
  if (item.cutByTopK) return '#3a4048';
  if (!item.inNucleus) return '#6e7681';
  return '#f472b6';
}

function draw() {
  if (!ctx || !dist || !displayProbs) return;
  const W = canvas.width;
  const H = canvas.height;
  const n = dist.items.length;
  const chartH = H - PAD_TOP - PAD_BOTTOM;
  const slot = (W - PAD_X * 2) / n;
  const barW = Math.max(slot - 3, 2);

  ctx.clearRect(0, 0, W, H);

  // Nucleus shading: from bar 0 up to the last in-nucleus bar
  let lastNucleus = -1;
  dist.items.forEach((item, i) => { if (item.inNucleus) lastNucleus = i; });
  if (lastNucleus >= 0) {
    ctx.fillStyle = 'rgba(251, 191, 36, 0.07)';
    ctx.fillRect(PAD_X, PAD_TOP, (lastNucleus + 1) * slot, chartH);
  }

  // Bars
  for (let i = 0; i < n; i++) {
    const item = dist.items[i];
    const h = Math.max(displayProbs[i] * chartH, 2);
    const x = PAD_X + i * slot + (slot - barW) / 2;
    const y = PAD_TOP + chartH - h;

    ctx.fillStyle = barColor(item);
    ctx.globalAlpha = item.cutByTopK ? 0.55 : 1;
    ctx.fillRect(x, y, barW, h);
    ctx.globalAlpha = 1;

    if (i === hoverIdx) {
      ctx.strokeStyle = '#e6edf3';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 1, y - 1, barW + 2, h + 2);
    }

    // ★ + word over the sampled bar
    if (item.isSampled) {
      ctx.fillStyle = '#fbbf24';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.font = '13px system-ui';
      ctx.fillText('★', x + barW / 2, y - 14);
      ctx.font = 'bold 11px monospace';
      const word = item.word.trim() || '⎵';
      ctx.fillText(word.length > 8 ? word.slice(0, 8) + '…' : word, x + barW / 2, y - 2);
    }
  }

  // Top-k cut: dashed vertical line after bar topK-1
  if (dist.topK < n) {
    const cutX = PAD_X + dist.topK * slot;
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(cutX, PAD_TOP - 4);
    ctx.lineTo(cutX, PAD_TOP + chartH);
    ctx.strokeStyle = '#8b949e';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('corte top-k', Math.min(cutX + 4, W - 70), PAD_TOP - 4);
  }

  // Nucleus label
  if (lastNucleus >= 0) {
    ctx.fillStyle = 'rgba(251, 191, 36, 0.7)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('nucleus (top-p)', PAD_X + 2, H - 16);
  }

  // Baseline + rank ticks
  ctx.strokeStyle = 'rgba(139, 148, 158, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD_X, PAD_TOP + chartH + 0.5);
  ctx.lineTo(W - PAD_X, PAD_TOP + chartH + 0.5);
  ctx.stroke();

  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('1', PAD_X + slot / 2, PAD_TOP + chartH + 4);
  ctx.fillText(String(n), PAD_X + (n - 0.5) * slot, PAD_TOP + chartH + 4);
}

function handleHover(e) {
  if (!dist) return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const n = dist.items.length;
  const slot = (canvas.width - PAD_X * 2) / n;
  const i = Math.floor((mx - PAD_X) / slot);

  if (i < 0 || i >= n) {
    hoverIdx = -1;
    if (tooltip) tooltip.hidden = true;
    draw();
    return;
  }

  if (i !== hoverIdx) {
    hoverIdx = i;
    draw();
  }

  const item = dist.items[i];
  let estado;
  if (item.isSampled) estado = '<span style="color:#fbbf24">★ Token elegido</span>';
  else if (item.cutByTopK) estado = '<span style="color:#8b949e">Eliminado por top-k</span>';
  else if (!item.inNucleus) estado = '<span style="color:#f87171">Fuera del nucleus (top-p)</span>';
  else estado = '<span style="color:#34d399">Candidato en el nucleus</span>';

  if (tooltip) {
    tooltip.innerHTML =
      `<b>"${item.word.trim() || '⎵'}"</b> &middot; rank #${item.rank + 1}<br>` +
      `<b>Prob:</b> ${(item.prob * 100).toFixed(2)}% &middot; <b>Logit:</b> ${item.rawLogit.toFixed(2)}<br>` +
      estado;
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.hidden = false;
  }
}
