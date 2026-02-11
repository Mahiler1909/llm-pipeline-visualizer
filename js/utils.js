/**
 * Shared utility functions.
 */

export function softmax(logits, temperature = 1.0) {
  const t = Math.max(temperature, 0.01);
  const scaled = logits.map(l => l / t);
  const maxVal = Math.max(...scaled);
  const exps = scaled.map(s => Math.exp(s - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

export function seededRandom(seed) {
  let t = (seed >>> 0) + 0x6D2B79F5;
  return function () {
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

/** Distinct colors for tokens */
const TOKEN_COLORS = [
  '#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399',
  '#22d3ee', '#60a5fa', '#a78bfa', '#f472b6', '#e879f9',
];

export function getTokenColor(index) {
  return TOKEN_COLORS[index % TOKEN_COLORS.length];
}

export function fmt(num, decimals = 2) {
  return num.toFixed(decimals);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}
