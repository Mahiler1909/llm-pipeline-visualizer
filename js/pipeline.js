/**
 * Pipeline orchestrator: tokenize → forward → compute predictions with sampling.
 */

import * as models from './models.js';
import * as config from './config.js';

let lastLogits = null;
let lastTokens = null;
let lastModelConfig = null;
let currentText = '';

/**
 * Run the full pipeline: tokenize + inference.
 * Returns { tokens, predictions, modelConfig }
 */
export async function run(text) {
  currentText = text;
  lastModelConfig = models.getConfig(models.getLoadedModelId());

  const { tokens, encoded } = models.tokenize(text);
  lastTokens = tokens;

  const { logits } = await models.forward(encoded);
  lastLogits = logits;

  const predictions = computePredictions(
    logits,
    config.get('temperature'),
    config.get('topK'),
    config.get('topP')
  );

  return { tokens, predictions, modelConfig: lastModelConfig };
}

/**
 * Recompute predictions with new temperature/top-k/top-p (no re-inference needed).
 */
export function recomputePredictions() {
  if (!lastLogits) return null;
  return computePredictions(
    lastLogits,
    config.get('temperature'),
    config.get('topK'),
    config.get('topP')
  );
}

/**
 * Generate one more token: append the top prediction and re-run.
 */
export async function generateMore() {
  if (!lastLogits || !lastTokens) return null;

  const predictions = computePredictions(
    lastLogits,
    config.get('temperature'),
    config.get('topK'),
    config.get('topP')
  );
  // Use the sampled token (not just the highest prob)
  const sampled = predictions.find(p => p.isSampled) || predictions[0];
  const newText = currentText + sampled.word;
  return { newText, topWord: sampled.word };
}

/**
 * Compute top-k predictions from logits with temperature, top-p nucleus filtering, and sampling.
 */
function computePredictions(logits, temperature, topK, topP) {
  // 1. Apply temperature
  const scaled = new Float32Array(logits.length);
  const t = Math.max(temperature, 0.01);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / t;
  }

  // 2. Find top-k indices
  const indexed = [];
  for (let i = 0; i < scaled.length; i++) {
    indexed.push({ val: scaled[i], rawLogit: logits[i], idx: i });
  }
  indexed.sort((a, b) => b.val - a.val);
  const topItems = indexed.slice(0, topK);

  // 3. Softmax over top-k
  const maxVal = topItems[0].val;
  const exps = topItems.map(item => ({
    ...item,
    exp: Math.exp(item.val - maxVal),
  }));
  const sumExp = exps.reduce((sum, item) => sum + item.exp, 0);

  const predictions = exps.map(item => ({
    word: models.decodeToken(item.idx),
    prob: item.exp / sumExp,
    logit: item.rawLogit,
    tokenId: item.idx,
    inNucleus: false,
    isSampled: false,
    nucleusProb: 0,
  }));

  // 4. Top-p nucleus filtering
  let cumProb = 0;
  for (const pred of predictions) {
    cumProb += pred.prob;
    pred.inNucleus = true;
    if (cumProb >= topP) break;
  }

  // 5. Re-normalize nucleus probabilities
  const nucleus = predictions.filter(p => p.inNucleus);
  const nucleusSum = nucleus.reduce((s, p) => s + p.prob, 0);
  nucleus.forEach(p => { p.nucleusProb = p.prob / nucleusSum; });

  // 6. Sample one token from nucleus
  const r = Math.random();
  let cum = 0;
  for (const p of nucleus) {
    cum += p.nucleusProb;
    if (r <= cum) {
      p.isSampled = true;
      break;
    }
  }
  // Fallback: if none sampled (floating point), pick first
  if (!nucleus.some(p => p.isSampled)) {
    nucleus[0].isSampled = true;
  }

  return predictions;
}

export function getLastTokens() {
  return lastTokens;
}

export function getCurrentText() {
  return currentText;
}

export function hasResults() {
  return lastLogits !== null;
}
