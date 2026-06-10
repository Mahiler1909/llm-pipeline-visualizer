/**
 * Pipeline orchestrator: tokenize → forward → compute predictions with sampling.
 */

import * as models from './models.js';
import * as config from './config.js';

let lastLogits = null;
let lastTokens = null;
let lastModelConfig = null;
let lastPredictions = null;
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
    config.get('topP'),
    config.get('greedy')
  );
  lastPredictions = predictions;

  return { tokens, predictions, modelConfig: lastModelConfig };
}

/**
 * Recompute predictions with new temperature/top-k/top-p (no re-inference needed).
 */
export function recomputePredictions() {
  if (!lastLogits) return null;
  lastPredictions = computePredictions(
    lastLogits,
    config.get('temperature'),
    config.get('topK'),
    config.get('topP'),
    config.get('greedy')
  );
  return lastPredictions;
}

/**
 * Generate one more token: append the currently sampled prediction
 * (the one marked with ★ in the viz) and return the new text.
 */
export async function generateMore() {
  if (!lastPredictions || !lastTokens) return null;

  const sampled = lastPredictions.find(p => p.isSampled) || lastPredictions[0];
  const newText = currentText + sampled.word;
  return { newText, topWord: sampled.word };
}

/**
 * Compute top-k predictions from logits with temperature, top-p nucleus filtering, and sampling.
 */
function computePredictions(logits, temperature, topK, topP, greedy = false) {
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

  // 6. Sample one token from nucleus (greedy: always the most probable)
  if (greedy) {
    nucleus[0].isSampled = true;
  } else {
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
  }

  return predictions;
}

/**
 * Distribution snapshot for the live sampling panel: top-n candidates by
 * RAW logit (stable order while sliders move) with softmax(T) over them.
 * Nucleus/sampled flags mirror the current predictions so the panel and
 * the canvas always agree.
 */
export function getDistribution(n = 20) {
  if (!lastLogits) return null;
  const temperature = config.get('temperature');
  const topK = config.get('topK');
  const topP = config.get('topP');

  // Top-n indices by raw logit
  const indexed = [];
  for (let i = 0; i < lastLogits.length; i++) {
    indexed.push({ rawLogit: lastLogits[i], idx: i });
  }
  indexed.sort((a, b) => b.rawLogit - a.rawLogit);
  const top = indexed.slice(0, n);

  // Softmax with temperature over the top-n
  const t = Math.max(temperature, 0.01);
  const maxVal = top[0].rawLogit / t;
  let sum = 0;
  const exps = top.map(item => {
    const e = Math.exp(item.rawLogit / t - maxVal);
    sum += e;
    return e;
  });

  const flags = new Map((lastPredictions || []).map(p => [p.tokenId, p]));

  const items = top.map((item, rank) => {
    const pred = flags.get(item.idx);
    return {
      word: models.decodeToken(item.idx),
      tokenId: item.idx,
      prob: exps[rank] / sum,
      rawLogit: item.rawLogit,
      rank,
      cutByTopK: rank >= topK,
      inNucleus: pred ? pred.inNucleus : false,
      isSampled: pred ? pred.isSampled : false,
    };
  });

  return { items, temperature, topK, topP, greedy: config.get('greedy') };
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
