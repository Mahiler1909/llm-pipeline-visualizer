/**
 * Real layer-0 self-attention computed in JS.
 * The ONNX model only exposes logits, so we fetch the actual layer-0
 * weights (ln_1 + c_attn, ~7MB one-time for GPT-2) from the original
 * safetensors on HuggingFace Hub and replicate the attention math:
 *   x = wte[token] + wpe[pos]
 *   h = LayerNorm(x) * ln_1.weight + ln_1.bias
 *   [q,k,v] = h @ c_attn.weight + c_attn.bias   (Conv1D: weight is (in, out))
 *   attn[h][i][j] = softmax_j( q_i · k_j / sqrt(headDim) )  for j <= i
 */

import * as weights from './weights.js';

const memCache = new Map(); // modelId → { ln1w, ln1b, cattnB, cattnW, dim }
const wpeCache = new Map(); // modelId → Float32Array (prefix of wpe rows)

const CACHE_NAME = 'llm-viz-weights';

function cacheKey(modelId) {
  return `https://weights.llm-viz.local/${encodeURIComponent(modelId)}/h0_c_attn_w`;
}

async function cacheGet(key) {
  if (typeof caches === 'undefined') return null;
  try {
    const c = await caches.open(CACHE_NAME);
    const res = await c.match(key);
    return res ? new Float32Array(await res.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

async function cachePut(key, f32) {
  if (typeof caches === 'undefined') return;
  try {
    const c = await caches.open(CACHE_NAME);
    await c.put(key, new Response(f32.buffer.slice(0)));
  } catch { /* sin espacio o contexto no seguro: seguir sin persistir */ }
}

export function isLoaded(modelId) {
  return memCache.has(modelId);
}

/**
 * Download layer-0 weights once. onProgress({loaded, total}) reports the
 * large c_attn fetch. Persisted in the Cache API so reloads are instant.
 */
export async function loadLayer0(modelId, onProgress) {
  if (memCache.has(modelId)) return memCache.get(modelId);

  const [ln1w, ln1b, cattnB] = await Promise.all([
    weights.getTensor(modelId, 'h.0.ln_1.weight'),
    weights.getTensor(modelId, 'h.0.ln_1.bias'),
    weights.getTensor(modelId, 'h.0.attn.c_attn.bias'),
  ]);

  const key = cacheKey(modelId);
  let cattnW = await cacheGet(key);
  if (cattnW) {
    onProgress?.({ loaded: 1, total: 1 });
  } else {
    cattnW = await weights.getTensorWithProgress(modelId, 'h.0.attn.c_attn.weight', onProgress);
    await cachePut(key, cattnW);
  }

  const layer = {
    ln1w: ln1w.data,
    ln1b: ln1b.data,
    cattnB: cattnB.data,
    cattnW,
    dim: ln1w.data.length,
  };
  memCache.set(modelId, layer);
  return layer;
}

/**
 * Compute the real layer-0 attention matrix for a token sequence.
 * Returns { heads, seqLen, data, get(h,i,j), avg(i,j) } where rows are
 * queries (token that looks) and columns are keys (token looked at).
 */
export async function computeAttention(modelId, tokenIds, numHeads) {
  const L = await loadLayer0(modelId);
  const n = tokenIds.length;
  const dim = L.dim;
  const headDim = dim / numHeads;
  const eps = 1e-5;

  // Positional embeddings: one contiguous fetch, cached as a prefix
  let wpe = wpeCache.get(modelId);
  if (!wpe || wpe.length < n * dim) {
    wpe = await weights.getRows(modelId, 'wpe.weight', 0, n);
    wpeCache.set(modelId, wpe);
  }
  const embRows = await Promise.all(tokenIds.map(t => weights.getEmbeddingRow(modelId, t)));

  // Q and K for every position (V is not needed for attention weights)
  const Q = new Float32Array(n * dim);
  const K = new Float32Array(n * dim);
  const x = new Float32Array(dim);
  const h = new Float32Array(dim);
  const acc = new Float32Array(2 * dim);
  const stride = 3 * dim;

  for (let i = 0; i < n; i++) {
    const wte = embRows[i];
    let mean = 0;
    for (let d = 0; d < dim; d++) {
      x[d] = wte[d] + wpe[i * dim + d];
      mean += x[d];
    }
    mean /= dim;
    let variance = 0;
    for (let d = 0; d < dim; d++) {
      const dv = x[d] - mean;
      variance += dv * dv;
    }
    const inv = 1 / Math.sqrt(variance / dim + eps);
    for (let d = 0; d < dim; d++) {
      h[d] = (x[d] - mean) * inv * L.ln1w[d] + L.ln1b[d];
    }

    // First 2*dim outputs of c_attn are Q and K
    acc.set(L.cattnB.subarray(0, 2 * dim));
    for (let k = 0; k < dim; k++) {
      const hk = h[k];
      const base = k * stride;
      for (let j = 0; j < 2 * dim; j++) {
        acc[j] += hk * L.cattnW[base + j];
      }
    }
    Q.set(acc.subarray(0, dim), i * dim);
    K.set(acc.subarray(dim, 2 * dim), i * dim);

    // Keep the UI responsive on long sequences
    if (n > 64 && (i & 7) === 7) await new Promise(r => setTimeout(r));
  }

  // Scaled dot-product attention with causal mask, softmax per row
  const data = new Float32Array(numHeads * n * n);
  const scale = 1 / Math.sqrt(headDim);
  const row = new Float64Array(n);

  for (let hd = 0; hd < numHeads; hd++) {
    const off = hd * headDim;
    const base = hd * n * n;
    for (let i = 0; i < n; i++) {
      let maxv = -Infinity;
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let d = 0; d < headDim; d++) {
          s += Q[i * dim + off + d] * K[j * dim + off + d];
        }
        s *= scale;
        row[j] = s;
        if (s > maxv) maxv = s;
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        row[j] = Math.exp(row[j] - maxv);
        sum += row[j];
      }
      for (let j = 0; j <= i; j++) {
        data[base + i * n + j] = row[j] / sum;
      }
    }
  }

  return {
    heads: numHeads,
    seqLen: n,
    data,
    get(hd, i, j) {
      return data[hd * n * n + i * n + j];
    },
    avg(i, j) {
      let s = 0;
      for (let hd = 0; hd < numHeads; hd++) s += data[hd * n * n + i * n + j];
      return s / numHeads;
    },
  };
}
