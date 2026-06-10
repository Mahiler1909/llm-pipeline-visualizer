/**
 * Fetch real model weights from HuggingFace Hub via HTTP Range requests
 * over the original safetensors files. The ONNX model only exposes logits,
 * so real embeddings / layer weights are read directly from the repo:
 * one safetensors header fetch (~14KB, cached) + per-row fetches (~3KB/token).
 */

// ONNX repo (inference) → original repo (weights in safetensors)
const REPO_MAP = {
  'onnx-community/distilgpt2-ONNX': 'distilbert/distilgpt2',
  'onnx-community/gpt2-ONNX': 'openai-community/gpt2',
  'onnx-community/gpt2-medium-ONNX': 'openai-community/gpt2-medium',
  'onnx-community/gpt2-large-ONNX': 'openai-community/gpt2-large',
};

const headerCache = new Map(); // repo → Promise<{ header, dataStart, url }>
const rowCache = new Map();    // `${repo}:${tensorKey}:${row}` → Float32Array

function repoFor(modelId) {
  const repo = REPO_MAP[modelId];
  if (!repo) throw new Error(`Sin repo de pesos para ${modelId}`);
  return repo;
}

function safetensorsUrl(repo) {
  return `https://huggingface.co/${repo}/resolve/main/model.safetensors`;
}

async function fetchRange(url, start, end) {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok) throw new Error(`Range fetch fallo (${res.status})`);
  return res.arrayBuffer();
}

/**
 * Load and parse the safetensors header for a repo.
 * Format: 8 bytes u64 LE header length, then JSON header with
 * { tensorName: { dtype, shape, data_offsets: [start, end] } }.
 * Data offsets are relative to dataStart = 8 + headerLen.
 */
function loadHeader(modelId) {
  const repo = repoFor(modelId);
  if (headerCache.has(repo)) return headerCache.get(repo);

  const promise = (async () => {
    const url = safetensorsUrl(repo);
    const lsKey = `hf-st-header:${repo}`;

    let headerLen, headerJson;
    try {
      const cached = localStorage.getItem(lsKey);
      if (cached) {
        const parsed = JSON.parse(cached);
        headerLen = parsed.headerLen;
        headerJson = parsed.header;
      }
    } catch { /* localStorage no disponible o corrupto */ }

    if (!headerJson) {
      const lenBuf = await fetchRange(url, 0, 7);
      headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
      const headerBuf = await fetchRange(url, 8, 8 + headerLen - 1);
      headerJson = JSON.parse(new TextDecoder().decode(headerBuf));
      try {
        localStorage.setItem(lsKey, JSON.stringify({ headerLen, header: headerJson }));
      } catch { /* quota llena: seguir sin persistir */ }
    }

    return { header: headerJson, dataStart: 8 + headerLen, url };
  })();

  headerCache.set(repo, promise);
  promise.catch(() => headerCache.delete(repo));
  return promise;
}

/**
 * Find a tensor by name suffix (tolerates repo-specific prefixes
 * like "transformer.wte.weight" vs "wte.weight"). Validates F32.
 */
function findTensor(header, suffix) {
  let best = null;
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (name === suffix || name.endsWith('.' + suffix)) {
      if (!best || name.length < best.name.length) best = { name, ...info };
    }
  }
  if (!best) throw new Error(`Tensor "${suffix}" no encontrado en el header`);
  if (best.dtype !== 'F32') throw new Error(`Tensor "${best.name}" es ${best.dtype}, se esperaba F32`);
  return best;
}

/**
 * Fetch `count` contiguous rows of a 2D tensor starting at `startRow`.
 * Returns Float32Array(count * rowDim).
 */
export async function getRows(modelId, suffix, startRow, count) {
  const { header, dataStart, url } = await loadHeader(modelId);
  const tensor = findTensor(header, suffix);
  const [numRows, rowDim] = tensor.shape;
  if (startRow < 0 || startRow + count > numRows) {
    throw new Error(`Filas ${startRow}..${startRow + count} fuera de rango (${numRows})`);
  }
  const rowBytes = rowDim * 4;
  const start = dataStart + tensor.data_offsets[0] + startRow * rowBytes;
  const buf = await fetchRange(url, start, start + count * rowBytes - 1);
  if (buf.byteLength !== count * rowBytes) {
    throw new Error(`Respuesta incompleta: ${buf.byteLength} bytes (el servidor ignoro Range?)`);
  }
  return new Float32Array(buf);
}

/**
 * Fetch a full (small) tensor by suffix. Returns { data: Float32Array, shape }.
 */
export async function getTensor(modelId, suffix) {
  const { header, dataStart, url } = await loadHeader(modelId);
  const tensor = findTensor(header, suffix);
  const start = dataStart + tensor.data_offsets[0];
  const end = dataStart + tensor.data_offsets[1] - 1;
  const buf = await fetchRange(url, start, end);
  return { data: new Float32Array(buf), shape: tensor.shape };
}

/**
 * Get tensor metadata (shape, byte size) without fetching data.
 */
export async function getTensorInfo(modelId, suffix) {
  const { header } = await loadHeader(modelId);
  const t = findTensor(header, suffix);
  return { name: t.name, shape: t.shape, byteLength: t.data_offsets[1] - t.data_offsets[0] };
}

/**
 * Fetch a large tensor with download progress. Returns Float32Array.
 */
export async function getTensorWithProgress(modelId, suffix, onProgress) {
  const { header, dataStart, url } = await loadHeader(modelId);
  const tensor = findTensor(header, suffix);
  const start = dataStart + tensor.data_offsets[0];
  const total = tensor.data_offsets[1] - tensor.data_offsets[0];

  const res = await fetch(url, { headers: { Range: `bytes=${start}-${start + total - 1}` } });
  if (!res.ok) throw new Error(`Range fetch fallo (${res.status})`);

  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value.subarray(0, Math.min(value.length, total - loaded)), loaded);
    loaded += value.length;
    onProgress?.({ loaded: Math.min(loaded, total), total });
    if (loaded >= total) break;
  }
  if (loaded < total) throw new Error('Descarga incompleta');
  return new Float32Array(out.buffer);
}

/**
 * Get the REAL embedding row (wte.weight[tokenId]) for a token.
 * Cached in memory; ~3KB per token over the network.
 */
export async function getEmbeddingRow(modelId, tokenId) {
  const repo = repoFor(modelId);
  const key = `${repo}:wte:${tokenId}`;
  if (rowCache.has(key)) return rowCache.get(key);
  const row = await getRows(modelId, 'wte.weight', tokenId, 1);
  rowCache.set(key, row);
  return row;
}

/**
 * Whether real weights are reachable for this model (repo mapped).
 * Network availability is only known after the first fetch attempt.
 */
export function hasRepo(modelId) {
  return Boolean(REPO_MAP[modelId]);
}
