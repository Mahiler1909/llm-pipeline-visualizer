/**
 * Model definitions and Transformers.js v3 integration.
 * Loads real GPT-2 family models for browser inference.
 */

let transformers = null;

async function loadTransformers() {
  if (!transformers) {
    transformers = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
    transformers.env.allowLocalModels = false;
  }
  return transformers;
}

export const MODEL_CONFIGS = {
  'onnx-community/distilgpt2-ONNX': {
    name: 'DistilGPT-2',
    layers: 6,
    hidden_dim: 768,
    ffn_dim: 3072,
    vocab_size: 50257,
    params: '82M',
    heads: 12,
    dtype: 'fp32',
  },
  'onnx-community/gpt2-ONNX': {
    name: 'GPT-2',
    layers: 12,
    hidden_dim: 768,
    ffn_dim: 3072,
    vocab_size: 50257,
    params: '124M',
    heads: 12,
    dtype: 'fp32',
  },
  'onnx-community/gpt2-medium-ONNX': {
    name: 'GPT-2 Medium',
    layers: 24,
    hidden_dim: 1024,
    ffn_dim: 4096,
    vocab_size: 50257,
    params: '355M',
    heads: 16,
    dtype: 'q8',
  },
  'onnx-community/gpt2-large-ONNX': {
    name: 'GPT-2 Large',
    layers: 36,
    hidden_dim: 1280,
    ffn_dim: 5120,
    vocab_size: 50257,
    params: '774M',
    heads: 20,
    dtype: 'q8',
  },
};

let currentTokenizer = null;
let currentModel = null;
let currentModelId = null;

export function getConfig(modelId) {
  return MODEL_CONFIGS[modelId];
}

export function getLoadedModelId() {
  return currentModelId;
}

export function getAvailableModels() {
  return Object.entries(MODEL_CONFIGS).map(([id, cfg]) => ({ id, ...cfg }));
}

/**
 * Load a model and tokenizer. Shows progress via callback.
 */
export async function loadModel(modelId, onProgress) {
  const config = MODEL_CONFIGS[modelId];
  if (!config) throw new Error(`Unknown model: ${modelId}`);

  onProgress?.({ phase: 'init', message: 'Inicializando Transformers.js...' });
  const tf = await loadTransformers();

  // Load tokenizer
  onProgress?.({ phase: 'tokenizer', message: 'Cargando tokenizer...' });
  currentTokenizer = await tf.AutoTokenizer.from_pretrained(modelId);

  // Load model with progress (track total bytes across all files)
  onProgress?.({ phase: 'model', message: 'Descargando modelo...', progress: 0 });

  const fileProgress = {}; // { filename: { loaded, total } }

  currentModel = await tf.AutoModelForCausalLM.from_pretrained(modelId, {
    dtype: config.dtype || 'fp32',
    progress_callback: (p) => {
      if (p.status === 'initiate' && p.name) {
        fileProgress[p.name] = { loaded: 0, total: 0 };
      }
      if (p.status === 'progress' && p.name) {
        fileProgress[p.name] = { loaded: p.loaded || 0, total: p.total || 0 };
        let totalLoaded = 0, totalSize = 0;
        for (const f of Object.values(fileProgress)) {
          totalLoaded += f.loaded;
          totalSize += f.total;
        }
        const overall = totalSize > 0 ? (totalLoaded / totalSize) * 100 : 0;
        onProgress?.({
          phase: 'model',
          message: `Descargando modelo... ${Math.round(overall)}%`,
          progress: overall,
        });
      }
      if (p.status === 'done' && p.name) {
        const f = fileProgress[p.name];
        if (f) f.loaded = f.total;
      }
    },
  });

  currentModelId = modelId;
  onProgress?.({ phase: 'ready', message: 'Modelo listo' });
  return config;
}

/**
 * Tokenize text using the loaded tokenizer.
 * Returns { tokens: [{id, text}], encoded: {input_ids, attention_mask} }
 */
export function tokenize(text) {
  if (!currentTokenizer) throw new Error('Tokenizer not loaded');
  const encoded = currentTokenizer(text);
  const ids = Array.from(encoded.input_ids.data).map(Number);
  const tokens = ids.map(id => ({
    id,
    text: currentTokenizer.decode([id]),
  }));
  return { tokens, encoded };
}

/**
 * Run forward pass and return logits for the last position.
 * Returns { logits: Float32Array, vocabSize: number }
 */
export async function forward(encoded) {
  if (!currentModel) throw new Error('Model not loaded');
  const config = MODEL_CONFIGS[currentModelId];

  // Yield to event loop before heavy ONNX inference
  await new Promise(r => setTimeout(r, 10));

  const output = await currentModel(encoded);

  const vocabSize = config.vocab_size;
  // Handle both v2 tensor shapes
  const dims = encoded.input_ids.dims || [1, encoded.input_ids.data.length];
  const seqLen = dims[1] || dims[0];
  const start = (seqLen - 1) * vocabSize;
  const rawLogits = output.logits.data;
  const lastLogits = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    lastLogits[i] = Number(rawLogits[start + i]);
  }

  return { logits: lastLogits, vocabSize };
}

/**
 * Decode a token ID back to text.
 */
export function decodeToken(id) {
  if (!currentTokenizer) return `[${id}]`;
  return currentTokenizer.decode([id]);
}

/**
 * Generate simulated embedding for visualization.
 * Real embeddings are inside the ONNX model and not directly accessible,
 * so we generate deterministic vectors with the correct dimensions.
 */
export function getEmbeddingVector(tokenId, dims) {
  const { seededRandom } = await_utils();
  const rng = seededRandom(tokenId * 7919 + 31);
  const vec = new Float32Array(dims);
  for (let i = 0; i < dims; i++) {
    vec[i] = (rng() * 2 - 1) * 0.15;
  }
  return vec;
}

// Lazy import utils to avoid circular deps
let _utils = null;
function await_utils() {
  if (!_utils) {
    // Inline the seededRandom to avoid import issues
    _utils = {
      seededRandom(seed) {
        let t = (seed >>> 0) + 0x6D2B79F5;
        return function () {
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
    };
  }
  return _utils;
}
