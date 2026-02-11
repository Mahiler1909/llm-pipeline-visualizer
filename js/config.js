/**
 * Reactive configuration state with pub/sub.
 */

const state = {
  temperature: 1.0,
  topK: 10,
  topP: 0.9,
  modelId: 'onnx-community/gpt2-ONNX',
};

const listeners = new Set();

export function get(key) {
  return state[key];
}

export function set(key, value) {
  if (state[key] === value) return;
  state[key] = value;
  listeners.forEach(fn => fn(key, value));
}

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function getAll() {
  return { ...state };
}
