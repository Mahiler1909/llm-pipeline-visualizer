# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

«Anatomía de una predicción»: a scrollytelling, browser-based LLM explainer. Runs DistilGPT-2 in-browser via Transformers.js (ONNX) and walks the user through seven full-screen sections — Texto → Tokens → Embeddings → Atención → Logits → Muestreo → El bucle — each pairing Spanish educational prose with one live widget fed by real model data. The last section appends the sampled token and re-runs everything (autoregression made visible).

## Development

No build system, bundler, or package manager. Serve with any static server (`python3 -m http.server`). Cache-busting via `?v=N` on the script/style tags in `index.html` — bump when changing JS/CSS.

Transformers.js is loaded from CDN (`cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1`). No local dependencies.

## Architecture

**Data flow:** `js/main.js` owns the central `state` (text, tokens, predictions, history, cycle) and orchestrates `runCycle(text)`: tokenize (works as soon as the tokenizer downloads, before the ONNX model finishes) → update front stages (tokens/embeddings/atención) → await model → `pipeline.run` → update back stages (logits/muestreo/bucle).

**ML core (UI-agnostic, reusable):**
- `js/models.js` — Transformers.js wrapper. MODEL_CONFIGS for 4 GPT-2 variants; the UI uses DistilGPT-2 fixed, with `?model=gpt2|gpt2-medium|gpt2-large` URL escape hatch. Note: `loadModel`'s `onProgress` reports phase `'model'` once the tokenizer is ready — `main.js` uses that to resolve `tokenizerReady` early.
- `js/pipeline.js` — tokenize → forward → sampling. Caches logits so `recomputePredictions()` re-derives (and re-samples) without re-inference — this powers the sliders AND the «Muestrear» button. `getDistribution(n)` returns top-n by raw logit (stable order while sliding).
- `js/weights.js` — real weight rows from HF Hub via HTTP Range requests over the original `model.safetensors` (REPO_MAP maps ONNX ids to original repos).
- `js/attention.js` — real layer-0 attention computed in JS (~7MB one-time fetch, persisted in Cache API).
- `js/config.js` — pub/sub store for temperature/topK/topP/greedy. `main.js` subscribes: any change → `recomputePredictions()` → update logits/muestreo/bucle.

**UI (one module per section):** each `js/stages/*.js` exports `init(rootEl, …)` and `update(state)`. Sliders/controls are built ONCE in `init` (re-rendering them mid-drag breaks the drag); `update` only refreshes data nodes. `js/stages/shared.js` has helpers (esc, tokenLabel with ␣, pastel palette, divergent color scale). `js/content.js` holds all educational prose as template functions receiving the model config.

## Key Patterns

- **No floating tooltips:** each widget has a fixed `.widget-caption` line that hover writes into.
- **Lazy attention:** an IntersectionObserver on `#st-atencion` triggers the 7MB download only when the section approaches the viewport, with an inline progress bar.
- **Scroll with fallback:** `scrollToSection()` in main.js tries smooth scrollIntoView, then after 700ms jumps instantly if not near the target (smooth scrolling can be disabled/cancelled by the environment or scroll anchoring during re-renders). Always use it instead of raw scrollIntoView.
- **Async race guards:** embeddings/atención fetches use a request-id counter so stale responses don't overwrite newer renders.
- **`hidden` vs CSS display:** any element styled with `display:flex/grid` needs an explicit `[hidden] { display:none }` rule (see `.context-bar`).
- **Token cap on long texts:** atención shows the last 12 tokens, similarity matrix the last 10 unique; the sticky context bar always shows the full text (RTL-ellipsis trick keeps the end visible).

## UI Language

All user-facing text is in **Spanish** (with proper accents). Prompts/examples are in English because GPT-2 was trained on English.
