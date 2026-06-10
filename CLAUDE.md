# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive browser-based LLM visualizer. Runs GPT-2 models in-browser via Transformers.js (ONNX) and renders the full pipeline on Canvas: Tokens → Embeddings → Transformer Layers → Logits → Sampling.

## Development

No build system, bundler, or package manager. Open `index.html` directly or serve with any static server. Cache-busting is done via `?v=N` on the script tag in `index.html` — bump the number when changing JS.

Transformers.js is loaded from CDN (`cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1`). No local dependencies to install.

## Architecture

**Data flow:** `app.js` orchestrates everything. User types text → `pipeline.js` calls `models.js` (tokenize + ONNX forward pass) → computes predictions with temperature/top-k/top-p → `viz.js` renders on Canvas.

**Key modules:**
- `js/app.js` — Entry point. Wires DOM, manages state, handles educational content (INFO_CARDS with tagline/color for the context pill + Básico/Profundizar tabs rendered in the learn drawer), embedding/similarity/layer modals, autoregressive generation loop
- `js/pipeline.js` — ML pipeline: tokenize → forward → sampling. Caches logits AND predictions so slider changes recompute without re-inference (`recomputePredictions()`). `getDistribution()` feeds the live sampling panel. `greedy` config makes sampling deterministic
- `js/models.js` — Transformers.js wrapper. MODEL_CONFIGS defines 4 GPT-2 variants with metadata (layers, hidden_dim, heads, etc). `getEmbeddingVectorAsync()` returns the REAL wte row (via weights.js) with seeded-random fallback
- `js/weights.js` — Fetches real weights from HF Hub via HTTP Range requests over the original `model.safetensors` (header parsed once, rows fetched on demand, ~3KB/token). ONNX repo IDs map to original repos in REPO_MAP
- `js/attention.js` — Real layer-0 attention computed in JS: one-time fetch of ln_1 + c_attn weights (~7MB for GPT-2, persisted in Cache API), then wte+wpe → LayerNorm → QK → causal softmax per head
- `js/sampling-panel.js` — Animated live distribution bar chart in the sidebar (top-20 candidates, top-k cut line, nucleus shading, ★ sampled token)
- `js/viz.js` — Canvas rendering engine (largest file). Columns layout, glow animation loop, zoom/pan, hover zone detection, output probability bars, token travel animation, attention arcs on token hover, layer click callback
- `js/config.js` — Reactive state store with pub/sub (`get`/`set`/`onChange`)

**CSS is modular:** `main.css` (layout + theme variables), `sidebar.css`, `viz.css`, `input.css`, `info-panel.css`. CSS variables defined in `:root` in `main.css`.

## Key Patterns

- **Hover zones in viz.js:** 5 zones (token, embedding, transformer, logit, sampling) detected by column x-position ranges. Triggers `onHoverZoneChange` callback → app.js shows a one-line context pill (`#zone-pill`, bottom-center); clicking it opens the persistent learn drawer (`#learn-drawer`, right side) with the full INFO_CARDS content. The drawer stays open and follows zone changes
- **Sidebar:** `position: fixed` with `transform: translateX` for collapse animation. Body class `sidebar-hidden` toggles grid from `260px 1fr` to `0 1fr`
- **Autoregressive generation:** Loop in `startAutoGenerate()` checks `autoGenAbort` flag between each async step. `animateTokenTravel()` returns a Promise for sequencing
- **Embedding heatmap:** Floating tooltip uses the global `#tooltip` element (positioned in body) to escape sidebar's `overflow` clipping
- **Welcome state:** Canvas shows welcome overlay until first `runPipeline()` call. Reset button restores it via `viz.clear()` + `hasGenerated = false`
- **Config info icons:** Use `data-tooltip` attribute + JS mouseenter/mouseleave on `.config__info` elements, reusing `#tooltip` element

## UI Language

All user-facing text is in **Spanish**. Info panel educational content, button labels, loading messages, and tooltips are all in Spanish.
