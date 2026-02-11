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
- `js/app.js` — Entry point. Wires DOM, manages state, handles info panel content (INFO_CARDS), embedding modal, autoregressive generation loop
- `js/pipeline.js` — ML pipeline: tokenize → forward → sampling. Caches logits so slider changes recompute without re-inference (`recomputePredictions()`)
- `js/models.js` — Transformers.js wrapper. MODEL_CONFIGS defines 4 GPT-2 variants with metadata (layers, hidden_dim, heads, etc). Progress callback tracks bytes across multiple files for accurate loading bar
- `js/viz.js` — Canvas rendering engine (largest file). Columns layout, glow animation loop, zoom/pan, hover zone detection, output probability bars, token travel animation
- `js/config.js` — Reactive state store with pub/sub (`get`/`set`/`onChange`)

**CSS is modular:** `main.css` (layout + theme variables), `sidebar.css`, `viz.css`, `input.css`, `info-panel.css`. CSS variables defined in `:root` in `main.css`.

## Key Patterns

- **Hover zones in viz.js:** 5 zones (token, embedding, transformer, logit, sampling) detected by column x-position ranges. Triggers `onHoverZoneChange` callback → app.js updates info panel and legend visibility
- **Sidebar:** `position: fixed` with `transform: translateX` for collapse animation. Body class `sidebar-hidden` toggles grid from `260px 1fr` to `0 1fr`
- **Autoregressive generation:** Loop in `startAutoGenerate()` checks `autoGenAbort` flag between each async step. `animateTokenTravel()` returns a Promise for sequencing
- **Embedding heatmap:** Floating tooltip uses the global `#tooltip` element (positioned in body) to escape sidebar's `overflow` clipping
- **Welcome state:** Canvas shows welcome overlay until first `runPipeline()` call. Reset button restores it via `viz.clear()` + `hasGenerated = false`
- **Config info icons:** Use `data-tooltip` attribute + JS mouseenter/mouseleave on `.config__info` elements, reusing `#tooltip` element

## UI Language

All user-facing text is in **Spanish**. Info panel educational content, button labels, loading messages, and tooltips are all in Spanish.
