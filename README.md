# Anatomía de una predicción — LLM Pipeline Visualizer

Interactive, scrollytelling-style explainer that shows step by step how a Large Language Model (DistilGPT-2) processes text and chooses the next word. Everything runs directly in the browser, no server required — and every number on screen is real model data.

**[Live Demo](https://mahiler1909.github.io/llm-pipeline-visualizer/)**

## What it does

You type a prompt once and walk down through twelve full-screen sections, each teaching exactly one concept with one live widget:

**Texto → ¿Qué es un LLM? → Tokens → Embeddings → Posición → Atención → El bloque transformer → Logits → Muestreo → El bucle → Entrenamiento → Límites**

The loop section appends the sampled token to your text and the whole page re-runs — autoregressive generation you can *feel*, one word per cycle. The closing sections cover where the weights come from (pre-training → fine-tuning → inference) and what an LLM is *not* (a live hallucination demo).

Every section is slide-like: a quotable **definition card**, three **keyword blocks**, a one-line **analogy**, a collapsible «Profundizar» block with the dense theory and formulas, a one-line «En el mundo real» connection to APIs and products, and a hands-on «Pruébalo» experiment. The full spoken narration lives in [`GUION.md`](GUION.md) — the presenter's script.

## Features

- **Real in-browser inference** — DistilGPT-2 running via Transformers.js (ONNX), not a simulation. The model downloads in the background; the first sections work immediately with just the tokenizer.
- **Tokens** — Your prompt split into real BPE pieces with their IDs, plus a live mini-tokenizer to try any word.
- **Real embeddings** — The actual `wte` rows fetched on demand from the original safetensors via HTTP Range requests (~3KB per token), shown as a 48-dimension bar strip plus a cosine-similarity matrix between your tokens.
- **Real positional embeddings** — The actual `wpe` rows, same Range-request trick: pick a token and a position and watch `wte[token] + wpe[position]` change — the reason "dog bites man" ≠ "man bites dog".
- **Hallucination demo** — Curated prompts run live against the model: a frequent fact comes out at 72% confidence, an invented 2030 World Cup winner comes out of the same mechanism. The lesson teaches itself.
- **Real layer-0 attention** — The ~7MB of layer-0 weights download lazily when you reach the section (persisted in the Cache API); one query row at a time, per-head or averaged, with percentages on the most-attended tokens.
- **Logits + softmax** — Top-15 candidates with raw logits and their *true* probabilities (softmax over the full 50k vocabulary, not renormalized); the temperature slider reshapes the bars instantly with no re-inference.
- **Sampling roulette** — Surviving nucleus candidates as a stacked probability bar; top-k / top-p sliders visibly eliminate candidates; «Muestrear» re-rolls the dice on the same distribution; greedy toggle.
- **The loop** — One button appends the chosen token and repeats the cycle, with a sticky context bar tracking the growing text and cycle count.
- **Educational content in Spanish** — Definition cards, keyword blocks, analogies, real-world connections, formulas and hands-on experiments in every section; a full presenter script in `GUION.md`.
- **Shareable journeys** — The prompt is encoded in the URL (`?p=…`), so any walkthrough can be sent as a link.
- **Presentation mode** — Add `?presentar` to the URL (or press `P`) and the content reveals step by step with a fade, advancing with Space / arrow keys / a presenter clicker (PageDown). Widgets stay fully interactive for live demos; press `P` or `Esc` to exit.

## How to use

No installation, build step, or local dependencies needed. Just a static server:

```bash
# Option 1: Python
python3 -m http.server 8080

# Option 2: Node
npx serve .

# Option 3: VS Code
# Install the "Live Server" extension and click "Go Live"
```

Open `http://localhost:8080` in your browser.

> The first load downloads DistilGPT-2 (~165MB, fp16) in the background. It's cached by the browser for subsequent visits. Other GPT-2 variants are available via `?model=gpt2`, `?model=gpt2-medium`, or `?model=gpt2-large`.

## Stack

| | Technology |
|---|---|
| Frontend | Vanilla JavaScript (ES modules), DOM + SVG (no canvas) |
| Inference | [Transformers.js](https://huggingface.co/docs/transformers.js) v3 |
| Model | DistilGPT-2 ([onnx-community](https://huggingface.co/onnx-community)) |
| Styles | Single CSS file, no preprocessor |
| Build | None |

## Structure

```
├── index.html              # Single page: rail + sticky bar + 12 sections
├── js/
│   ├── main.js             # Orchestrator: state, model boot, autoregressive cycle
│   ├── content.js          # Educational prose (Spanish), templated on model config
│   ├── presenter.js        # Presentation mode (?presentar / P key)
│   ├── stages/             # One UI module per section (llm, tokens, embeddings,
│   │                       #   posicion, atencion, bloque, logits, muestreo,
│   │                       #   bucle, entrenamiento, limites, shared helpers)
│   ├── pipeline.js         # Tokenize → infer → sample (+ stable distribution)
│   ├── models.js           # Model loading via Transformers.js
│   ├── weights.js          # Real weights via Range requests over safetensors
│   ├── attention.js        # Real layer-0 attention computed in JS
│   ├── config.js           # Reactive state (pub/sub)
│   └── utils.js            # Utilities (softmax, cosine similarity)
├── css/style.css           # Paper-like, typography-led theme
└── assets/                 # Favicon
```

## Browser requirements

- Recent Chrome, Firefox, or Safari
- WebAssembly enabled
- ~200MB of storage for model cache

## License

Educational project. Models courtesy of [Hugging Face](https://huggingface.co) and [onnx-community](https://huggingface.co/onnx-community).
