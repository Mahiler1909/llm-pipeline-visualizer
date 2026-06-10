# Anatomía de una predicción — LLM Pipeline Visualizer

Interactive, scrollytelling-style explainer that shows step by step how a Large Language Model (DistilGPT-2) processes text and chooses the next word. Everything runs directly in the browser, no server required — and every number on screen is real model data.

**[Live Demo](https://mahiler1909.github.io/llm-pipeline-visualizer/)**

## What it does

You type a prompt once and walk down through seven full-screen sections, each teaching exactly one concept with one live widget:

**Texto → Tokens → Embeddings → Atención → Logits → Muestreo → El bucle**

The last section closes the loop: the sampled token is appended to your text and the whole page re-runs — autoregressive generation you can *feel*, one word per cycle.

## Features

- **Real in-browser inference** — DistilGPT-2 running via Transformers.js (ONNX), not a simulation. The model downloads in the background; the first sections work immediately with just the tokenizer.
- **Tokens** — Your prompt split into real BPE pieces with their IDs, plus a live mini-tokenizer to try any word.
- **Real embeddings** — The actual `wte` rows fetched on demand from the original safetensors via HTTP Range requests (~3KB per token), shown as a 48-dimension bar strip plus a cosine-similarity matrix between your tokens.
- **Real layer-0 attention** — The ~7MB of layer-0 weights download lazily when you reach the section (persisted in the Cache API); one query row at a time, per-head or averaged, with percentages on the most-attended tokens.
- **Logits + softmax** — Top-15 candidates with raw logits and their *true* probabilities (softmax over the full 50k vocabulary, not renormalized); the temperature slider reshapes the bars instantly with no re-inference.
- **Sampling roulette** — Surviving nucleus candidates as a stacked probability bar; top-k / top-p sliders visibly eliminate candidates; «Muestrear» re-rolls the dice on the same distribution; greedy toggle.
- **The loop** — One button appends the chosen token and repeats the cycle, with a sticky context bar tracking the growing text and cycle count.
- **Educational prose in Spanish** — Each section pairs its widget with a plain-language explanation, a collapsible «Profundizar» block with the actual formulas, and a hands-on «Pruébalo» experiment.
- **Shareable journeys** — The prompt is encoded in the URL (`?p=…`), so any walkthrough can be sent as a link.

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
├── index.html              # Single page: rail + sticky bar + 7 sections
├── js/
│   ├── main.js             # Orchestrator: state, model boot, autoregressive cycle
│   ├── content.js          # Educational prose (Spanish), templated on model config
│   ├── stages/             # One UI module per section (tokens, embeddings,
│   │                       #   atencion, logits, muestreo, bucle, shared helpers)
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
