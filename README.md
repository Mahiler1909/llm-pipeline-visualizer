# LLM Pipeline Visualizer

Interactive visualizer that shows step by step how a Large Language Model (GPT-2) processes text and generates the next word. Everything runs directly in the browser, no server required.

**[Live Demo](https://mahiler1909.github.io/llm-pipeline-visualizer/)**

![LLM Pipeline Visualizer](assets/screenshot.png)

## What it does

You type a prompt, the model processes it, and you see each pipeline stage in real time:

**Text → Tokens → Embeddings → Transformer (12 layers) → Logits → Sampling → Next token**

Each stage is rendered as an interactive network on Canvas with animations, zoom/pan, and educational tooltips.

## Features

- **Full pipeline visualization** — Animated network with nodes for each token, layer, and prediction
- **Real in-browser inference** — GPT-2 running via Transformers.js (ONNX), not a simulation
- **4 available models** — DistilGPT-2 (82M), GPT-2 (124M), GPT-2 Medium (355M), GPT-2 Large (774M)
- **Interactive sampling** — Adjust temperature, top-k, and top-p in real time and see how probabilities change
- **Autoregressive generation** — "Auto-generate" button that runs multiple steps with token travel animation
- **Educational panel** — Hover over each zone to see an explanation of what that stage does
- **Embedding explorer** — Click on an embedding node to see its vector as a heatmap
- **Collapsible sidebar** — Side panel with model info and configuration

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

> The first load downloads the model (~500MB for GPT-2). It's cached in the browser for subsequent visits.

## Stack

| | Technology |
|---|---|
| Frontend | Vanilla JavaScript (ES modules) |
| Inference | [Transformers.js](https://huggingface.co/docs/transformers.js) v3 |
| Models | GPT-2 family ([onnx-community](https://huggingface.co/onnx-community)) |
| Visualization | Canvas 2D |
| Styles | Modular CSS (no preprocessor) |
| Build | None |

## Structure

```
├── index.html          # Single page
├── js/
│   ├── app.js          # Main orchestrator
│   ├── pipeline.js     # Tokenize → infer → sample
│   ├── models.js       # Model loading via Transformers.js
│   ├── viz.js          # Canvas visualization engine
│   ├── config.js       # Reactive state (pub/sub)
│   └── utils.js        # Utilities (softmax, colors, etc)
├── css/                # One CSS file per component
└── assets/             # Favicon
```

## Supported models

| Model | Parameters | Layers | Dimension | Type |
|-------|-----------|--------|-----------|------|
| DistilGPT-2 | 82M | 6 | 768 | fp32 |
| GPT-2 | 124M | 12 | 768 | fp32 |
| GPT-2 Medium | 355M | 24 | 1024 | q8 |
| GPT-2 Large | 774M | 36 | 1280 | q8 |

## Browser requirements

- Recent Chrome, Firefox, or Safari
- WebAssembly enabled
- ~500MB of storage for model cache

## License

Educational project. Models courtesy of [Hugging Face](https://huggingface.co) and [onnx-community](https://huggingface.co/onnx-community).
