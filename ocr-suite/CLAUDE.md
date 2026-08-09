# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This folder holds every OCR-related deployment for the parent repo's Modal apps, sharing **one**
Gradio client (`gradio_app.py`) behind a model dropdown. Each `modal_*.py` is a fully independent
Modal app (own `modal.App(...)` name, own image/GPU) — they are not variants of one script, they
are separately deployed. `README.md` is the source of truth for what each backend does, how to
pick between them, and every measured perf number; don't duplicate that content here, read it.
Per-app deep dives live beside their scripts: `paddleocr-vl/`, `unlimited-ocr/`, and `docling/`.

Scripts are grouped into model folders (`paddleocr-vl/`, `glm-ocr/`, `mineru/`, `unlimited-ocr/`,
`docling/`). `gradio_app.py`, `Dockerfile`, `.env*` stay at the top level — they're shared across
every backend, not owned by one model.

The general repo-wide conventions (Modal app shape, `modal` CLI basics, memory snapshots,
Indonesian-language strings, numbered script variants) are documented in the root `CLAUDE.md` one
level up — this file only covers what's specific to `ocr-suite/`.

## Common commands

Run from inside `ocr-suite/`:

```bash
# Deploy a backend (each is a separate Modal app)
modal deploy paddleocr-vl/modal_paddleocr_vl_gguf.py           # app "paddleocr-vl-gguf" — default, llama.cpp
modal deploy paddleocr-vl/modal_paddleocr_vl.py                # app "paddleocr-vl" — transformers
modal deploy paddleocr-vl/modal_paddleocr_vl_gguf_official.py  # GGUF + PaddleOCR's official layout pipeline
modal deploy unlimited-ocr/modal_unlimited_ocr.py               # app "unlimited-ocr"
modal deploy glm-ocr/modal_glm_ocr_transformers.py             # app "glm-ocr-transformers" — the GLM-OCR path that actually works

# Smoke test any of them without deploying
modal run paddleocr-vl/modal_paddleocr_vl_gguf.py --image-url "<url>" --task ocr
modal run unlimited-ocr/modal_unlimited_ocr.py --image-url "<url>" --prompt "..." --mode ...

# app names come from modal.App("...") inside each script, not the filename — use those for:
modal app logs <app-name>

# Run the shared client locally (reads .env in this folder)
cp .env.example .env   # fill in the base URLs of whichever backends you deployed
uv run gradio_app.py

# Docker build of just the client (models stay remote on Modal)
docker build -t ocr-suite-ui .
docker run --rm -p 7870:7860 --env-file .env ocr-suite-ui
```

Backends whose base-URL env var is unset show as "unconfigured" in the UI rather than erroring —
a typo in `.env` looks identical to "not deployed yet", check that first.

## Architecture

- **`gradio_app.py`** — the only shared file. PEP 723 inline deps (`uv run` needs no venv setup).
  Every handler is a generator so the UI can show a "working" event before the network call
  returns (first real server event on a big PDF can be 90s+ away). Talks to each Modal backend
  over its `*_BASE_URL` env var, plus one local, no-GPU backend (OpenDataLoader, needs a Java 11+
  runtime on the machine/container).
- **`paddleocr-vl/modal_paddleocr_vl_gguf.py`** (`paddleocr-vl-gguf`) — PaddleOCR-VL-1.6-GGUF via
  `llama-server`, the default backend. DocLayout-YOLO for layout, custom page-restructuring.
- **`paddleocr-vl/modal_paddleocr_vl.py`** (`paddleocr-vl`) — same model, transformers runtime,
  kept for accuracy comparison against the GGUF path.
- **`paddleocr-vl/modal_paddleocr_vl_gguf_official.py`** — same GGUF `llama-server` recognition
  backend, but layout/restructuring comes from PaddleOCR's own `PaddleOCRVL` pipeline (CPU-side
  PP-DocLayoutV3, cached in the `paddleocr-vl-official-paddlex-cache` Volume) instead of the custom
  DocLayout-YOLO parser. Separate app, deployed side by side with `paddleocr-vl/modal_paddleocr_vl_gguf.py`
  for comparison — does not replace it.
- **`unlimited-ocr/modal_unlimited_ocr.py`** (`unlimited-ocr`) — `baidu/Unlimited-OCR`, whole-page
  prompt-steerable VLM parse.
- **`Dockerfile`** — builds only the Gradio client (with a JRE for OpenDataLoader); every model
  backend stays remote on Modal and is reached via the env vars in `.env`.

## Gotchas worth knowing

- **`table` task returns OTSL**, not HTML/Markdown. `/parse-page` converts server-side; the
  client's `/recognize` path still needs `otsl_to_html()`. Never feed raw OTSL to a Markdown pane.
- **Merged table cells need `markdown_with_html`**, not plain markdown (which has no span concept)
  — set in `_odl_run` in `gradio_app.py`.
- **Payload size dominates wall clock** far more than model speed: prefer passing a URL over
  base64 upload (measured 51s server time vs 144s end-to-end on a 27-page PDF from upload
  overhead alone). The client re-encodes images above `JPEG_ABOVE_KB` to JPEG for this reason.
- **Cold starts can reset TLS** — a request landing mid-container-start can fail the handshake
  (`ConnectionReset`) rather than time out; retry once the backend is warm.
- The GGUF app's memory/GPU snapshot needs `--no-mmap`; the transformers and Unlimited-OCR
  snapshot phases (`snap=True`) must stay CPU-only or the snapshot can't restore — see root
  `CLAUDE.md` for the general pattern.
- **`glm-ocr/` — three backends for the same model, only two of which work.** All three wrap the
  `glmocr[selfhosted]` SDK (layout via PP-DocLayoutV3, region batching) around a different serving
  engine, matching this repo's own `ocr_api` config, which is built around vLLM/SGLang by default.
  - `modal_glm_ocr_transformers.py` — **the one to deploy.** No serving engine at all: a hand-rolled
    OpenAI-shaped `/v1/chat/completions` backed by plain `transformers` `.generate()`, run as a
    background thread in the same process the model is loaded in (not a subprocess — that would put
    the weights outside what `enable_memory_snapshot` can capture). vLLM/SGLang's engine-startup
    cost (CUDA graph capture, JIT kernels, multiprocess spawn) only pays for itself under sustained
    load, which a cold-start-per-request container never has; this model's own weights load in
    under a second regardless of engine. Snapshotting is wired up (`@modal.enter(snap=True)` loads
    CPU-only, `snap=False` moves to CUDA and starts the inner server) but empirically **did not
    reduce cold start** in testing — Modal recreated the snapshot on both retries instead of
    reusing one, so there's a real speed win still on the table if that gets root-caused.
  - `modal_glm_ocr_vllm.py` — works, confirmed on both plain text and a merged-cell table. Two
    non-obvious fixes were required: no `--speculative-config` (vLLM's generic `"mtp"` method
    expects `model.layers.N.mtp_block.*` weight keys this checkpoint doesn't ship — its speculative
    weights are laid out for SGLang's NEXTN naming instead), and the image must install vLLM fresh
    from PyPI rather than start from `lmsysorg/sglang`/vLLM's own docker images.
  - `modal_glm_ocr_sglang.py` — **left broken, do not deploy as-is.** Every `lmsysorg/sglang`
    docker image tag tried (default and `-cu130-runtime`) ships an `sgl_kernel` binary with only an
    sm100 (Hopper/Blackwell) build of `common_ops` — no sm89 (Ada/L4) — so it ABI-fails regardless
    of torch/CUDA version matching. Building from a plain CUDA `-devel-` base (needs `nvcc`, not
    just `build-essential`) plus a fresh `pip install sglang` got further (past model load, into
    GPU memory-pool setup) but the last attempt still didn't reach a clean run before the vLLM path
    proved out instead — kept in the repo as a starting point, not a working deployment.
  - Dead end, not kept: an **Ollama** backend (`ocr_api.api_mode: ollama_generate`, the mode
    glmocr's own docs recommend as the 502-error workaround) silently returned empty text for every
    region regardless of context size or worker count; switching to Ollama's own
    `/v1/chat/completions` generated real tokens but most regions then rambled toward the
    8192-token cap instead of stopping. Root cause never isolated — abandoned in favor of the
    transformers/vLLM paths above, which both work.
