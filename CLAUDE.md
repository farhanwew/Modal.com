# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A collection of independent scripts for deploying LLMs and other models (text generation, embeddings, VLM classification, speech transcription) on the **Modal** cloud platform (https://modal.com). Each top-level directory is its own self-contained Modal app — there is no shared build system, no root-level package manifest, and no test suite. Dependencies are declared per-script via `modal.Image` pip/apt installs.

## Common commands

```bash
pip install modal
modal setup                              # one-time local auth

modal run <script.py> [--arg value ...]  # run the @app.local_entrypoint() for a quick smoke test
modal serve <script.py>                  # serve with a temporary dev URL, live-reloads on save
modal deploy <script.py>                 # deploy a persistent app + endpoint
modal app logs <app-name>                # stream logs (app-name comes from modal.App("...") in the script, not the folder name)
```

Scripts must be run from a location where their relative imports resolve (e.g. `Hermes/app.py` does `from .hijack import ...`, so it needs `modal deploy -m Hermes.app` / to be invoked as a package, per `Hermes/deploy.py`).

There is no linter, formatter, or test runner configured anywhere in this repo.

## Architecture

Every app in this repo follows the same shape:

1. Build a `modal.Image` (either `Image.from_registry("nvidia/cuda:...")` for GPU/compiled deps like `llama-cpp-python`/`flash-attn`, or `modal.Image.debian_slim()` for pure-Python/transformers stacks), with a `run_function`/`prewarm_models` build step that bakes model weights into the image or downloads them into a `modal.Volume` — so warm/cold starts don't redownload weights.
2. `app = modal.App("some-name")` — this name is what shows up in `modal app logs` and the Modal dashboard, and frequently differs from the folder or file name.
3. An `@app.cls(gpu=..., image=..., volumes=...)` class with `@modal.enter()` doing one-time model load into `self`, and `@modal.method()` holding the actual inference call.
4. A thin `@app.function(...)` wrapped with `@modal.fastapi_endpoint(method="POST")` (or `@modal.asgi_app()` when a script needs multiple routes) that parses a raw `dict` payload, instantiates the `@app.cls`, calls `.remote()`, and returns JSON.
5. An `@app.local_entrypoint()` `main()` used by `modal run` for smoke testing without deploying.

### Directories

- **Hermes/** — Hermes-2-Pro-Mistral-7B via `transformers` + `bitsandbytes`, with `repeng` control-vector steering. `app.py` bakes the model into the image at build time; `app_modal.py` is an earlier variant that instead downloads to a `modal.Volume` via a standalone `prewarm_models` function. `deploy.py` wraps `app_modal`'s app for `modal deploy -m Hermes.deploy`. `hijack.py` monkeypatches `transformers.GenerationMixin`/`GenerationConfig` to add sampler options (`min_p`, `tfs`, `top_a`, mirostat, dynamic temperature, quadratic sampling) that stock `transformers.generate()` doesn't expose. Control vectors can arrive as `.npz` (native repeng format), `.npy` (custom `{"directions": {...}}` dict), or inline base64-encoded `.npy` (`control_vector_npy_b64`).
- **Hermas GGUF/** — same Hermes-2-Pro model served via `llama-cpp-python` against a quantized GGUF file instead of `transformers`; much lighter image and GPU (T4). `app.py`/`app_8.py`/`app_repeng.py` are separate quantization/feature variants kept side by side rather than replacing each other.
- **generate controlvec/** — the `cvec-generator` app that *trains* new repeng `ControlVector`s from a persona description at request time: it generates positive/negative example sentences and an antonym using the LLM itself, builds a `repeng.DatasetEntry` dataset, trains the vector, and returns it as base64 `.npy`. Its output is meant to be fed into Hermes/Hermas GGUF's `control_vector` / `control_vector_npy_b64` request fields. `app.py`/`app2.py`/`app3.py` are successive iterations.
- **emding model/** — `sentence-transformers` (`all-mpnet-base-v2`) embedding endpoint, model cached in a `modal.Volume`.
- **qwen35-ws3/** — vision-language classifier (Qwen VLM base model + optional PEFT/LoRA adapter, auto-detected via `adapter_config.json`) for social-media fact-check ("DFK") content moderation. Accepts `title`/`context`/image (`image_url` or `image_base64`) and returns Indonesian-language `Label:` / `Analisis:` structured text.
- **whisper-transcriber/** — OpenAI Whisper `turbo` transcription/translation endpoint; audio supplied as a URL or base64, `ffmpeg` installed in the image so video URLs work too (audio track is extracted automatically).
- **ocr-suite/** — independent document-OCR deployments behind one shared Gradio client (`gradio_app.py`). Current scripts, deployment commands, backend URLs, and stopped apps are documented in `ocr-suite/README.md`.
  - `modal_paddleocr_vl_gguf.py` (app `paddleocr-vl-gguf`) — `PaddleOCR-VL-1.6-GGUF` via `llama-server`, the **default**. Same weights as the transformers app, ~7-9× faster (**145-247 vs 22-28 tok/s**). GGUF is *not* smaller here (935 MB + 881 MB mmproj ≈ the 1.9 GB safetensors) — speed only. Multimodal GGUF needs `llama-server --mmproj`, so `Hermas GGUF/`'s `llama-cpp-python` pattern does **not** transfer (text-only). Base image `ghcr.io/ggml-org/llama.cpp:server-cuda` with `.entrypoint([])`.
  - `modal_paddleocr_vl.py` (app `paddleocr-vl`) — the transformers path; kept for accuracy comparison. Three upstream gotchas are worked around in it, all found by running rather than reading: `processor.image_processor.min_pixels` doesn't exist; on transformers v5 `images_kwargs` must be nested under `processor_kwargs` or it is silently ignored; and the model ships `"use_cache": false` in both configs, which makes decoding O(n²) — passing `use_cache=True` took throughput from ~2.8 to 29-49 tok/s.
  - `unlimited-ocr/modal_unlimited_ocr.py` (app `unlimited-ocr`) — `baidu/Unlimited-OCR`. `infer()` doesn't yield text, so a background thread + `sys.stdout` redirect captures its token-by-token `print()`s; final clean text is read from the `result.md` it writes. Deliberately per-page `infer()`, not `infer_multi()` — the official Space never calls the latter either. This is why `@modal.concurrent` is unsafe there but fine on the PaddleOCR-VL apps.
  - The GGUF app uses `llama-server` with `--mmproj`; DocLayout-YOLO loads in a background thread and `/parse-page` waits for it before inference. Keep detailed startup and benchmark notes in the backend README.
  - **Weights belong in a Volume, not baked into the image.** Baking was tried and fails: `modal deploy` runs two builders concurrently against the same image id, and the one that saves first can win with an empty `/models` — the deployed image had no such directory and llama-server crash-looped. A Volume is immune because it lives outside the image.
  - Still unmeasured: whether GGUF's speed costs accuracy on `table`/`formula`. Every comparison so far has been speed. The UI's **Compare** tab exists for exactly that.
- **ppocr-v6/** — classic detect-then-recognise OCR (`PP-OCRv6_{tier}_det` + `_rec`, tier via `PPOCR_TIER`, default `medium`). Deliberately a *separate app* from `ocr-suite/`, not a replacement: it is 133 MB vs 1.9 GB and does the whole demo page in ~3-5 s vs ~43 s, because it is CNN+CTC rather than autoregressive — but it emits **text lines only**, with no table structure, LaTeX or chart/seal parsing. Plain text and speed → here; structure → `ocr-suite/`. Despite the name it needs **no PaddlePaddle**: `paddleocr` is pure-Python orchestration and the `_safetensors` weights run via `engine="transformers"` on torch (verified — logs print `paddle_present=False`). Reading order uses gutter detection (uncovered vertical bands = column boundaries, with >50%-width headlines excluded from the coverage map); clustering left edges and a left/right split were both tried first and failed on a 4-column newspaper. Accepts images or PDFs. A `/table` route exists but is **known-broken upstream**: PaddleOCR's `TableRecognitionPipelineV2` builds fine on the transformers backend (all its models do have safetensors builds) yet `predict()` raises `KMeans n_clusters ... Got 0` because cell detection returns empty — reproduced on PaddleOCR's own demo image with both SLANeXt and SLANet_plus. It returns that as JSON rather than a 500; use ocr-suite's `table` task instead. Two settings were needed just to get that far and are kept for when it's fixed: `use_layout_detection=False` (PP-DocLayout-L has no safetensors build) and pinning det/rec to PP-OCRv6 (the default PP-OCRv4_server_det has none either). `gradio_app.py` follows the same standalone-PEP-723 pattern as the other UIs (env var `PPOCR_V6_BASE_URL`) and additionally draws detection polygons over page 1, colour-coded by confidence; run it with `GRADIO_SERVER_PORT=7861` to sit alongside ocr-suite's UI on 7870. No memory snapshot yet — `load_s≈15` per cold start.

### Conventions worth knowing

- Newer endpoints (qwen35-ws3, whisper-transcriber) take a raw `dict` payload and validate fields manually; older ones (Hermes) use a Pydantic `GenerateRequest` model with a `_clamp_and_validate` helper that clamps out-of-range sampling params instead of rejecting them outright.
- Model persistence is either "baked into the image" (`run_function` at build time — fastest cold start, but requires a full image rebuild to pick up new weights) or "Volume + prewarm function" (updatable without rebuilding the image, but the container must find the volume already populated or it will raise/fallback to a Hub download).
  - For cold-start-sensitive apps, `ocr-suite/unlimited-ocr/modal_unlimited_ocr.py` demonstrates Modal **memory snapshots**: `enable_memory_snapshot=True` plus a split `@modal.enter(snap=True)` (load weights on CPU, snapshotted once and reused) / `@modal.enter(snap=False)` (`.cuda()` only, reruns per cold start), with `with image.imports():` hoisting `torch`/`transformers` into the snapshot. The hard constraint is that the `snap=True` phase must never initialize CUDA, or the snapshot can't be restored.
- GPU type is commonly overridable via a `MODAL_GPU` env var, defaulting to `L4` or `T4` depending on model size.
- Comments and user-facing validation/error messages are frequently written in Indonesian (Bahasa Indonesia); preserve that language when editing existing strings/comments unless told otherwise.
- Several folders keep numbered/suffixed variants of the same script (`app.py`, `app2.py`, `app3.py`, `app_8.py`, `app_repeng.py`) as successive iterations rather than deleting the old one — if asked to change "the app" in one of these folders, check which variant is actually in use (e.g. via git history or by asking) before assuming it's the lowest-numbered file.
