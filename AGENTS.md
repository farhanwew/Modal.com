# AGENTS.md

Each top-level directory here is a **self-contained, independent Modal app**. There is no shared build system, no root package manifest, and **no linter, formatter, or test runner** anywhere in the repo. Dependencies are declared per-script inside `modal.Image` pip/apt installs. Don't look for shared config — read the app's own script.

## Modal workflow

```bash
modal setup                     # one-time local auth
modal run <script.py> [args]    # smoke test via @app.local_entrypoint()
modal serve <script.py>         # dev URL, live-reload
modal deploy <script.py>        # persistent deploy
modal app logs <app-name>       # app-name = modal.App("...") STRING, NOT folder/file name!
```

- Several `App("...")` names differ from their folder (e.g. `Hermes/deploy.py` wraps an app for `modal deploy -m Hermes.deploy`). Use the string inside `modal.App(...)` for logs/dashboard, never the folder name.
- Relative imports are used (e.g. `Hermes/hijack.py` imported as `from .hijack import ...`), so run those as packages: `modal deploy -m Hermes.app`, not a bare path.

## Every app follows the same shape

1. `modal.Image` — `Image.from_registry("nvidia/cuda:...")` for GPU/compiled deps, `debian_slim()` for pure-Python/transformers. Weights are either **baked into the image** via `run_function` (fastest cold start; rebuild required for new weights) or **downloaded to a `modal.Volume`** via a prewarm function (updatable without image rebuild).
2. `app = modal.App("name")`
3. `@app.cls(...)` with `@modal.enter()` doing one-time model load into `self`, `@modal.method()` for inference.
4. Thin `@modal.fastapi_endpoint` / `@modal.asgi_app()` wrapper parsing a raw `dict`, calling `.remote()`, returning JSON.
5. `@app.local_entrypoint()` `main()` for smoke testing.

GPU type is overridable via `MODAL_GPU` env (default varies per app, `L4` or `T4`).

## Gotchas

- Several folders keep numbered/suffixed variants side by side (`app.py`, `app2.py`, `app3.py`, `app_8.py`, `app_repeng.py`). **Check which variant is actually in use** (git history, or ask) before editing "the app"; later-numbered ones are often newer iterations, not legacy.
- Comments and user-facing validation/error messages are frequently written in **Indonesian (Bahasa Indonesia)** — preserve that language when editing existing strings/comments.
- `unlimited-ocr/` shows Modal **memory snapshots**: `enable_memory_snapshot=True` with a split `@modal.enter(snap=True)` (CPU load, snapshotted/reused) / `@modal.enter(snap=False)` (`.cuda()` per cold start). The `snap=True` phase **must never initialize CUDA** or the snapshot can't restore.
- `ppocr-v6/` — despite the name it needs **no PaddlePaddle** (`engine="transformers"`). Its `/table` route is **known-broken upstream** (returns `KMeans n_clusters ... Got 0`), don't chase it; use `ocr-suite/`'s PaddleOCR-VL `table` task.
- Newer endpoints take a raw `dict` and validate fields manually; older ones (Hermes) use Pydantic.

## Standalone UIs (not Modal apps)

- `gradio_app.py` in several folders are thin local clients (PEP 723 inline deps) run with `uv run gradio_app.py`, driven by a base-URL env var (`UNLIMITED_OCR_BASE_URL`, `PADDLEOCR_VL_BASE_URL`, `PPOCR_V6_BASE_URL`). Ports: `GRADIO_SERVER_PORT=7861` for ppocr-v6 to sit alongside paddleocr-vl on 7860.

- `ocr-suite/` contains independent Modal OCR apps plus one local Gradio client. Active scripts include PaddleOCR-VL transformers/GGUF/official, Unlimited-OCR, GLM-OCR transformers/vLLM/GGUF, MinerU, and Docling. `glm-ocr/modal_glm_ocr_sglang.py` is retained but its deployment is stopped.
- Deploy or smoke-test the selected script from its directory; each app has its own `@app.local_entrypoint()`. Use the exact script paths in `ocr-suite/README.md`.
- The shared `gradio_app.py` is a local client, not a Modal app. Run it with `uv run gradio_app.py`; it reads the adjacent `.env`. OpenDataLoader is the local, no-upload path and requires Java 11+.
- `Dockerfile` builds only the Gradio client; model weights and GPU runtimes remain remote on Modal. Keep its dependency list synchronized with the PEP 723 block in `gradio_app.py`, and pass backend URLs at runtime with `--env-file .env`.
- The transformers and Unlimited-OCR memory-snapshot `snap=True` phases must load on CPU and must not initialize CUDA. The GGUF GPU snapshot instead requires `--no-mmap`; if a restore is stale or broken, redeploy to rebuild it.
- The OCR client converts PaddleOCR `table` OTSL to HTML; do not treat raw OTSL as Markdown. The GGUF and transformers apps share the DocLayout-YOLO page pipeline, whose reading order is heuristic and whose merged-cell limitations are documented in the app READMEs.

## App-specific notes (details in each script's comments)

- **Hermes/** — Hermes-2-Pro via transformers + bitsandbytes; `hijack.py` monkeypatches `transformers` GenerationMixin to add sampler options (`min_p`, `tfs`, `top_a`, mirostat, etc.). Accepts control vectors as `.npz` / `.npy` / base64 `.npy`.
- **generate controlvec/** — trains repeng control vectors from a persona description at request time; output feeds Hermes' `control_vector(_npy_b64)` fields.
- **emding model**: sentence-transformers embeddings, model cached in a Volume.
- **qwen35-ws3**: Qwen VLM + auto-detected PEFT adapter, Indonesian fact-check classifier.
- **whisper-transcriber**: Whisper turbo transcription; ffmpeg in image so video URLs work.
- **`ocr-suite/` PaddleOCR-VL apps**: same model (PaddleOCR-VL-1.6), transformers vs llama-server. GGUF path needs `--mmproj`; the Hermes-GGUF pattern is text-only.
- **ppocr-v6 vs paddleocr-vl**: ppocr-v6 = text lines only, fast/small; paddleocr-vl = table structure/LaTeX/char-scan.
