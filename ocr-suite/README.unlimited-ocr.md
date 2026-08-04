# Modal deployment for `baidu/Unlimited-OCR`

Deploys [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (a ~3.3B-param DeepSeek-OCR-style
vision-language model, `trust_remote_code=True`) on Modal as a **streaming** POST endpoint. Supports single-image
OCR/document parsing, multi-image (multi-page) parsing, and PDF input (rasterized page-by-page with PyMuPDF).
Multi-page/PDF input is parsed **one page at a time** via `infer()`, the same way the official
[`baidu/Unlimited-OCR` Space](https://huggingface.co/spaces/baidu/Unlimited-OCR/tree/main) does it — the model repo
also ships an `infer_multi()` method, but the official demo never calls it, so this deployment sticks to the
per-page code path that's actually been exercised/tuned by the model authors.

- `modal.App("unlimited-ocr")`
- one `modal.Cls` (`UnlimitedOCRServer`) that loads the model once per warm GPU container
- HF weights/tokenizer cached in a Modal Volume (`unlimited-ocr-cache`) so cold starts don't re-download ~6.7 GB of
  safetensors every time
- **one deployed base URL** (`web_app`, an `@modal.asgi_app()` wrapping a small FastAPI app) with routes `/parse`
  (GPU, streaming), `/explode-pdf` (CPU-only PDF page splitter), and `/health`. The web app container itself stays
  CPU-only/cheap — `/parse` just dispatches to the GPU `UnlimitedOCRServer` class via `.remote_gen()`.
- `gradio_app.py` — a standalone Gradio UI that talks to that one base URL (see below)

## Why streaming, and why `save_results=True` instead of `eval_mode=True`

The model repo's own `infer()`/`infer_multi()` (in `modeling_unlimitedocr.py`, loaded via `trust_remote_code=True`)
don't return text incrementally — they stream generated tokens by `print()`-ing them from inside `model.generate()`
(via an internal `TPSTextStreamer`), and only expose the *final* text two ways:

- `eval_mode=True` returns the **raw** decoded string, including `<|ref|>...<|/ref|>` / `<|det|>...<|/det|>` markup
  tags — not what you want for readable OCR output.
- `save_results=True` writes the **cleaned** markdown (tags stripped, cropped image regions turned into
  `![](images/N.jpg)` links) to `{output_path}/result.md`, but `infer()` doesn't return anything at all in that
  mode — you have to read the file.

This app calls `infer(..., save_results=True)` once per page (see above), reads `result.md` after each call as the
source of truth, and — since the alternative is waiting silently until the whole (possibly 8k+ token) generation
finishes — redirects the model's internal `print()` calls into a queue on a background thread so partial output can
be streamed back to the HTTP client as newline-delimited JSON while generation is still running.

## Setup

```bash
pip install modal
modal setup
```

The model repo is public, so no Hugging Face secret is required. If it ever becomes gated, add
`secrets=[modal.Secret.from_name("huggingface-secret")]` to both the `@app.cls(...)` and `@app.function(...)`
decorators in `modal_app.py`, and create the secret with:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

## Test once

```bash
modal run modal_app.py --image-url "https://example.com/document.jpg"
```

This streams partial text to your terminal as it's generated, then prints the final cleaned result.

## Serve during development

```bash
modal serve modal_app.py
```

Modal prints a temporary base URL for `web_app`, e.g. `https://<workspace>--unlimited-ocr-web-app-dev.modal.run`
(after `modal deploy` it's the same shape without `-dev`). `/parse` streams newline-delimited JSON — one
`{"text": ..., "done": bool, "pages": N}` object per line, `done: false` while generating and a final `done: true`
line with the cleaned result (pages are joined with `── PAGE i / N ──` separators for multi-page input):

```bash
curl -N -X POST "$BASE_URL/parse" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/document.jpg", "mode": "gundam"}'
```

`/explode-pdf` is a plain (non-streaming, CPU-only) route that rasterizes a PDF into per-page PNGs without touching
the GPU — useful if you'd rather drive OCR one page at a time (progress bars, cancel a page, retry a page) instead
of sending the whole PDF to `/parse` in one call:

```bash
curl -X POST "$BASE_URL/explode-pdf" \
  -H "Content-Type: application/json" \
  -d '{"pdf_url": "https://example.com/document.pdf", "pdf_dpi": 200}'
# => {"pages": ["<base64 png>", "<base64 png>", ...]}
```

### `/parse` request fields

Provide exactly one input source:

| Field                          | Type       | Description                                             |
| ------------------------------- | ---------- | -------------------------------------------------------- |
| `image_url` / `image_base64`   | `str`      | Single-page image                                        |
| `image_urls` / `images_base64` | `list[str]`| Multiple pre-rendered page images, parsed one page at a time |
| `pdf_url` / `pdf_base64`       | `str`      | PDF; pages are rasterized with PyMuPDF (`pdf_dpi`, default 200), then parsed one page at a time |

Optional generation fields:

| Field                    | Type    | Default                              | Description                                                    |
| ------------------------ | ------- | ------------------------------------ | ---------------------------------------------------------------- |
| `prompt`                 | `str`   | `document parsing.`                  | `<image>` is prepended automatically; used as-is for every page |
| `mode`                   | `str`   | `gundam`                              | `gundam` (base_size=1024, image_size=640, crop_mode=True) or `base` (base_size=1024, image_size=1024, crop_mode=False) |
| `max_length`             | `int`   | 8192                                  | Total generation sequence length budget (model card default is 32768; 8192 is a faster, still generous default — raise it for very dense documents) |
| `temperature`            | `float` | 0.0                                   | 0.0 = greedy decoding                                           |
| `no_repeat_ngram_size`   | `int`   | 35                                    | Sliding-window n-gram repetition block size (per model card)    |
| `ngram_window`           | `int`   | 128                                   | Sliding window size for the n-gram repetition blocker           |

## Deploy

```bash
modal deploy modal_app.py
```

After deployment, stream logs with:

```bash
modal app logs unlimited-ocr
```

## Gradio UI

`gradio_app.py` is a standalone client — it doesn't load the model, it just calls the deployed base URL over HTTP.
It's a [PEP 723](https://peps.python.org/pep-0723/) inline-metadata script, so `uv run` installs its two
dependencies (`gradio`, `requests`) into an ephemeral environment automatically — no `pip install`/venv needed:

```bash
# bash/zsh
UNLIMITED_OCR_BASE_URL="https://<workspace>--unlimited-ocr-web-app.modal.run" uv run gradio_app.py
```

```powershell
# PowerShell
$env:UNLIMITED_OCR_BASE_URL = "https://<workspace>--unlimited-ocr-web-app.modal.run"
uv run gradio_app.py
```

`UNLIMITED_OCR_BASE_URL` defaults to `http://localhost:8000` if unset (useful if you're proxying/tunneling
`modal serve` locally).

### Running the UI in Docker

`Dockerfile` containerises **the Gradio client only** — no GPU, no model weights. The model stays on Modal, whose
image is defined in code via `modal.Image` in `modal_app.py`; there is deliberately no Dockerfile for that side.

```bash
docker build -t unlimited-ocr-ui .
docker run --rm -p 7860:7860 \
  -e UNLIMITED_OCR_BASE_URL="https://<workspace>--unlimited-ocr-web-app.modal.run" \
  unlimited-ocr-ui
```

Then open `http://localhost:7860`. `UNLIMITED_OCR_BASE_URL` must be passed at run time — without it the client
falls back to `http://localhost:8000`, which inside the container points at the container itself and fails.

The image sets `GRADIO_SERVER_NAME=0.0.0.0` because Gradio otherwise binds `127.0.0.1`, which is unreachable from
outside the container. Note that the dependency list in the Dockerfile duplicates the PEP 723 block at the top of
`gradio_app.py` (used by `uv run`) — keep the two in sync when changing dependencies.

The layout/flow mirrors the [official Space's `index.html`](https://huggingface.co/spaces/baidu/Unlimited-OCR/blob/main/index.html):
a dark, monospace-styled UI with a **Long**/**Base** mode toggle (→ `gundam`/`base`), an **NGRAM** toggle, a
document preview + prompt bar on the left, and a streaming output box with Start/Stop/Reset on the right.

Dropping a file previews it immediately: an image renders directly, a PDF is rasterized via `/explode-pdf` and
shown as a page gallery (the resulting pages are cached client-side so pressing START doesn't re-explode the same
PDF). Pressing START then streams `/parse` — once for an image, once per page for a PDF (same as the official
demo), joining pages with `── PAGE i / N ──` headers as they complete.

## Cold start

`UnlimitedOCRServer` uses Modal **memory snapshots** (`enable_memory_snapshot=True`), following the pattern in
[`aitf-its-tim3-dfk/deployment-model`](https://github.com/aitf-its-tim3-dfk/deployment-model) (whose README reports
~20s vs ~70s without it). Startup is split so the expensive half is only ever paid once:

| Phase | Runs | Work |
| --- | --- | --- |
| `@modal.enter(snap=True)` (`load_to_cpu`) | once — result is snapshotted and reused | Download (Volume-cached) + load tokenizer and model **on CPU**, in parallel via `ThreadPoolExecutor` |
| `@modal.enter(snap=False)` (`move_to_gpu`) | every cold start | `.cuda()` only; logs `[COLD_START] move_to_gpu elapsed_ms=…` |

Supporting pieces:

- `with image.imports():` hoists `torch` / `transformers` so those imports land inside the snapshot instead of
  re-running on each cold start.
- `HF_XET_HIGH_PERFORMANCE=1` + `huggingface_hub[hf_xet]` for faster weight downloads on the first (snapshot-building) start.

**Constraint:** nothing in the `snap=True` phase may touch CUDA — initializing a CUDA context during the snapshot
phase makes the snapshot unrestorable. That's why the model is loaded plainly (no `device_map`, no `.cuda()`) there
and only moved to the GPU in the `snap=False` phase, mirroring the model card's own load-then-`.cuda()` two-step.

## Notes

- GPU defaults to `L4` (24 GB), overridable via the `MODAL_GPU` env var; the ~3.3B bf16 weights plus vision towers
  comfortably fit.
- `UnlimitedOCRServer` deliberately has no `@modal.concurrent` — the stdout-redirect trick used to stream tokens
  only stays correct if a container runs one generation at a time (concurrent generations would race on `sys.stdout`
  and could interleave/clobber each other's stream). Under load, Modal will scale out more containers rather than
  packing concurrent requests into one; that's intentional here.
