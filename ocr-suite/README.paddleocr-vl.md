# Modal deployment for `PaddlePaddle/PaddleOCR-VL-1.6`

Deploys [`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
(958.6M params, ~1.9 GB — roughly 3.5× lighter than `unlimited-ocr/`'s Unlimited-OCR) on Modal as a
streaming POST endpoint, plus a Gradio client.

- `modal.App("paddleocr-vl")`
- one `modal.Cls` (`PaddleOCRVLServer`) on an `L4`, weights cached in a Modal Volume (`paddleocr-vl-cache`)
- **one deployed base URL** (`web_app`, an `@modal.asgi_app()`) with routes `/recognize` (GPU, streaming)
  and `/health`
- `gradio_app.py` — a standalone UI following the official Space's tab structure

## Two levels of parsing

This deployment uses the model's **`transformers` inference path**, which the model card limits to:

> The example code below only supports element-level recognition and text spotting.

That covers `/recognize`. Page-level parsing is built on top rather than taken from the official
pipeline — see `/parse-page` below for what that means and where it differs.

Six tasks are supported, via the prompts given in the model card:

| `task` | Prompt sent to the model |
| --- | --- |
| `ocr` | `OCR:` |
| `table` | `Table Recognition:` |
| `formula` | `Formula Recognition:` |
| `chart` | `Chart Recognition:` |
| `spotting` | `Spotting:` |
| `seal` | `Seal Recognition:` |

## Setup

```bash
pip install modal
modal setup
```

The model repo is public — no Hugging Face secret required. (Logs will warn about unauthenticated
Hub requests; adding an `HF_TOKEN` secret only affects download rate limits.)

## Deploy and use

```bash
modal deploy modal_app.py
modal app logs paddleocr-vl
```

`/recognize` streams newline-delimited JSON — one `{"text":…, "done":bool, "task":…}` object per line,
`done: false` while generating and a final `done: true` line:

```bash
curl -N -X POST "$BASE_URL/recognize" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png","task":"ocr"}'
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_url` / `image_base64` | `str` | — | Exactly one required |
| `task` | `str` | `ocr` | One of the six above |
| `max_new_tokens` | `int` | 1024 | Dense pages need more; this dominates latency (see below) |
| `max_pixels` | `int` | 1003520 (2048·28·28 for `spotting`) | Image-token budget: the image is cut into 28×28-pixel tokens, so this ÷ 784 is how many image tokens the model attends over. Halving it roughly halves prefill (see below) but resolves small text less reliably. |

Smoke test without deploying: `modal run modal_app.py --image-url "…" --task ocr`.

### `/parse-page` — page-level document parsing

`/recognize` sends a whole page to the VLM as one image and gets flat text back: no block types, no
reliable reading order on multi-column layouts. `/parse-page` adds a layout pass in front of it:

1. **Detect** blocks with [DocLayout-YOLO](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)
   (Apache-2.0, ~40 MB, pure PyTorch — deliberately *not* PaddlePaddle, which would drag a second
   framework into the image and break the CPU-load requirement of memory snapshots), then NMS.
2. **Order** them — full-width blocks act as horizontal separators; between them, narrow blocks read
   left column then right.
3. **Route** each block to the task that fits it: `title`/`plain text`/captions → `ocr`,
   `table` → `table`, `isolate_formula` → `formula`. `abandon` (headers/footers/page numbers) and
   `figure` are dropped.
4. **Assemble** markdown in order — titles become `##`, tables are converted from OTSL to HTML,
   formulas are wrapped in `$$`.

Detection and recognition run **in the same container on purpose**: a page can have 30+ blocks, and a
network hop per block would dominate the cost.

```bash
curl -N -X POST "$BASE_URL/parse-page" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"…"}'          # or {"pdf_base64":"…"} / {"pdf_url":"…"}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_url` / `image_base64` / `pdf_url` / `pdf_base64` | `str` | — | Exactly one. PDFs are rasterised server-side and every page goes through the same path. |
| `pdf_dpi` | `int` | 200 | Rasterisation resolution for PDF input |
| `conf` | `float` | 0.25 | Layout-detection confidence threshold |
| `iou` | `float` | 0.45 | NMS IoU threshold |
| `max_new_tokens` | `int` | 512 | **Per block**, not per page |

Streams NDJSON: `{"blocks":[…], "markdown":…, "total":N, "pages":N, "progress":i, "done":bool}`.
Each block carries `type`, `bbox`, `score`, `page`, `task` and `text`.

Measured on the demo page (1524×1368): 31 blocks, `detect_ms=4542`, recognition 38.1 s. A 2-page
synthetic PDF (4 blocks) took ~69 s wall including cold start.

**Known limits, verified or by construction:**

- Reading order is a heuristic, not an XY-cut. Three-or-more-column layouts, floating side notes and
  text wrapped around figures will come out in the wrong order.
- `<ucel>` (merge-with-above) in table output is rendered as a blank cell, not a real `rowspan`.
- `figure` blocks are dropped entirely — captions are kept, the image itself is not extracted.
- No doc orientation detection or unwarping; the official PaddleOCR pipeline has both, this does not.
- At most `MAX_BLOCKS` (60) blocks per page are recognised.

## Performance

Measured on this deployment (L4, `[DEVICE] cuda_available=True name=NVIDIA L4 model_device=cuda:0
dtype=torch.bfloat16`):

- **Cold start is not the bottleneck.** With memory snapshots enabled, restoring the snapshot and
  moving the model to the GPU logs `[COLD_START] move_to_gpu elapsed_ms=1360` — about 1.4 s.
- **The KV cache had to be re-enabled by hand.** The model ships `"use_cache": false` in *both*
  `config.json` and `generation_config.json`. Left at the default, every generated token re-attends
  over the entire sequence — including ~1k image tokens — so decoding is O(n²):

  | | Throughput | 512-token OCR |
  | --- | --- | --- |
  | Model default (`use_cache: false`) | ~2.8 tok/s | ~180 s |
  | `generate(..., use_cache=True)` | 29–49 tok/s | ~17 s |

  `modal_app.py` passes `use_cache=True` explicitly. Each request logs
  `[GEN] task=… chunks=… elapsed_s=… chunks_per_s=…` so this stays easy to re-check.
- **Attention** is pinned to `sdpa` (`[DEVICE] … attn=sdpa`). **FlashAttention-2 was tried and
  rejected**, and the reason is structural rather than a packaging problem: transformers validates FA2
  at load time and refuses on CPU (`FlashAttention2 is not available on CPU`), but the memory-snapshot
  phase *requires* a CPU load. Calling `set_attn_implementation("flash_attention_2")` after
  `.to("cuda")` hits the same check. Enabling it would mean giving up memory snapshots and paying a
  full model load (~60 s) on every cold start instead of ~1.4 s — to chase a gain smaller than the
  run-to-run variance already observed on sdpa (26 / 33.8 / 36.5 tok/s across three identical
  requests), because decode on a 0.9B model is memory-bandwidth bound, not attention bound.
- **`max_pixels` is the other real lever.** Same request, same 512 output tokens, only the image-token
  budget changed:

  | `max_pixels` | image tokens | prompt tokens | generation | throughput |
  | --- | --- | --- | --- | --- |
  | 1003520 (default) | 1280 | 1234 | 15.2 s | 33.8 tok/s |
  | 501760 | 640 | 611 | 9.6 s | 53.2 tok/s |

  Speed was measured, **output quality at the lower budget was not** — halve it only after checking
  the text is still correct for your documents.
- After those, latency scales with `max_new_tokens` — lower it when you don't need a full page.

Not attempted, roughly in order of expected payoff: FlashAttention-2 (needs a matching prebuilt wheel);
a higher-bandwidth GPU (decode is largely memory-bound and L4 is the entry tier); serving via vLLM,
which the model card links a recipe for — the biggest win under concurrent load, but a rewrite that
would likely forfeit the memory-snapshot cold start. `torch.compile` is a poor fit here: its warmup
cost lands on exactly the cold-start path the snapshot exists to avoid.

Cold start uses the same pattern as `unlimited-ocr/`: `enable_memory_snapshot=True`, with
`@modal.enter(snap=True)` loading the processor and model **on CPU** in parallel (`ThreadPoolExecutor`)
and `@modal.enter(snap=False)` doing only `.to("cuda")`. Nothing in the snap phase may touch CUDA or
the snapshot cannot be restored. `with image.imports():` hoists `torch`/`transformers` into the snapshot.

Unlike `unlimited-ocr/`, this app streams with `transformers.TextIteratorStreamer` rather than by
redirecting `sys.stdout` — PaddleOCR-VL uses stock `model.generate()`, so no such hack is needed. That
also makes `@modal.concurrent(max_inputs=2)` safe here, whereas `unlimited-ocr/` must avoid it.

## Two corrections to the model card's example

Both were found by running it, not by reading it:

1. **`processor.image_processor.min_pixels` does not exist** — it raises
   `AttributeError: 'PaddleOCRVLImageProcessor' object has no attribute 'min_pixels'`. The value is
   read from the model's own `preprocessor_config.json` instead (`112896`), via `getattr(...)` with
   that as fallback.
2. **`images_kwargs=` must be nested under `processor_kwargs=`** on transformers v5. Passed at the top
   level, as the model card does, transformers logs
   *"Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict"* and **silently
   ignores it** — meaning the pixel budget, including spotting's larger one, never takes effect.

## Output formats per task

Not every task returns Markdown. Verified against the live deployment:

- `ocr`, `formula`, `chart`, `seal` — plain text / Markdown / LaTeX; render directly.
- `table` — **OTSL markup, not HTML or Markdown**, e.g.
  `<fcel>CRuncover<lcel><lcel><lcel><nl><fcel>Dres<fcel>...<nl>` where `<fcel>` is a filled cell,
  `<ecel>` empty, `<lcel>` merge-with-left (colspan), `<ucel>` merge-with-above, `<nl>` end of row.
  Passing this to a Markdown pane looks broken — browsers silently drop unknown tags like `<fcel>`
  and the cell text runs together. `gradio_app.py`'s `otsl_to_html()` converts it to a real table;
  `<ucel>` is rendered blank rather than as a true rowspan.
- `spotting` — format not yet verified against a real response; the UI shows it raw on purpose
  rather than guessing at a bbox parser.

## Gradio UI

`gradio_app.py` is a standalone client — it doesn't load the model, it calls the deployed base URL over
HTTP. It's a [PEP 723](https://peps.python.org/pep-0723/) script, so `uv run` installs its dependencies
into an ephemeral environment automatically:

```bash
# bash/zsh
PADDLEOCR_VL_BASE_URL="https://<workspace>--paddleocr-vl-web-app.modal.run" uv run gradio_app.py
```

```powershell
# PowerShell
$env:PADDLEOCR_VL_BASE_URL = "https://<workspace>--paddleocr-vl-web-app.modal.run"
uv run gradio_app.py
```

Three tabs, following the official Space: **Document Parsing** (image or PDF → streamed markdown, with
block progress), **Element-level Recognition** (one button per task, with a Markdown pane using the
Space's `latex_delimiters` so formula output renders, plus a raw-output pane) and **Spotting**. A
visible notice states that Document Parsing here is a DocLayout-YOLO pipeline rather than the official
PaddleOCR one, and names the reading-order limitation.

### Running the UI in Docker

`Dockerfile` containerises the Gradio client only — no GPU, no weights. The model stays on Modal, whose
image is defined in code via `modal.Image`; there is deliberately no Dockerfile for that side.

```bash
docker build -t paddleocr-vl-ui .
docker run --rm -p 7860:7860 \
  -e PADDLEOCR_VL_BASE_URL="https://<workspace>--paddleocr-vl-web-app.modal.run" \
  paddleocr-vl-ui
```

`PADDLEOCR_VL_BASE_URL` must be passed at run time — without it the client falls back to
`http://localhost:8000`, which inside the container points at the container itself. The image sets
`GRADIO_SERVER_NAME=0.0.0.0` because Gradio otherwise binds `127.0.0.1`. Note the dependency list in the
Dockerfile duplicates the PEP 723 block in `gradio_app.py`; keep the two in sync.
