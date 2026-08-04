# Modal deployment for PP-OCRv6

Classic detect-then-recognise OCR (`PP-OCRv6_*_det` + `PP-OCRv6_*_rec`) on Modal — a much smaller,
much faster alternative to `paddleocr-vl/` for **plain text**, built as a separate app so that
deployment stays untouched.

- `modal.App("ppocr-v6")`, one `@modal.asgi_app()` base URL with `/ocr` and `/health`
- Weights cached in the `ppocr-v6-cache` Volume

## Why this exists alongside `paddleocr-vl/`

`paddleocr-vl/` runs a 958M-parameter VLM that decodes text autoregressively, one token at a time.
PP-OCRv6 is a CNN detector plus a CTC-style recogniser: no token-by-token decoding at all. On the
same demo page:

| | Time | Coverage |
| --- | --- | --- |
| `paddleocr-vl` `/parse-page` | ~43 s (detect 4.5 s + recognise 38.1 s) | 31 blocks |
| `paddleocr-vl` `/recognize` | 15.2 s | 512 tokens (truncated) |
| **`ppocr-v6` `/ocr` (medium)** | **3.0–5.0 s** | **145 lines, whole page** |

Model sizes, from the [official pipeline docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html):

| Tier | det | rec | Total | det Hmean / rec acc |
| --- | --- | --- | --- | --- |
| `medium` (default) | 59.4 MB | 73.3 MB | 133 MB | 86.2% / 83.2% |
| `small` | 9.6 MB | 20.4 MB | 30 MB | 84.1% / 81.3% |
| `tiny` | 1.9 MB | 4.4 MB | 6.3 MB | 80.6% / 73.5% |

…against 1.9 GB for PaddleOCR-VL. Override with the `PPOCR_TIER` env var.

## What it cannot do

This is the deciding tradeoff, and the official docs are explicit — the general OCR pipeline

> outputs text lines with bounding boxes only … does **not** produce table structure or formula
> recognition. Those require separate specialized pipelines.

So there is **no OTSL/HTML table structure, no LaTeX formulas, no chart or seal parsing**. Those are
exactly what `paddleocr-vl/` is for. If a benchmark table in the model cards shows a "Table" column,
that is *text-detection Hmean on images containing tables* — whether the text lines were located, not
whether the table was reconstructed.

Rule of thumb: **plain text and speed → this app. Structure → `paddleocr-vl/`.**

### `/table` — tried, and currently blocked upstream

PaddleOCR does ship a separate `TableRecognitionPipelineV2` that emits real HTML tables, and all the
models it needs turn out to have safetensors builds (`PP-LCNet_x1_0_table_cls` 1.7M,
`RT-DETR-L_wired_table_cell_det` 32.9M, `SLANeXt_wired`/`SLANet_plus` 90.9M/1.8M) — contrary to what
the pipeline docs imply. It **loads** cleanly on the transformers backend (`[INIT] table pipeline
load_s=6.7`, no PaddlePaddle), but it does not **run**:

```
[transformers] Resampling is not supported in SLANeXt
InvalidParameterError: The 'n_clusters' parameter of KMeans must be an int in the
range [1, inf). Got 0 instead.
```

Cell detection returns nothing, so the clustering step gets zero clusters. Reproduced on PaddleOCR's
own `table_recognition.jpg` demo image with both SLANeXt and SLANet_plus.

Getting that far needed two non-obvious settings, kept in the code in case the backend is fixed later:

- `use_layout_detection=False` — otherwise the pipeline builds `PP-DocLayout-L`, which has no
  safetensors build and rejects the engine outright.
- Pinning `text_detection_model_name` / `text_recognition_model_name` to PP-OCRv6 — the default
  `PP-OCRv4_server_det` has no safetensors build either.

`/table` returns this as a JSON `error` + `hint` rather than a 500. Even if it were fixed, the accuracy
ceiling is 63.69% (SLANet_plus) to 69.65% (SLANeXt) — well below the VLM. **Use `paddleocr-vl`'s
`table` task for tables.**

## No PaddlePaddle

Despite the name, this needs no `paddlepaddle-gpu`. The `paddleocr` package is pure-Python
orchestration, and the `_safetensors` model variants run through `engine="transformers"` on torch.
Verified rather than assumed — locally, installing `paddleocr` resolved 61 packages with
`paddle installed? False`, and the container logs confirm it at runtime:

```
[INIT] tier=medium device=gpu:0 load_s=15.1 paddle_present=False cuda=True
```

That matters: PaddlePaddle would have added a second framework to a torch-only image and its CUDA
initialisation conflicts with the CPU-only load that memory snapshots require.

## Use

```bash
modal deploy modal_app.py
modal run modal_app.py --image-url "https://…"        # smoke test

curl -X POST "$BASE_URL/ocr" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://…"}'
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_url` / `image_base64` | `str` | — | Exactly one |
| `min_score` | `float` | 0.5 | Detection emits a tail of low-confidence single characters (observed `"1"` at 0.57, `"a"` at 0.14); this drops them |
| `reading_order` | `bool` | `true` | Sort lines into reading order (see below) |

Returns `{"tier", "lines":[{"text","score","poly"}], "text", "lines_kept", "lines_detected", "elapsed_s"}`.

## Reading order

Detection returns lines in its own order, which interleaves columns. Two approaches were tried
against the demo page (a multi-column Chinese newspaper):

1. **Left-half / right-half split** — failed. A newspaper page holds four narrow columns, so "left
   half" still contains two of them and lines alternate.
2. **Clustering left edges** — also failed. Within a column the left edges spread out enough that no
   single gap stands out, so no boundary is found and everything collapses into one column.
3. **Gutter detection** (current) — works. Mark which x positions any text box covers, then treat
   uncovered vertical bands wider than 1.5% of the page as column boundaries. Lines spanning more
   than half the page are excluded from the coverage map, since one headline would otherwise bridge
   every gutter.

Still a heuristic: headlines are filed into whichever column their centre falls in, and text wrapped
around a figure will come out wrong.

PDF input is rasterised server-side; a 2-page test PDF returned 4 lines in **1.8 s** (the same PDF
through `paddleocr-vl/parse-page` took ~69 s).

## Gradio UI

`gradio_app.py` is a standalone client — no model loading, just HTTP to the base URL. PEP 723 script,
so `uv run` handles dependencies:

```bash
# bash/zsh
PPOCR_V6_BASE_URL="https://<workspace>--ppocr-v6-web-app.modal.run" uv run gradio_app.py
```

```powershell
# PowerShell
$env:PPOCR_V6_BASE_URL = "https://<workspace>--ppocr-v6-web-app.modal.run"
uv run gradio_app.py
```

Set `GRADIO_SERVER_PORT=7861` to run it alongside `paddleocr-vl/gradio_app.py` (which defaults to
7860) and compare the two side by side.

Upload an image or PDF; controls for **min confidence** and **reading order**. Two output tabs: the
extracted text, and the detected boxes drawn over page 1 — green above 0.9 confidence, amber below,
which makes a badly chosen `min_score` visible at a glance. Boxes are drawn for page 1 only; stacking
several pages' coordinates onto one bitmap would be meaningless.

## Not done yet

- **No memory snapshot.** `[INIT] … load_s=15.1` is paid on every cold start. Whether PaddleOCR's
  pipeline can be constructed without touching CUDA — the hard requirement for `@modal.enter(snap=True)`
  — is unverified, and a snapshot that cannot restore fails at runtime rather than at deploy.
- **No hybrid with the VLM.** The obvious next step is routing DocLayout-YOLO's `plain text`/`title`
  blocks here and only `table`/`formula` blocks to `paddleocr-vl/`, which would cut most of that
  38 s recognise time while keeping structure where it matters.
