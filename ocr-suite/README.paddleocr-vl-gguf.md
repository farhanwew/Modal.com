# PaddleOCR-VL-1.6 as GGUF on llama.cpp

The same model as `paddleocr-vl/`, served through `llama-server` instead of `transformers`, to find
out how much of that deployment's latency was the runtime rather than the model.

Answer: most of it. **~7-9× faster generation.**

## Benchmark

Same L4, same image (PaddleOCR's `paddleocr_vl_demo.png`), same task (`ocr`), same 512-token budget,
both containers warm. In-container generation time, from each app's own `[GEN]` log line:

| Backend | 512 tokens | Throughput |
| --- | --- | --- |
| **GGUF / llama.cpp** | **2.07 – 3.53 s** | **145 – 247 tok/s** |
| transformers (`paddleocr-vl/`) | 18.0 – 23.0 s | 22.3 – 28.5 tok/s |

Cold start is better too: `[INIT] llama-server ready in 4.8s`, against the transformers path needing
memory snapshots to reach ~1.4 s of GPU move on top of a snapshot restore.

Note on that 4.8 s: it was measured before `--no-mmap` was added, which makes the *first* load slower
(read into anonymous memory instead of file-backed mappings) but is what lets the **GPU memory snapshot**
round-trip — see below. What you pay on a cold start is no longer that load at all.

Two caveats worth stating plainly:

- **End-to-end wall clock only differed ~2×** (19-22 s vs 39 s), not 8×. The rest is overhead outside
  the model: fetching the test image from bcebos, base64, and the Modal round trip. The 8× is the
  inference speedup; what you feel depends on how much other work surrounds it.
- **GGUF is not smaller here.** 935 MB model + 881 MB mmproj ≈ 1.8 GB, against 1.9 GB of safetensors.
  This is a speed win, not a size win — the repo ships no aggressively quantised variant.

Output was checked against the transformers deployment on the same page and matches.

## How it runs

The model card's own instruction is `llama-server -m … --mmproj …`, so this is **not** the
`llama-cpp-python` pattern used by `Hermas GGUF/` elsewhere in this repo — that path is text-only and
has no way to pass the vision projector.

- Base image is llama.cpp's maintained CUDA build, `ghcr.io/ggml-org/llama.cpp:server-cuda`, with
  `.entrypoint([])` so Modal controls the process instead of the image's own `llama-server` entrypoint.
  Building llama.cpp from source with CUDA costs 10-20 minutes per rebuild.
- Both `.gguf` files are pulled into a `modal.Volume` at **build** time via `run_function`. The model
  card suggests `llama-server -hf PaddlePaddle/PaddleOCR-VL-1.6-GGUF`, which is neater locally but on
  Modal would re-download on every cold start unless the HF cache also lives on a volume.
- `@modal.enter()` starts `llama-server` on `127.0.0.1:8080` and polls `/health` until it answers —
  it has ~1.8 GB to load first — then requests go to its OpenAI-compatible
  `/v1/chat/completions` with the image as a `data:` URL. `@modal.exit()` terminates it.

### GPU memory snapshot (cold start)

The class uses Modal's **GPU memory snapshot** (`enable_memory_snapshot=True` +
`experimental_options={"enable_gpu_snapshot": True}`), split like this:

- `@modal.enter(snap=True)` — full boot: spawn llama-server, wait for `/health`, send one tiny
  warmup OCR request (so CUDA kernels and the mmproj image path are resident), then load the layout
  detector. Everything lands in the snapshot.
- `@modal.enter(snap=False)` — runs on every cold start, normally just a `/health` check: after a
  restore the subprocess and its sockets come back alive. Respawns (no warmup) only if the restore
  left it dead.

Two consequences:

- `--no-mmap` is **required** for the restore to round-trip — file-backed GGUF mappings do not
  survive, and llama-server would come back with NaN/garbage or exit outright. The slower first load
  is paid once, when the snapshot is created. `--seed 42` keeps generation deterministic across
  restores (already greedy at `--temp 0`).
- The first cold start after a deploy builds the snapshot (~15-20 s, same as before); **every cold
  start after that is a restore — llama-server, torch imports and the layout detector are all
  already resident**, no 1.8 GB volume read and no 5-15 s of import time. Expect seconds, not
  tens of seconds.

GPU snapshots are an **alpha** feature. Symptoms of a broken restore (llama-server exiting on the
first request, CUDA errors, NaN output) mean the snapshot is stale — redeploy to rebuild it, and
check that model/weights paths are identical between boots.

## Use

```bash
modal deploy modal_app.py
modal run modal_app.py --image-url "https://…" --task ocr

curl -X POST "$BASE_URL/recognize" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"…","task":"ocr","max_new_tokens":512}'
```

Same six tasks as `paddleocr-vl/`: `ocr`, `table`, `formula`, `chart`, `spotting`, `seal`.
Returns `{"text", "task", "elapsed_s", "completion_tokens", "prompt_tokens", "tokens_per_s"}`.

### `/parse-page` — document processing

Same pipeline as `paddleocr-vl/parse-page` — DocLayout-YOLO finds the blocks, a full-width-separator
heuristic orders them, each block is routed to the task that fits its type, markdown is assembled —
but with llama.cpp doing the recognition. Accepts images and PDFs, streams NDJSON.

| | detect | recognise |
| --- | --- | --- |
| `paddleocr-vl` (transformers) | 4.5 s | 38.1 s |
| this app, sequential | 4.4 s | 8.0 s |
| **this app, 32 slots** | 4.2 s | **2.5 s** |

Same page, same 31 blocks. Detection is identical work either way; recognition is **15× faster** than
the transformers deployment.

**Batching.** Blocks used to be sent to llama-server one at a time while a 24 GB L4 held a 1.8 GB
model. `llama-server` now runs with `--parallel` and the blocks go through a `ThreadPoolExecutor`:

| Slots | recognise (31 blocks) | VRAM |
| --- | --- | --- |
| sequential | 7.78 s | 2470 MiB (10.7%) |
| 8 | 3.54 s | 3130 MiB peak (13.6%) |
| **32** | **2.49 – 2.64 s** | 4858 MiB (21.1%) |

VRAM is reported live by `/health` via `nvidia-smi`, so this is measured rather than estimated. Note
the first run after a slot-count change was 4.21 s — slots warm up, so a single cold sample will
mislead. (Fair warning on the numbers above: there are four warm samples at 32 slots but only one at
8, so treat the 8-slot figure as indicative.)

Even at 32 slots the GPU is 79% empty, but pushing further stops paying — the constraint is compute
and memory bandwidth per block, not capacity.

Two details:

- Work is flattened across *all* pages rather than batched per page, so a PDF with only three blocks
  per page still fills the in-flight window.
- Futures are collected in submission order, not completion order, so reading order survives.

Tunable via `LLAMA_PARALLEL` (32), `PARSE_CONCURRENCY` (matches it) and `LLAMA_CTX` (131072 total —
llama.cpp splits context across slots, so this has to grow with `--parallel`; each slot then gets
4096, against a block's ~1-1.5k image tokens plus up to 512 out).

**Detection is now the bottleneck**: ~4.2 s of DocLayout-YOLO against ~2.5 s of recognition. Recognition
went from 38.1 s (transformers) to 2.5 s — a 15× swing that inverted which half of the pipeline is
worth optimising. Anything further on this endpoint belongs in the detector, not the VLM.

### Table output format

`table_format` on `/parse-page` — `html` (default), `markdown`, or `otsl`. Measured on a small table
whose header spans three columns:

| Format | chars | merged cells |
| --- | --- | --- |
| `otsl` (raw model output) | 143 | preserved, via `<lcel>` |
| `markdown` (pipe table) | 118 | **lost** |
| `html` | 252 | preserved, via `colspan` |

Markdown is the cheapest, but the spanning header comes out as `| Laporan Keuangan 2026 |  |  |` — an
LLM reads that as a value belonging to column one. Markdown has no colspan; that is a format
limitation, not a conversion bug. So: **markdown for simple grids, html when cells merge.** `otsl` is
the most compact and lossless, but LLMs have not been trained on it and will not reliably parse it.

Two consequences of the earlier refactor show up here. Because the endpoint is served from the GPU
container, `/parse-page` is a plain generator feeding `StreamingResponse` — no `.remote_gen()` hop
per event, which `paddleocr-vl` still pays. And crops are sent to llama-server as JPEG, following the
payload finding above.

This is why the image now installs torch, doclayout-yolo and pymupdf despite llama.cpp needing none
of them: keeping detection in the same container avoids a network hop per block, and a page can have
30+ blocks.

```bash
curl -N -X POST "$BASE_URL/parse-page" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"…"}'        # or pdf_base64 / image_url / pdf_url
```

`conf` (0.25), `iou` (0.45), `pdf_dpi` (200) and `max_new_tokens` (512, **per block**) are all
overridable. The reading-order and `<ucel>` caveats from `paddleocr-vl/README.md` apply unchanged.

### Where the wall clock actually goes — the model was never the bottleneck

Generation is 2-3.5 s but wall clock started at 19-22 s. Chasing that gap produced a bigger win than
any model-side tuning, and two changes were made as a result.

**1. The endpoint now lives on the GPU container.** It used to be a separate CPU ASGI app dispatching
over `.remote()`, which cost 0.9-4.1 s on `/health` alone — against 2-3.5 s of real inference. Moving
`@modal.asgi_app()` onto the class made that 0.79-0.84 s, and stable. The URL changed from
`…-web-app.modal.run` to `…-llamaserver-serve.modal.run`.

**2. Send JPEG, not PNG.** This was the large one. Identical 1524×1368 page, identical model and task —
only the encoding of the uploaded payload differed:

| Payload | Size | Wall clock |
| --- | --- | --- |
| base64 PNG | 3029 KB | 18.0 – 18.3 s |
| **base64 JPEG** | **929 KB** | **7.5 – 8.3 s** |

2.4× end to end, with nothing touched on the GPU. `gradio_app.py` now re-encodes anything over
`JPEG_ABOVE_KB` (default 400 KB) at `JPEG_QUALITY` (default 92 — higher than the 88 used in the
measurement above, because OCR of small text is precisely where JPEG artefacts would hurt). Set
`JPEG_ABOVE_KB=0` to send bytes verbatim.

Passing `image_url` instead is worse and erratic (13.4-25.8 s): the container then fetches the image
itself, and the benchmark URL is a Beijing CDN roughly 5 s away from a local machine and further from
Modal's region.

Net effect of these two, unchanged model: **19-22 s → 7.5 s.**

### Ideas not yet tried, roughly by expected payoff

1. **Quantise further.** The shipped `.gguf` is ~1 byte/param (≈Q8) for a 0.9B model. `llama-quantize`
   to Q4_K_M would roughly halve it, and decode here is memory-bandwidth bound, so that is the biggest
   model-side lever. The 881 MB mmproj is the vision tower — quantising that is riskier for accuracy.
2. **Collapse the container hop** by serving the endpoint from the GPU class itself instead of a
   separate ASGI app.
3. ~~**`-fa`** (flash attention)~~ — **tried, no measurable gain.** The flag is accepted and is now
   passed (`LLAMA_FLASH_ATTN`, default `on`), but throughput is unchanged: best warm run was
   247.2 tok/s without it and 249.5 tok/s with, i.e. under 1% and inside the run-to-run spread
   (145-249 tok/s across warm calls). The likely explanation is that this llama.cpp build already
   enables flash attention by default, so forcing it on changes nothing. Attempting the `-fa off`
   half of the A/B failed — the env override did not reach the deploy, and the container kept
   reporting `-fa on` — so "already on by default" remains the probable cause rather than a proven one.
   Kept because it is harmless and may matter on other hardware.
4. **`--parallel N`** for continuous batching, which only helps under concurrent load.

Note the spread: warm calls range 2.05-3.52 s (145-249 tok/s) with identical inputs. Any optimisation
worth chasing has to beat that noise, which rules out single-flag tweaks.

## Gradio UI

```bash
PADDLEOCR_VL_GGUF_BASE_URL="https://…-gguf-web-app.modal.run" \
PADDLEOCR_VL_BASE_URL="https://…-vl-web-app.modal.run" \
GRADIO_SERVER_PORT=7862 uv run gradio_app.py
```

```powershell
$env:PADDLEOCR_VL_GGUF_BASE_URL = "https://…-gguf-web-app.modal.run"
$env:PADDLEOCR_VL_BASE_URL      = "https://…-vl-web-app.modal.run"
$env:GRADIO_SERVER_PORT         = "7862"
uv run gradio_app.py
```

Two tabs. **Recognize** runs any of the six tasks against llama.cpp and prints wall / generation /
tok-s for each call. **Compare vs transformers** sends the *same* image and task to both deployments
and shows the outputs side by side with a timing table — that is the tab that answers the question the
benchmark left open, namely whether the speedup costs accuracy on `table` and `formula`. It is only
enabled when `PADDLEOCR_VL_BASE_URL` is set.

## Not done

- **`spotting` needs `image_max_pixels=1605632`** per the model card; that knob isn't wired up here,
  so spotting will run at the default pixel budget.
- **No Gradio UI** — `paddleocr-vl/gradio_app.py` points at a compatible `/recognize`, so switching it
  over is mostly an env-var change.
- **No `@modal.concurrent`.** `llama-server` handles its own batching, so the right setting depends on
  `--parallel`; untested.
- **Output quality was spot-checked, not evaluated.** The text matched on one page. Whether quantisation
  costs accuracy on tables, formulas or dense small text is unmeasured — if you care about
  `table`/`formula` fidelity, compare against `paddleocr-vl/` on your own documents before switching.
