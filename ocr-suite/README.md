# ocr-suite

OCR deployments and one local Gradio client. Each Modal script is an independent app; the
client selects among configured remote backends and the local OpenDataLoader path.

Each `modal_*.py` is still its own independent Modal app — they cannot be merged into one
script, because they need different base images (llama.cpp CUDA vs. a torch/transformers
`debian_slim`) and different GPUs. What is shared is the client.

```
gradio_app.py                             one UI, backends chosen from a dropdown
unlimited-ocr/modal_unlimited_ocr.py       app "unlimited-ocr"       DeepSeek-OCR-style VLM
unlimited-ocr/README.md
paddleocr-vl/modal_paddleocr_vl.py        app "paddleocr-vl"        transformers
paddleocr-vl/modal_paddleocr_vl_gguf.py   app "paddleocr-vl-gguf"   llama.cpp
paddleocr-vl/modal_paddleocr_vl_gguf_official.py  official PaddleOCR pipeline + llama.cpp
paddleocr-vl/README.paddleocr-vl.md       transformers details
paddleocr-vl/README.paddleocr-vl-gguf.md  custom GGUF details
glm-ocr/modal_glm_ocr_transformers.py     app "glm-ocr-transformers" — the default GLM-OCR path
glm-ocr/modal_glm_ocr_vllm.py             app "glm-ocr-vllm"
 glm-ocr/modal_glm_ocr_sglang.py           app "glm-ocr-sglang" (stopped)
mineru/modal_mineru.py
mineru/modal_mineru_gguf.py
docling/modal_docling.py
Dockerfile                                the Gradio client, JRE included
```

## Which model to pick

| | Where | Structure | Use when |
| --- | --- | --- | --- |
| **OpenDataLoader** | **local CPU** | headings, tables, images, reading order | **the PDF has a text layer — try this first** |
| **PaddleOCR-VL · GGUF** | Modal GPU | tables, formulas, charts, seals | scanned pages; the default OCR path |
| **PaddleOCR-VL · transformers** | Modal GPU | same | checking whether GGUF costs accuracy |
| **Unlimited-OCR** | Modal GPU | whole-page parse, prompt-steerable | free-text prompting, a different model's opinion |

**Reach for OpenDataLoader before any of the OCR backends.** It reads the text the PDF already
contains rather than guessing it back from pixels: exact, no GPU, no upload, and measured at
**1.2 s** for a page the GPU pipeline spends seconds on. It runs in the Gradio process (Java 11+
plus `opendataloader-pdf`; the app finds a JDK that is installed but off `PATH`, and hides the
backend with a reason if it cannot).

Its limit is the mirror image of that strength: a scanned page has no text layer, so it returns
nothing and says so. That is what the OCR backends are for. Deciding between them per page is the
triage described in the repo root — cheap to check, and it is the difference between seconds and
milliseconds for a digital PDF.

The two PaddleOCR-VL apps run the **same weights** through different runtimes — 7–9× apart on
identical hardware. GGUF is not smaller here (935 MB model + 881 MB mmproj ≈ the 1.9 GB
safetensors); it is a speed win only.

## Deploy

```bash
modal deploy paddleocr-vl/modal_paddleocr_vl_gguf.py
modal deploy paddleocr-vl/modal_paddleocr_vl.py
modal deploy unlimited-ocr/modal_unlimited_ocr.py
```

### Official PaddleOCR pipeline variant

`paddleocr-vl/modal_paddleocr_vl_gguf_official.py` keeps the same GGUF + `llama-server` backend, but
replaces the custom DocLayout-YOLO parser with PaddleOCR's official `PaddleOCRVL` pipeline. It is a
separate app for side-by-side testing and does not change the existing GGUF deployment.

```bash
modal deploy paddleocr-vl/modal_paddleocr_vl_gguf_official.py
modal run paddleocr-vl/modal_paddleocr_vl_gguf_official.py --image-url "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
```

This variant installs CPU PaddlePaddle and `paddleocr[doc-parser]`; its `/parse-page` route uses the
official layout analysis, page restructuring, and merged-table handling while recognition remains
served by the local GPU GGUF `llama-server`. CPU layout is intentional here to avoid installing a
second multi-gigabyte CUDA runtime inside the llama.cpp image. `PP-DocLayoutV3` is persisted in the
Modal Volume `paddleocr-vl-official-paddlex-cache`, so its first download is not repeated by new
containers.

App names come from `modal.App("…")` inside each script, not the filename — use those for
`modal app logs`.

## Run the UI

```powershell
$env:PADDLEOCR_VL_GGUF_BASE_URL = "https://<workspace>--paddleocr-vl-gguf-llamaserver-serve.modal.run"
$env:PADDLEOCR_VL_BASE_URL      = "https://<workspace>--paddleocr-vl-web-app.modal.run"
$env:UNLIMITED_OCR_BASE_URL     = "https://<workspace>--unlimited-ocr-web-app.modal.run"
uv run gradio_app.py
```

Backends whose env var is unset are listed as unconfigured rather than failing on click, so the
UI is usable with only one deployed.

### In Docker

`Dockerfile` builds the merged client. The three Modal models stay remote; the only backend baked
in is OpenDataLoader, which needs `openjdk-17-jre-headless` — without a JRE it hides itself with
*"no Java 11+ runtime found"*.

```bash
cp .env.example .env          # the three base URLs
docker build -t ocr-suite-ui .
docker run --rm -p 7870:7860 --env-file .env ocr-suite-ui
```

Three long URLs on a command line are easy to get subtly wrong, and an unset backend shows as
*unconfigured* rather than erroring — so a typo looks like the model simply isn't deployed. Hence
`--env-file` rather than repeated `-e` flags. There is no compose file: one container with no
network, volumes or start-up ordering gets nothing from compose that `--env-file` doesn't already
give.

The container health check probes only Gradio itself. It deliberately does **not** probe the Modal
backends: they scale to zero, and polling would keep GPU containers alive and billing.

### Tabs

- **Document Parsing** — pages and PDFs, by upload **or URL**. PaddleOCR-VL runs DocLayout-YOLO
  first and routes each block to the task that fits it; Unlimited-OCR parses each page in one
  pass. Options swap with the selected model (`max_new_tokens`/`conf`/table format vs.
  mode/NGRAM/prompt). An **Input document** tab shows thumbnails of what is about to be parsed.

  Prefer the URL when you have one: it is the single biggest lever on wall clock. A 27-page PDF
  measured 51 s of server execution against 144 s end to end, the difference being base64 upload
  from the client. With a URL, Modal fetches the file over datacentre network and that gap
  disappears. Uploads preview instantly (the bytes are already local); a URL previews only on
  demand, because fetching it client-side would undo the saving — hence the separate button.

  Preview renders every page by default (`PREVIEW_MAX_PAGES=0`) at `PREVIEW_DPI` (80). Set a
  positive cap for very large PDFs. The parse always uses every page.
- **Recognize** — one image, one task, no layout detection. PaddleOCR-VL only; Unlimited-OCR has
  no task concept.
- **Compare** — the same document through up to three models, run sequentially, panes side by
  side. This is the tab for the question the benchmarks never answered: whether GGUF's speed
  costs accuracy on tables and formulas.

The download bundle is `document.md` + `document.html` + `images/`. The HTML is OpenDataLoader's
own rendering when that backend produced it, and a markdown-it conversion otherwise; the converter
runs with `html=True` because PaddleOCR-VL's table blocks are already raw `<table>` markup with
colspan and rowspan, and escaping them would put visible tags on the page.

**Merged cells need `markdown_with_html`.** Plain markdown is pipe tables, which have no concept
of a span, so every merge flattens to a blank. The flag is on in `_odl_run`; measured on a table
whose grid genuinely omits the internal borders:

```
markdown              colspan=0 rowspan=0
markdown_with_html    <td rowspan="2">No</td> <td colspan="2">Semester</td>
html                  <td rowspan="2">No</td> <td colspan="2">Semester</td>
```

Extra formats (`json`, `text`, `pdf`, `tagged-pdf`) are ticked per run and land in `extras/` in the
ZIP; the **Extras** tab previews whichever is text.

### Streaming and scrolling

Every handler is a generator. The first event fires **before** the network call, because the
first *server* event only arrives after upload + rasterise + layout detection — on a 27-page PDF
that measured over 90 s of blank screen, indistinguishable from a hung UI.

Output panes carry `elem_classes="ocr-pane"`, which gives them `max-height: 68vh; overflow-y:
auto` — their own scrollbox, so a long document no longer pushes the controls off the page. The
CSS is layout-only on purpose: an earlier attempt at hand-theming this UI set background tokens
without the matching text tokens and rendered dark-on-dark.

## Gotchas worth keeping

- **`table` returns OTSL**, not HTML or Markdown. `/parse-page` converts server-side; `/recognize`
  does not, so the client still carries `otsl_to_html()`. Feeding raw OTSL to a Markdown pane
  renders as garbled text because browsers drop the unknown tags.
- **Payload size dominates wall clock.** The same page took 18.0 s as base64 PNG and 7.5 s as
  base64 JPEG. The client re-encodes images above `JPEG_ABOVE_KB` (400) at quality 92; PDFs go
  over verbatim.
- **Cold starts reset TLS.** A request that arrives while a container is starting can fail at the
   handshake with `ConnectionReset`, not a timeout. Retrying once it is warm is the fix.

## Current Backend Map

| Backend | Script | Status |
| --- | --- | --- |
| PaddleOCR-VL GGUF | `paddleocr-vl/modal_paddleocr_vl_gguf.py` | active |
| PaddleOCR-VL transformers | `paddleocr-vl/modal_paddleocr_vl.py` | active |
| PaddleOCR official + GGUF | `paddleocr-vl/modal_paddleocr_vl_gguf_official.py` | active |
| Unlimited-OCR | `unlimited-ocr/modal_unlimited_ocr.py` | active |
| GLM-OCR transformers | `glm-ocr/modal_glm_ocr_transformers.py` | active |
| GLM-OCR vLLM | `glm-ocr/modal_glm_ocr_vllm.py` | active |
| GLM-OCR GGUF | `glm-ocr/modal_glm_ocr_gguf.py` | active |
| GLM-OCR SGLang | `glm-ocr/modal_glm_ocr_sglang.py` | stopped; do not deploy |
| MinerU | `mineru/modal_mineru.py`, `mineru/modal_mineru_gguf.py` | available |
| Docling | `docling/modal_docling.py` | available |

The SGLang deployment is intentionally stopped. Its script is retained for reference only.
