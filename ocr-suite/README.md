# ocr-suite

Three OCR deployments that used to live in `paddleocr-vl/`, `paddleocr-vl-gguf/` and
`unlimited-ocr/`, merged into one folder behind **one** Gradio client with a model dropdown.

Each `modal_*.py` is still its own independent Modal app — they cannot be merged into one
script, because they need different base images (llama.cpp CUDA vs. a torch/transformers
`debian_slim`) and different GPUs. What is shared is the client.

```
gradio_app.py                  one UI, three backends, model chosen from a dropdown
modal_paddleocr_vl.py          app "paddleocr-vl"        transformers
modal_paddleocr_vl_gguf.py     app "paddleocr-vl-gguf"   llama.cpp
modal_unlimited_ocr.py         app "unlimited-ocr"       DeepSeek-OCR-style VLM
README.paddleocr-vl.md         per-app notes, kept verbatim from the old folders
README.paddleocr-vl-gguf.md
README.unlimited-ocr.md
Dockerfile.paddleocr-vl        local/non-Modal builds
Dockerfile.unlimited-ocr
```

## Which model to pick

| | Speed | Structure | Use when |
| --- | --- | --- | --- |
| **PaddleOCR-VL · GGUF** | 145–247 tok/s | tables, formulas, charts, seals | default |
| **PaddleOCR-VL · transformers** | 22–28 tok/s | same | checking whether GGUF costs accuracy |
| **Unlimited-OCR** | — | whole-page parse, prompt-steerable | free-text prompting, different model's opinion |

The two PaddleOCR-VL apps run the **same weights** through different runtimes — 7–9× apart on
identical hardware. GGUF is not smaller here (935 MB model + 881 MB mmproj ≈ the 1.9 GB
safetensors); it is a speed win only.

## Deploy

```bash
modal deploy modal_paddleocr_vl_gguf.py
modal deploy modal_paddleocr_vl.py
modal deploy modal_unlimited_ocr.py
```

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

`Dockerfile` builds the merged client — no GPU, no weights, just the HTTP frontend.

```bash
cp .env.example .env          # three base URLs
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

- **Document Parsing** — pages and PDFs. PaddleOCR-VL runs DocLayout-YOLO first and routes each
  block to the task that fits it; Unlimited-OCR parses each page in one pass. Options swap with
  the selected model (`max_new_tokens`/`conf`/table format vs. mode/NGRAM/prompt).
- **Recognize** — one image, one task, no layout detection. PaddleOCR-VL only; Unlimited-OCR has
  no task concept.
- **Compare** — the same document through up to three models, run sequentially, panes side by
  side. This is the tab for the question the benchmarks never answered: whether GGUF's speed
  costs accuracy on tables and formulas.

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
