"""PaddleOCR-VL-1.6 on ZeroGPU — single-file Hugging Face Space.

This is the **transformers** path, not GGUF/llama.cpp, and that is forced by how
ZeroGPU works rather than by preference. From the ZeroGPU docs:

    ZeroGPU Spaces are designed to be compatible with most PyTorch-based GPU
    Spaces … a PyTorch CUDA emulation mode is enabled outside @spaces.GPU
    functions, allowing CUDA operations without a real GPU. Inside @spaces.GPU,
    real CUDA is used.

The whole mechanism sits at the PyTorch layer: torch CUDA calls are intercepted
and the decorated call runs where a card is actually attached. `llama-server` is
a native, non-PyTorch process that outlives every call, so there is no seam for
that to hook into — on ZeroGPU it can only ever run on CPU. See
`app_llamacpp.py` in this folder for that variant; it belongs on CPU-basic
hardware or a Docker Space built from ghcr.io/ggml-org/llama.cpp:server-cuda.

Two ZeroGPU rules this file follows, both easy to get wrong:

  * The model is moved to `cuda` at **module level**, not inside the decorated
    function. The docs call lazy `.to('cuda')` "significantly less efficient";
    the emulation mode is what makes a module-level placement legal with no real
    GPU present.
  * Every function that touches the GPU is decorated, and long ones declare a
    duration — the default cap is 60 s, and a multi-page parse blows through it.

Quota is the thing to watch: a free account gets **5 minutes of GPU per day**,
PRO 40. Document parsing spends that fast, so `duration` is computed from the
actual work rather than padded, since a shorter declared duration also improves
queue priority.

Dependencies: requirements.txt next to this file.
"""

from __future__ import annotations

import base64
import html
import re
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import gradio as gr
import spaces
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}
TASK_BY_LABEL = {
    "Text Recognition": "ocr",
    "Table Recognition": "table",
    "Formula Recognition": "formula",
    "Chart Recognition": "chart",
    "Seal Recognition": "seal",
    "Spotting": "spotting",
}
# A paragraph needs a few hundred tokens; a wide table emits OTSL per cell and
# needs thousands. One flat cap truncates tables mid-structure, silently.
TASK_TOKEN_SCALE = {"table": 6, "chart": 4, "formula": 2}

# The image is cut into 28x28 tokens, so max_pixels/(28*28) is literally how many
# image tokens the model attends over — the single biggest lever on prefill cost.
SPOTTING_MAX_PIXELS = 2048 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
SPOTTING_UPSCALE_THRESHOLD = 1500
# The model card reads this off processor.image_processor.min_pixels, but that
# attribute does not exist on PaddleOCRVLImageProcessor — it raises
# AttributeError. This is the value its own preprocessor_config.json ships.
MIN_PIXELS = 112896

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]


# ------------------------------------------------------------ model, at import
# Module level on purpose: ZeroGPU's emulation makes this legal without a real
# card, and the docs say a placement done here is much cheaper than one done
# inside the decorated call.
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()
print(f"[init] {MODEL_ID} loaded, dtype={next(model.parameters()).dtype}")


# ---------------------------------------------------------- optional: layout
_layout = None
_layout_lock = threading.Lock()
_layout_ready = threading.Event()

LAYOUT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_WEIGHTS = "doclayout_yolo_docstructbench_imgsz1024.pt"
LAYOUT_CLASSES = {
    0: "title", 1: "plain text", 2: "abandon", 3: "figure", 4: "figure_caption",
    5: "table", 6: "table_caption", 7: "table_footnote", 8: "isolate_formula",
    9: "formula_caption",
}
LAYOUT_TASK = {
    "title": "ocr", "plain text": "ocr", "figure_caption": "ocr",
    "table": "table", "table_caption": "ocr", "table_footnote": "ocr",
    "isolate_formula": "formula", "formula_caption": "ocr",
}
SKIP_CLASSES = {"abandon"}
FIGURE_CLASSES = {"figure"}
MAX_BLOCKS = 60


def _load_layout() -> None:
    """Background thread: `import doclayout_yolo` pulls in ultralytics and cost
    ~10 s when measured, against 0.3 s to read the weights. Nothing to speed up,
    only somewhere else to put it."""
    global _layout
    try:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        _layout = YOLOv10(hf_hub_download(repo_id=LAYOUT_REPO, filename=LAYOUT_WEIGHTS))
        print("[init] layout detector ready")
    except Exception as e:
        print(f"[init] no layout detector ({type(e).__name__}: {e}) — "
              "pages go to the model whole")
    finally:
        _layout_ready.set()


threading.Thread(target=_load_layout, daemon=True).start()

try:
    import fitz  # pymupdf

    HAS_PDF = True
except Exception:
    HAS_PDF = False


# ------------------------------------------------------------------ inference
def _prepare(pil_image, task: str):
    from PIL import Image

    if task == "spotting":
        w, h = pil_image.size
        if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
            pil_image = pil_image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    max_pixels = SPOTTING_MAX_PIXELS if task == "spotting" else DEFAULT_MAX_PIXELS

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPTS[task]},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        # transformers v5 requires processor kwargs nested under
        # `processor_kwargs`. Passing `images_kwargs=` at the top level — as the
        # model card does — is silently ignored, so the pixel budget never
        # takes effect and spotting quietly runs at the default resolution.
        processor_kwargs={"images_kwargs": {"size": {
            "shortest_edge": getattr(processor.image_processor, "min_pixels", MIN_PIXELS),
            "longest_edge": max_pixels,
        }}},
    ).to(model.device)


def _decode(out, inputs) -> str:
    # The trailing token is EOS; the model card's own example trims it this way.
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:-1]).strip()


def _generate(pil_image, task: str, max_new_tokens: int) -> str:
    inputs = _prepare(pil_image, task)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # The model ships "use_cache": false in BOTH config.json and
            # generation_config.json, which disables the KV cache and makes
            # decoding O(n²). Setting it here took measured throughput from
            # ~2.8 tok/s to 29-49 tok/s. Do not remove.
            use_cache=True,
        )
    return _decode(out, inputs)


@spaces.GPU(duration=60)
def gpu_recognize(pil_image, task: str, max_new_tokens: int) -> str:
    return _generate(pil_image, task, int(max_new_tokens))


def _parse_duration(file_path, max_tokens, conf, dpi):
    """Dynamic duration: pages × a rough per-page budget, capped.

    A flat 60 s fails on anything long, and a flat 300 s wastes quota and queue
    priority on a single image — the docs note a shorter declared duration
    improves both.
    """
    try:
        if HAS_PDF and str(file_path).lower().endswith(".pdf"):
            with fitz.open(file_path) as doc:
                pages = len(doc)
        else:
            pages = 1
    except Exception:
        pages = 1
    return int(min(300, max(60, pages * 45)))


@spaces.GPU(duration=_parse_duration)
def gpu_parse(file_path, max_tokens: int, conf: float, dpi: int):
    """One decorated call for the whole document.

    Deliberately not one call per block: each @spaces.GPU entry is a scheduling
    round trip and counts against the daily quota, and a 30-block page would pay
    that thirty times.
    """
    pages = to_pages(file_path, int(dpi))
    _layout_ready.wait(timeout=120)

    run = Path(tempfile.mkdtemp(prefix="run-"))
    (run / "images").mkdir()
    parts: list[str] = []
    n_fig = 0

    for pi, page in enumerate(pages):
        if len(pages) > 1:
            parts.append(f"---\n\n**Page {pi + 1} / {len(pages)}**")

        if _layout is not None:
            blocks = detect_blocks(page, float(conf), 0.45)
        else:
            blocks = [{"type": "plain text", "bbox": [0, 0, *page.size]}]

        for bi, blk in enumerate(blocks):
            x1, y1, x2, y2 = blk["bbox"]
            crop = page.crop((x1, y1, x2, y2))

            if blk["type"] in FIGURE_CLASSES:
                name = f"p{pi + 1}_b{bi}.jpg"
                crop.thumbnail((1600, 1600))
                crop.save(run / "images" / name, "JPEG", quality=85)
                parts.append(f"![figure p{pi + 1}](images/{name})")
                n_fig += 1
                continue

            task = LAYOUT_TASK.get(blk["type"], "ocr")
            budget = int(max_tokens) * TASK_TOKEN_SCALE.get(task, 1)
            md = as_markdown(blk["type"], _generate(crop, task, budget))
            if md:
                parts.append(md)

    return "\n\n".join(parts), str(run), len(pages), n_fig


def detect_blocks(page, conf: float, iou: float) -> list[dict]:
    import torchvision

    with _layout_lock:   # Ultralytics models are not thread-safe when shared
        det = _layout.predict(page, imgsz=1024, conf=conf, device="cuda")[0]
    boxes, cls, scores = det.boxes.xyxy, det.boxes.cls, det.boxes.conf
    keep = torchvision.ops.nms(boxes.float().cpu(), scores.float().cpu(), iou)
    blocks = []
    for i in keep.tolist():
        kind = LAYOUT_CLASSES.get(int(cls[i].item()), "plain text")
        if kind in SKIP_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in boxes[i].tolist())
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        blocks.append({"type": kind, "bbox": [x1, y1, x2, y2]})
    # Top-to-bottom, left-to-right inside a band. Fine for one column; a
    # multi-column page needs gutter detection, which is out of scope here.
    blocks.sort(key=lambda b: (b["bbox"][1] // 40, b["bbox"][0]))
    return blocks[:MAX_BLOCKS]


# --------------------------------------------------------------- OTSL → HTML
_OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)")


def otsl_to_html(otsl: str) -> str:
    """The `table` task emits OTSL, not HTML or Markdown. Browsers drop the
    unknown tags silently, so rendering it raw looks like garbled text."""
    rows = []
    for raw in otsl.split("<nl>"):
        toks = _OTSL.findall(raw)
        if not toks:
            continue
        cells: list[list] = []
        for tag, text in toks:
            if tag in ("lcel", "xcel") and cells:
                cells[-1][1] += 1
            elif tag in ("ecel", "ucel"):
                cells.append(["", 1])
            else:
                cells.append([text.strip(), 1])
        rows.append(cells)
    if not rows:
        return ""
    out = ['<table border="1" style="border-collapse:collapse">']
    for cells in rows:
        out.append("<tr>")
        for text, span in cells:
            attr = f' colspan="{span}"' if span > 1 else ""
            out.append(f"<td{attr}>{html.escape(text)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def as_markdown(kind: str, text: str) -> str:
    if not text:
        return ""
    if kind == "title":
        return f"## {text}"
    if kind == "table":
        return otsl_to_html(text) or text
    if kind == "isolate_formula":
        return f"$$\n{text}\n$$"
    return text


def to_pages(path: str, dpi: int = 200) -> list:
    from PIL import Image

    if str(path).lower().endswith(".pdf"):
        if not HAS_PDF:
            raise gr.Error("PDF input needs pymupdf — add it to requirements.txt.")
        doc = fitz.open(path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pages = []
        for p in doc:
            # Straight from the pixmap buffer: via tobytes("png") the same
            # pixels are compressed then immediately decompressed, measured at
            # 429 ms/page against 51 ms.
            pix = p.get_pixmap(matrix=mat)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            pages.append(img.convert("RGB") if mode != "RGB" else img)
        doc.close()
        return pages
    return [Image.open(path).convert("RGB")]


def _inline(markdown: str, run: Path) -> str:
    """Data URIs for the preview only; the ZIP keeps images/NAME so the bundle
    resolves offline."""
    for img in (run / "images").glob("*.jpg"):
        b64 = base64.b64encode(img.read_bytes()).decode()
        markdown = markdown.replace(f"images/{img.name}",
                                    f"data:image/jpeg;base64,{b64}")
    return markdown


# ------------------------------------------------------------------ UI actions
def run_recognize(path: str | None, label: str, max_tokens: int):
    if not path:
        raise gr.Error("Upload an image first.")
    from PIL import Image

    task = TASK_BY_LABEL[label]
    t0 = time.time()
    text = gpu_recognize(Image.open(path).convert("RGB"), task, int(max_tokens))
    rendered = otsl_to_html(text) if task == "table" else text
    return rendered or text, text, f"done · wall **{time.time() - t0:.1f} s**"


def run_parse(path: str | None, max_tokens: int, conf: float, dpi: int):
    if not path:
        raise gr.Error("Upload an image or PDF first.")
    t0 = time.time()
    source, run_dir, n_pages, n_fig = gpu_parse(path, int(max_tokens),
                                                float(conf), int(dpi))
    run = Path(run_dir)
    (run / "document.md").write_text(source, encoding="utf-8")
    zip_path = run / f"{Path(path).stem}-ocr.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run / "document.md", "document.md")
        for img in sorted((run / "images").glob("*")):
            z.write(img, f"images/{img.name}")

    mode = "layout-routed" if _layout is not None else "whole-page (no detector)"
    return (
        _inline(source, run), source,
        f"done — {n_pages} pages"
        + (f" · {n_fig} figures" if n_fig else "")
        + f" · {mode} · wall **{time.time() - t0:.1f} s**",
        str(zip_path),
    )


# ------------------------------------------------------------------------- UI
CSS = ".ocr-pane img { max-width: 100%; height: auto; max-height: 45vh; }"

# theme/css go to launch(), not the Blocks constructor: Gradio 6 moved them and
# warns on every start otherwise. The Space installs gradio 6.22, so this is the
# version that actually runs.
with gr.Blocks(title="PaddleOCR-VL · ZeroGPU") as demo:
    gr.Markdown("# PaddleOCR-VL-1.6 · ZeroGPU")
    note = gr.Markdown()
    demo.load(
        lambda: (
            "transformers on ZeroGPU"
            + ("" if HAS_PDF else " · _PDF off (pymupdf missing)_")
            + ("" if _layout is not None else
               " · _no layout detector — pages go to the model whole, so "
               "table/formula routing is off_")
            + " · _free tier is 5 min GPU/day; a long PDF spends it quickly_"
        ),
        outputs=note,
    )

    with gr.Tab("Document Parsing"):
        with gr.Row():
            with gr.Column(scale=5):
                file_p = gr.File(
                    label="Image or PDF", type="filepath",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf"],
                )
                tok_p = gr.Slider(128, 2048, value=512, step=128,
                                  label="Max new tokens per block")
                conf_p = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                   label="Layout confidence")
                dpi_p = gr.Slider(100, 300, value=200, step=25, label="PDF DPI")
                btn_p = gr.Button("Parse Document", variant="primary")
            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab("Preview"):
                        md_p = gr.Markdown(latex_delimiters=LATEX_DELIMS,
                                           max_height="68vh",
                                           elem_classes="ocr-pane")
                    with gr.Tab("Source"):
                        raw_p = gr.Code(language="markdown", lines=26, max_lines=26)
                status_p = gr.Markdown("")
                zip_p = gr.File(label="Download (markdown + images)",
                                interactive=False)

        btn_p.click(run_parse, [file_p, tok_p, conf_p, dpi_p],
                    [md_p, raw_p, status_p, zip_p])

    with gr.Tab("Recognize"):
        with gr.Row():
            with gr.Column(scale=5):
                img_a = gr.Image(label="Image", type="filepath", height=320)
                task_a = gr.Radio(list(TASK_BY_LABEL), value="Text Recognition",
                                  label="Task")
                tok_a = gr.Slider(128, 4096, value=1024, step=128,
                                  label="Max new tokens")
                btn_a = gr.Button("Run", variant="primary")
                stats_a = gr.Markdown("")
            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab("Result"):
                        md_a = gr.Markdown(latex_delimiters=LATEX_DELIMS,
                                           max_height="68vh",
                                           elem_classes="ocr-pane")
                    with gr.Tab("Raw"):
                        raw_a = gr.Code(language="markdown", lines=26, max_lines=26)

        btn_a.click(run_recognize, [img_a, task_a, tok_a], [md_a, raw_a, stats_a])

    gr.Markdown("_`table` returns OTSL markup, converted to HTML before display._")


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), css=CSS)
