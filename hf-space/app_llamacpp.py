"""PaddleOCR-VL-1.6-GGUF — a single-file Hugging Face Space.

Everything is in here: it fetches a llama.cpp server binary, downloads the GGUF
weights, starts llama-server on localhost, and serves a Gradio UI against it.
No Dockerfile, no sibling modules.

Why llama-server and not llama-cpp-python: this model is multimodal, and the
vision half lives in a separate mmproj GGUF that only `llama-server --mmproj`
knows how to load. The llama-cpp-python path is text-only — it will start, and
it will ignore your image.

Hardware
    CPU Space works but is slow (the model is 0.9B, so it is usable for a demo).

    There is **no Linux CUDA build in the llama.cpp releases** — CUDA ships for
    Windows only (checked against tag b10262). On a GPU Space this tries the
    Vulkan build first, which does drive an NVIDIA card when the container has a
    Vulkan ICD, and falls back to CPU when it does not. For real CUDA, run a
    Docker Space from ghcr.io/ggml-org/llama.cpp:server-cuda and point
    LLAMA_BIN_DIR at it; this file will use that binary instead of downloading.

    ZeroGPU: the Space will *start* (a dummy @spaces.GPU function is declared,
    without which the platform kills it at boot with "No @spaces.GPU function
    detected during startup"), but inference is CPU-only. ZeroGPU attaches the
    card for the duration of a decorated call and llama-server outlives every
    call, so it never sees the GPU. If you are on ZeroGPU purely for the free
    tier, plain CPU-basic hardware gives the same result with less confusion.

Optional extras, both degrade gracefully if missing:
    pymupdf         PDF input
    doclayout-yolo  per-block layout routing; without it a page goes to the
                    model whole, which is fine for plain text but loses the
                    table/formula routing

Dependencies live in requirements.txt next to this file; the layout extras are
commented out there because they pull torch.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import gradio as gr
import requests
from huggingface_hub import hf_hub_download

# ZeroGPU stops any Space that declares no @spaces.GPU function ("No @spaces.GPU
# function detected during startup"), so declaring one is the difference between
# a Space that runs and one that is killed at boot. It does NOT buy GPU
# inference here: ZeroGPU attaches the card for the duration of a decorated
# call, and llama-server is a subprocess that outlives every call and never sees
# it. On ZeroGPU this app is CPU-only — for real acceleration use a paid GPU
# Space, or a Docker Space on ghcr.io/ggml-org/llama.cpp:server-cuda.
try:
    import spaces

    ON_ZEROGPU = True

    @spaces.GPU(duration=1)
    def _zerogpu_probe() -> str:
        return "ok"

except Exception:
    ON_ZEROGPU = False


REPO_ID = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"          # ~935 MB
MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"  # ~881 MB, vision projector
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
BIN_DIR = Path(os.environ.get("LLAMA_BIN_DIR", "llama-bin"))
PORT = int(os.environ.get("LLAMA_PORT", "8080"))

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
# needs thousands. One flat cap truncates big tables mid-structure, and the
# damage is silent — the OTSL just ends.
TASK_TOKEN_SCALE = {"table": 6, "chart": 4, "formula": 2}

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]


# --------------------------------------------------------------- capabilities
def has_gpu() -> bool:
    # On ZeroGPU nvidia-smi can answer even though no long-lived process may use
    # the card, so a positive probe there would send us down the Vulkan path for
    # nothing — several minutes of download to end up on CPU anyway.
    if ON_ZEROGPU:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


GPU = has_gpu()

try:
    import fitz  # pymupdf

    HAS_PDF = True
except Exception:
    HAS_PDF = False

HAS_LAYOUT = False
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
    """Runs in a background thread: `import doclayout_yolo` drags in torch and
    ultralytics, measured at ~10 s against 0.3 s to read the weights. There is
    nothing to speed up, only somewhere else to put it."""
    global _layout, HAS_LAYOUT
    try:
        from doclayout_yolo import YOLOv10

        path = hf_hub_download(repo_id=LAYOUT_REPO, filename=LAYOUT_WEIGHTS,
                               local_dir=str(MODEL_DIR))
        _layout = YOLOv10(path)
        HAS_LAYOUT = True
        print("[init] layout detector ready")
    except Exception as e:
        print(f"[init] layout detector unavailable ({type(e).__name__}: {e}) — "
              "pages will be sent to the model whole")
    finally:
        _layout_ready.set()


# ------------------------------------------------------------- llama.cpp setup
def find_llama_server() -> str | None:
    for candidate in (shutil.which("llama-server"), "/app/llama-server",
                      str(BIN_DIR / "llama-server"), str(BIN_DIR / "build/bin/llama-server")):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def pick_asset(assets: list[dict], flavour: str) -> dict | None:
    """Choose a Linux x64 release asset.

    Reality of the llama.cpp releases, checked against tag b10262: the Linux
    builds are .tar.gz (only the Windows ones are .zip), and **there is no Linux
    CUDA build at all** — CUDA ships for Windows only. So on a Linux Space the
    choice is the plain CPU build or the Vulkan one, which does drive an NVIDIA
    card but needs a Vulkan ICD present in the container.
    """
    want = {"cpu": "bin-ubuntu-x64", "vulkan": "bin-ubuntu-vulkan-x64"}[flavour]
    for a in assets:
        n = a["name"].lower()
        if want in n and (n.endswith(".tar.gz") or n.endswith(".zip")):
            return a
    return None


def download_llama_server(flavour: str) -> str:
    print(f"[init] fetching llama.cpp release ({flavour})…")
    rel = requests.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        timeout=60, headers={"Accept": "application/vnd.github+json"},
    )
    rel.raise_for_status()
    assets = rel.json().get("assets", [])

    asset = pick_asset(assets, flavour)
    if not asset:
        names = ", ".join(a["name"] for a in assets[:8])
        raise RuntimeError(
            f"No '{flavour}' Linux asset in the latest release. Saw: {names}… "
            "Set LLAMA_BIN_DIR to a directory containing llama-server instead."
        )

    target = BIN_DIR / flavour
    target.mkdir(parents=True, exist_ok=True)
    archive = target / asset["name"]
    print(f"[init] downloading {asset['name']} ({asset['size'] / 1e6:.0f} MB)")
    with requests.get(asset["browser_download_url"], stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target)
    archive.unlink(missing_ok=True)

    for p in target.rglob("llama-server"):
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        # The shared objects ship beside the binary, so it must look there.
        os.environ["LD_LIBRARY_PATH"] = (
            f"{p.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        return str(p)
    raise RuntimeError(f"llama-server not found inside {asset['name']}")


def fetch_weights() -> tuple[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, local_dir=str(MODEL_DIR))
    mmproj = hf_hub_download(repo_id=REPO_ID, filename=MMPROJ_FILE, local_dir=str(MODEL_DIR))
    return model, mmproj


PROC: subprocess.Popen | None = None
_llama_ready = threading.Event()
_llama_error: str | None = None


BACKEND = "cpu"   # what actually ended up running; shown in the UI


def _try_start(binary: str, model: str, mmproj: str, offload: bool) -> bool:
    """Spawn llama-server and wait for /health. False if it dies on the way up."""
    global PROC

    # Slots share the context equally and permanently — llama.cpp has no paged
    # KV cache — so total context must scale with the slot count. 3072 per slot
    # clears the largest budget any task gets (512 * 6 for tables).
    slots = int(os.environ.get("LLAMA_PARALLEL", "8" if offload else "2"))
    argv = [
        binary,
        "-m", model,
        "--mmproj", mmproj,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-ngl", "999" if offload else "0",
        "-c", str(3072 * slots),
        "--parallel", str(slots),
        "--temp", "0",       # the model card runs greedy
        "--seed", "42",
    ]
    print(f"[init] {' '.join(argv[1:])}")
    PROC = subprocess.Popen(argv)

    deadline = time.time() + 900
    while time.time() < deadline:
        if PROC.poll() is not None:
            print(f"[init] llama-server exited with {PROC.returncode}")
            return False
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    PROC.terminate()
    return False


def start_llama() -> None:
    """Bring up a server, preferring GPU offload but never depending on it.

    A Vulkan build does drive an NVIDIA card, but it needs a Vulkan ICD inside
    the container and there is no way to know from here whether the Space image
    has one. So try it, and fall back to CPU rather than leaving the Space dead:
    slow output beats no output.
    """
    global BACKEND
    model, mmproj = fetch_weights()

    existing = find_llama_server()
    if existing:
        # Someone supplied a binary (e.g. a Docker Space on the llama.cpp
        # image); trust their build and let -ngl follow the GPU probe.
        if _try_start(existing, model, mmproj, GPU):
            BACKEND = "gpu (supplied binary)" if GPU else "cpu (supplied binary)"
            print(f"[init] llama-server ready — {BACKEND}")
            return
        raise RuntimeError("the supplied llama-server failed to start")

    plan = [("vulkan", True), ("cpu", False)] if GPU else [("cpu", False)]
    for flavour, offload in plan:
        try:
            binary = download_llama_server(flavour)
        except Exception as e:
            print(f"[init] {flavour} build unavailable: {e}")
            continue
        if _try_start(binary, model, mmproj, offload):
            BACKEND = f"{flavour}{' + GPU offload' if offload else ''}"
            print(f"[init] llama-server ready — {BACKEND}")
            return
        print(f"[init] {flavour} build did not come up; trying the next option")

    raise RuntimeError("no llama.cpp build started successfully")


def boot() -> None:
    """Runs in the background so Gradio can bind its port immediately.

    Doing this before launch() means a Space spends several minutes downloading
    ~1.8 GB of weights plus a llama.cpp build with nothing listening, and the
    platform marks it as failed to start. Requests that arrive early wait on
    _llama_ready instead.
    """
    global _llama_error
    try:
        start_llama()
    except Exception as e:
        _llama_error = f"{type(e).__name__}: {e}"
        print(f"[init] FAILED: {_llama_error}")
    finally:
        _llama_ready.set()
    _load_layout()


def await_llama() -> None:
    if not _llama_ready.wait(timeout=1800):
        raise gr.Error("Model is still starting up. Try again in a minute.")
    if _llama_error:
        raise gr.Error(f"llama-server did not start — {_llama_error}")


# ------------------------------------------------------------------ inference
def _data_url(pil, quality: int = 92) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate(pil, task: str, max_tokens: int):
    """Yields growing text. llama-server answers `"stream": true` with SSE; we
    consume that and hand out plain strings."""
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(pil)}},
                {"type": "text", "text": PROMPTS[task]},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    parts: list[str] = []
    usage: dict = {}
    finish = None
    last = 0.0
    with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                       json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
            now = time.time()
            if parts and now - last >= 0.08:   # coalesce; 600 tokens is 600 events
                last = now
                yield "".join(parts), usage, False
    yield "".join(parts).strip(), usage, finish == "length"


# --------------------------------------------------------------- OTSL → HTML
_OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)")


def otsl_to_html(otsl: str) -> str:
    """The `table` task emits OTSL, not HTML or Markdown. Browsers silently drop
    the unknown tags, so rendering it raw looks like garbled text."""
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
            span_attr = f' colspan="{span}"' if span > 1 else ""
            out.append(f"<td{span_attr}>{html.escape(text)}</td>")
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


# ------------------------------------------------------------------ page prep
def to_pages(path: str, dpi: int = 200) -> list:
    from PIL import Image

    if Path(path).suffix.lower() == ".pdf":
        if not HAS_PDF:
            raise gr.Error("PDF input needs pymupdf — add it to requirements.txt.")
        doc = fitz.open(path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pages = []
        for p in doc:
            # Straight from the pixmap buffer: going via tobytes("png")
            # compresses and immediately decompresses the same pixels, measured
            # at 429 ms/page against 51 ms.
            pix = p.get_pixmap(matrix=mat)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            pages.append(img.convert("RGB") if mode != "RGB" else img)
        doc.close()
        return pages
    return [Image.open(path).convert("RGB")]


def detect_blocks(page, conf: float, iou: float) -> list[dict]:
    import torchvision

    with _layout_lock:   # Ultralytics models are not thread-safe when shared
        det = _layout.predict(page, imgsz=1024, conf=conf,
                              device="cuda" if GPU else "cpu")[0]
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
    # Top-to-bottom, left-to-right within a band. Good enough for one column;
    # multi-column pages need gutter detection, which is out of scope here.
    blocks.sort(key=lambda b: (b["bbox"][1] // 40, b["bbox"][0]))
    return blocks[:MAX_BLOCKS]


# ------------------------------------------------------------------ UI actions
def run_recognize(path: str | None, label: str, max_tokens: int):
    if not path:
        raise gr.Error("Upload an image first.")
    await_llama()
    from PIL import Image

    task = TASK_BY_LABEL[label]
    img = Image.open(path).convert("RGB")
    t0 = time.time()
    text, usage, truncated = "", {}, False
    for text, usage, truncated in generate(img, task, int(max_tokens)):
        rendered = otsl_to_html(text) if task == "table" else text
        yield rendered or text, text, f"streaming · {len(text)} chars · {time.time() - t0:.1f} s"

    stats = f"wall **{time.time() - t0:.1f} s**"
    if usage.get("completion_tokens"):
        tok_s = usage["completion_tokens"] / max(time.time() - t0, 1e-6)
        stats += f" · {usage['completion_tokens']} tokens · **{tok_s:.1f} tok/s**"
    if truncated:
        stats += " · ⚠️ hit the token cap — output is incomplete"
    rendered = otsl_to_html(text) if task == "table" else text
    yield rendered or text, text, stats


def run_parse(path: str | None, max_tokens: int, conf: float, dpi: int):
    if not path:
        raise gr.Error("Upload an image or PDF first.")

    t0 = time.time()
    yield "", "", "_waiting for the model to finish starting…_", None
    await_llama()
    yield "", "", "_preparing pages…_", None

    pages = to_pages(path, int(dpi))
    _layout_ready.wait(timeout=180)   # loaded in the background at startup

    run = Path(tempfile.mkdtemp(prefix="run-"))
    (run / "images").mkdir()
    parts: list[str] = []
    n_fig = 0

    def flush() -> tuple[str, str]:
        source = "\n\n".join(parts)
        preview = source
        for img_path in (run / "images").glob("*.jpg"):
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            preview = preview.replace(f"images/{img_path.name}",
                                      f"data:image/jpeg;base64,{b64}")
        return preview, source

    for pi, page in enumerate(pages):
        if len(pages) > 1:
            parts.append(f"---\n\n**Page {pi + 1} / {len(pages)}**")

        if HAS_LAYOUT:
            blocks = detect_blocks(page, float(conf), 0.45)
        else:
            # No detector: the whole page goes in as one `ocr` block. Fine for
            # plain text, but table structure and formulas are lost.
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
            text = ""
            for text, _usage, _trunc in generate(crop, task, budget):
                preview, source = flush()
                partial = as_markdown(blk["type"], text)
                yield (
                    preview + ("\n\n" + partial if partial else ""),
                    source + ("\n\n" + partial if partial else ""),
                    f"page {pi + 1}/{len(pages)} · block {bi + 1}/{len(blocks)} "
                    f"· {time.time() - t0:.1f} s",
                    None,
                )
            md = as_markdown(blk["type"], text)
            if md:
                parts.append(md)

    preview, source = flush()
    (run / "document.md").write_text(source, encoding="utf-8")
    zip_path = run / f"{Path(path).stem}-ocr.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run / "document.md", "document.md")
        for img in sorted((run / "images").glob("*")):
            z.write(img, f"images/{img.name}")

    mode = "layout-routed" if HAS_LAYOUT else "whole-page (no layout detector)"
    yield (
        preview, source,
        f"done — {len(pages)} pages"
        + (f" · {n_fig} figures" if n_fig else "")
        + f" · {mode} · wall **{time.time() - t0:.1f} s**",
        str(zip_path),
    )


# ------------------------------------------------------------------------- UI
CSS = """
.ocr-pane img { max-width: 100%; height: auto; max-height: 45vh; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="PaddleOCR-VL GGUF") as demo:
    gr.Markdown("# PaddleOCR-VL-1.6 · GGUF / llama.cpp")
    backend_note = gr.Markdown("_starting…_")
    demo.load(
        lambda: (
            f"Backend: **{BACKEND}**"
            + (" · _ZeroGPU: inference runs on CPU — the card is only attached "
               "during decorated calls, and llama-server outlives them_"
               if ON_ZEROGPU else "")
            + ("" if HAS_PDF else " · _PDF input off (pymupdf missing)_")
            + ("" if HAS_LAYOUT else " · _no layout detector — pages go to the "
                                     "model whole, so table/formula routing is off_")
            + (f" · ⚠️ {_llama_error}" if _llama_error else "")
        ),
        outputs=backend_note,
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
                                           max_height="68vh", elem_classes="ocr-pane")
                    with gr.Tab("Source"):
                        raw_p = gr.Code(language="markdown", lines=26, max_lines=26)
                status_p = gr.Markdown("")
                zip_p = gr.File(label="Download (markdown + images)", interactive=False)

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
                                           max_height="68vh", elem_classes="ocr-pane")
                    with gr.Tab("Raw"):
                        raw_a = gr.Code(language="markdown", lines=26, max_lines=26)

        btn_a.click(run_recognize, [img_a, task_a, tok_a], [md_a, raw_a, stats_a])

    gr.Markdown(
        "_`table` returns OTSL markup, converted to HTML here before display._"
    )


if __name__ == "__main__":
    threading.Thread(target=boot, daemon=True).start()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
