from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import modal
import modal.experimental


APP_NAME = "paddleocr-vl-gguf"
REPO_ID = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"          # ~935 MB
MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"  # ~881 MB, vision projector
MODEL_DIR = "/models"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
PORT = 8080

# Flash attention. Recent llama.cpp builds take on|off|auto and may already
# default to auto, in which case forcing "on" changes nothing — set to "off" to
# A/B it against the 145-247 tok/s baseline measured without the flag.
FLASH_ATTN = os.environ.get("LLAMA_FLASH_ATTN", "on")

# /parse-page sends one request per detected block, and a page had 31 of them,
# each ~0.25 s — all strictly sequential while a 24 GB L4 held a 1.8 GB model.
# --parallel gives llama-server that many decode slots; PARSE_CONCURRENCY is how
# many blocks we actually keep in flight. Context is divided across slots, so
# CTX_TOTAL must grow with PARALLEL or each slot gets too little.
#
# Sized from measurement, not guesswork: at --parallel 8 the container used
# 2470 MiB idle and peaked at 3130 MiB of 23034 — 87% of the L4 sat empty while
# 31 blocks queued through 8 slots in ~4 waves. 32 slots lets a typical page go
# in one wave. Each slot gets CTX_TOTAL/PARALLEL; a block's prompt is ~1-1.5k
# image tokens plus up to 512 out, so 4096 per slot is already generous.
#
# 64 slots at 3072 each rather than 32 at 4096. llama.cpp has no PagedAttention:
# it splits context equally and permanently across slots at startup, so a slot
# sized 4096 while blocks measure `n_tokens = 1756` in the logs wastes over half
# its KV cache. 3072 still clears the largest budget any task gets
# (512 * TASK_TOKEN_SCALE["table"] = 3072), so tables are not cut short, and the
# freed memory buys twice the slots.
PARALLEL = int(os.environ.get("LLAMA_PARALLEL", "64"))
CTX_TOTAL = int(os.environ.get("LLAMA_CTX", str(3072 * 64)))
PARSE_CONCURRENCY = int(os.environ.get("PARSE_CONCURRENCY", str(PARALLEL)))

# Same six prompts as the transformers deployment in paddleocr-vl/.
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

# ---- Layout detection for /parse-page --------------------------------------
# Ported from paddleocr-vl/modal_app.py, where this pipeline was built and
# verified. Same detector, same class map, same reading-order heuristic; only
# the recognition backend differs (llama.cpp here, transformers there).
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
SKIP_CLASSES = {"abandon", "figure"}  # headers/footers/page numbers, and images
MAX_BLOCKS = 60

# A text paragraph needs a couple of hundred tokens; a wide table emits OTSL for
# every cell and can need thousands. One flat max_new_tokens truncates big tables
# mid-row and the damage is silent — the OTSL just ends, and the converter
# happily renders whatever rows it got. Scale the budget by block type instead.
TASK_TOKEN_SCALE = {"table": 6, "chart": 4, "formula": 2}

app = modal.App(APP_NAME)
models = modal.Volume.from_name("paddleocr-vl-gguf-models", create_if_missing=True)


def _fetch_weights():
    """Downloads into the Volume at build time.

    Baking these into the image layer instead was tried and does not work here:
    `modal deploy` runs two builders concurrently against the same image id, and
    the one that saved first won with an empty /models — the deployed image had
    no /models directory at all and llama-server crash-looped on a missing GGUF.
    A Volume is immune to that race because it lives outside the image, so
    whichever builder wins, the weights are already there.

    The layout detector is fetched here too, which keeps hf_hub_download off the
    boot path (it was only hitting cache, but still checking the Hub)."""
    from huggingface_hub import hf_hub_download

    for fname in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=MODEL_DIR)
    hf_hub_download(repo_id=LAYOUT_REPO, filename=LAYOUT_WEIGHTS, local_dir=MODEL_DIR)


# llama.cpp's own prebuilt CUDA server image. Building llama.cpp from source
# with CUDA takes 10-20 minutes per rebuild; this is the maintained binary.
# `Hermas GGUF/` in this repo uses llama-cpp-python instead, but that path is
# text-only — multimodal GGUF needs llama-server's --mmproj.
image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12"
    )
    .entrypoint([])  # the base image starts llama-server; we manage it ourselves
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "huggingface_hub[hf_xet]",
        "requests",
        "Pillow",
        "fastapi[standard]",
        # For /parse-page only. This drags torch into an image that otherwise
        # needs none — llama.cpp does the inference — but keeping layout
        # detection in the same container avoids a network hop per block, and a
        # page can have 30+ blocks.
        "torch",
        "torchvision",
        "doclayout-yolo==0.0.3",
        "pymupdf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_fetch_weights, volumes={MODEL_DIR: models}, timeout=60 * 30)
)


def _download_bytes(url: str) -> bytes:
    import requests

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.content


def _pdf_to_pages(pdf_bytes: bytes, dpi: int = 200) -> list:
    import fitz
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    # Read the pixmap's buffer straight into PIL. Going via tobytes("png")
    # compresses the raster and immediately decompresses it back to the same
    # pixels: measured 429 ms/page against 51 ms/page direct, so an 8-page PDF
    # spent ~3 s doing nothing. get_pixmap() defaults to alpha=False, but check
    # rather than assume — the stride is wrong if that ever changes.
    pages = []
    for p in doc:
        pix = p.get_pixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        pages.append(img.convert("RGB") if mode != "RGB" else img)
    doc.close()
    return pages


def _resolve_pages(image_url, image_base64, pdf_url, pdf_base64, pdf_dpi) -> list:
    from PIL import Image

    if pdf_url or pdf_base64:
        data = _download_bytes(pdf_url) if pdf_url else base64.b64decode(pdf_base64)
        return _pdf_to_pages(data, dpi=pdf_dpi)
    if image_url:
        data = _download_bytes(image_url)
    elif image_base64:
        s = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        data = base64.b64decode(s)
    else:
        raise ValueError("Provide image_url, image_base64, pdf_url or pdf_base64.")
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def _reading_order(blocks: list[dict], page_w: int) -> list[dict]:
    """Full-width blocks act as horizontal separators; between them, narrow
    blocks read left column then right. Heuristic — 3+ columns and text wrapped
    around figures come out wrong. See paddleocr-vl/README.md."""
    ratio, mid = 0.55, page_w / 2

    def by_column(band):
        left = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
        right = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
        return sorted(left, key=lambda b: b["bbox"][1]) + sorted(right, key=lambda b: b["bbox"][1])

    wide = sorted(
        [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) >= ratio * page_w],
        key=lambda b: b["bbox"][1],
    )
    narrow = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) < ratio * page_w]

    ordered, prev_y = [], float("-inf")
    for wb in wide:
        ordered.extend(by_column([b for b in narrow if prev_y <= b["bbox"][1] < wb["bbox"][1]]))
        ordered.append(wb)
        prev_y = wb["bbox"][1]
    ordered.extend(by_column([b for b in narrow if b["bbox"][1] >= prev_y]))
    return ordered


def _otsl_to_html(otsl: str) -> str:
    """The `table` task emits OTSL, not HTML. Duplicated across the deployments
    on purpose — each modal_app.py stands alone."""
    import html
    import re

    rows = []
    for raw_row in otsl.split("<nl>"):
        tokens = re.findall(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)", raw_row)
        if not tokens:
            continue
        cells: list[list] = []
        for tag, text in tokens:
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


def _otsl_to_grid(otsl: str) -> list[list[str]]:
    """OTSL -> rectangular grid of cell texts. A cell spanning N columns keeps
    its text in the first slot and leaves the rest empty; that loses the span,
    which is exactly why markdown is the lossy option (see _as_markdown)."""
    import re

    grid = []
    for raw_row in otsl.split("<nl>"):
        tokens = re.findall(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)", raw_row)
        if not tokens:
            continue
        row: list[str] = []
        for tag, text in tokens:
            if tag in ("lcel", "xcel"):
                row.append("")           # continuation of the cell to the left
            elif tag in ("ecel", "ucel"):
                row.append("")
            else:
                row.append(text.strip())
        grid.append(row)
    return grid


def _otsl_to_pipe(otsl: str) -> str:
    grid = _otsl_to_grid(otsl)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    def line(cells):
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    head, *body = grid
    return "\n".join([line(head), "| " + " | ".join(["---"] * width) + " |",
                      *(line(r) for r in body)])


def _as_markdown(kind: str, text: str, table_format: str = "html") -> str:
    """table_format:
      html     - <table> with real colspan. Preserves merged cells. Default.
      markdown - pipe table. ~40% fewer characters, but markdown has no colspan,
                 so merged cells collapse to blanks and the structure is lossy.
      otsl     - the model's raw output. Most compact, but LLMs have not seen
                 this format and will not reliably interpret it.
    """
    if not text:
        return ""
    if kind == "title":
        return f"## {text}"
    if kind == "table":
        if table_format == "markdown":
            return _otsl_to_pipe(text) or text
        if table_format == "otsl":
            return text
        return _otsl_to_html(text) or text
    if kind == "isolate_formula":
        return f"$$\n{text}\n$$"
    return text


def _pil_data_url(pil) -> str:
    """JPEG, not PNG: payload size dominated wall clock in this app's own
    benchmark (3029 KB PNG -> 18 s, 929 KB JPEG -> 7.5 s). Crops go over
    localhost here, but the encode is still cheaper."""
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _image_data_url(image_url: str | None, image_base64: str | None) -> str:
    """llama-server takes images as OpenAI-style data URLs."""
    if image_base64:
        b64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    elif image_url:
        b64 = base64.b64encode(_download_bytes(image_url)).decode("ascii")
    else:
        raise ValueError("Provide image_url or image_base64.")
    return f"data:image/png;base64,{b64}"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=900,
    # Idle GPU minutes cost more than the occasional restore, so this stays
    # short. Raise it if repeated cold starts during a working session become
    # more annoying than the idle bill.
    scaledown_window=120,
    volumes={MODEL_DIR: models},
    # Snapshots removed. The GPU snapshot is experimental and was restoring in
    # ~42 s — worse than simply booting, because a plain start is llama-server
    # 4.4 s + warmup 0.4 s + DocLayout-YOLO 12.9 s, and mmap makes the first of
    # those cheaper still once --load-mode none is no longer forced. Restoring
    # several GB of GPU state over the network is not obviously cheaper than
    # re-reading 1.8 GB of weights, and here it measured slower.
    #
    # To bring it back: re-add enable_memory_snapshot=True +
    # experimental_options={"enable_gpu_snapshot": True}, split enter() into
    # snap=True/snap=False again, and restore "--load-mode", "none" in argv —
    # file-backed mmap does not survive a snapshot restore.
)
# Modal runs one input per container unless told otherwise, web endpoints
# included. Without this, the Gradio UI polling /health while /parse-page is in
# flight spawns a *second* container — a full cold start next to a warm one
# holding 64 idle llama.cpp slots. That, not the boot path, is what made cold
# starts show up during ordinary use.
#
# Kept at 2 deliberately: one real request plus a /health poll is the case that
# was costing a cold start, and a single /parse-page already fans out to
# PARSE_CONCURRENCY threads, so a second concurrent parse mostly queues behind
# the first on the same slots rather than going faster.
@modal.concurrent(max_inputs=2)
class LlamaServer:
    # ---------- llama-server boot helpers -----------------------------------
    def _spawn_llama(self) -> None:
        """Start llama-server and block until /health answers."""
        import shutil
        import subprocess

        import requests

        binary = shutil.which("llama-server") or "/app/llama-server"
        argv = [
            binary,
            "-m", f"{MODEL_DIR}/{MODEL_FILE}",
            "--mmproj", f"{MODEL_DIR}/{MMPROJ_FILE}",
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "-ngl", "999",       # all layers on GPU
            # Load mode is left at the default (mmap). It was forced to "none"
            # (the current spelling of the deprecated --no-mmap; `--help` on
            # this image lists none|mmap|mlock|mmap+mlock|dio) purely because
            # file-backed mappings do not survive a memory-snapshot restore.
            # With snapshots gone, mmap is the faster path — put it back only
            # alongside snapshots.
            "-c", str(CTX_TOTAL),
            "--parallel", str(PARALLEL),
            "--temp", "0",       # model card runs greedy
            "--seed", "42",      # deterministic across snapshot restores
            "-fa", FLASH_ATTN,
        ]
        print(f"[INIT] argv={' '.join(argv[1:])}")
        t0 = time.time()
        self.proc = subprocess.Popen(argv)

        # Poll until the server answers; it loads ~1.8 GB before it does.
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy in 600s")

        print(f"[INIT] llama-server ready in {time.time() - t0:.1f}s (binary={binary})")

    def _warmup_llama(self) -> None:
        """One tiny OCR request so CUDA kernels and the mmproj image path are
        resident in GPU memory before the snapshot is taken. Otherwise the
        first real request after each restore pays a one-off kernel/JIT tax."""
        import io as _io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (256, 64), "white")
        ImageDraw.Draw(img).text((8, 20), "warmup 0123", fill="black")
        buf = _io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        t0 = time.time()
        text, _ = self._llama_generate(data_url, "ocr", 16)
        print(
            f"[INIT] warmup request done in {time.time() - t0:.1f}s "
            f"(reply={len(text)} chars)"
        )

    def _load_layout(self) -> None:
        """Layout detector for /parse-page. Defensive: /recognize must keep
        working if this can't load."""
        self.layout = None
        try:
            # Split the timings: this step measured 10.4s and it was not obvious
            # how much was importing torch/ultralytics versus building the graph.
            t1 = time.time()
            from doclayout_yolo import YOLOv10

            t_import = time.time() - t1

            t2 = time.time()
            path = os.path.join(MODEL_DIR, LAYOUT_WEIGHTS)
            if not os.path.exists(path):  # weights are baked in; this is a fallback
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=LAYOUT_REPO, filename=LAYOUT_WEIGHTS, local_dir=MODEL_DIR
                )
            self.layout = YOLOv10(path)
            print(
                f"[INIT] layout detector ready in {time.time() - t1:.1f}s "
                f"(import {t_import:.1f}s + load {time.time() - t2:.1f}s)"
            )
        except Exception as e:
            print(f"[INIT] layout detector unavailable: {type(e).__name__}: {e}")
        finally:
            self.layout_ready.set()

    @modal.enter()
    def start(self):
        """Runs on every cold start. Previously split into snap=True/snap=False
        around a memory snapshot; that restored slower than booting outright, so
        it is one phase again."""
        import threading

        # Ultralytics-derived models keep mutable state on the instance across
        # predict() calls and are documented as not thread-safe on a shared
        # instance. llama.cpp handles its own concurrency across slots, so only
        # detection needs serialising.
        self.layout_lock = threading.Lock()
        self.layout = None
        self.layout_ready = threading.Event()

        t0 = time.time()
        self._spawn_llama()
        self._warmup_llama()
        # Off the critical path. Instrumenting the step showed 10.1s of it is
        # `import doclayout_yolo` pulling in torch + ultralytics and only 0.3s is
        # reading the weights, so there is nothing to make faster — only
        # somewhere else to put it. /recognize never needs the detector, and
        # /parse-page waits on layout_ready, by which time the request's own
        # upload (7s at best, measured) has usually covered the import.
        threading.Thread(target=self._load_layout, daemon=True).start()
        print(f"[INIT] container ready in {time.time() - t0:.1f}s (layout loading)")

    @modal.exit()
    def stop(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)

    # Served by *this* container, not a separate ASGI one. The earlier layout
    # had a CPU web_app dispatching over .remote(), which measured 0.9-4.1 s of
    # pure round trip on /health alone — against 2-3.5 s of actual inference.
    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, HTTPException

        api = FastAPI(title=APP_NAME)

        @api.get("/health")
        def health():
            info: dict[str, Any] = {
                "ok": True,
                "backend": "llama.cpp",
                "gpu": GPU_TYPE,
                "tasks": list(PROMPTS),
                "parallel_slots": PARALLEL,
                "ctx_total": CTX_TOTAL,
                "parse_concurrency": PARSE_CONCURRENCY,
                "layout_detector": self.layout is not None,
                # /health answering while this is true is the point of loading
                # the detector in the background — the container is already
                # serving /recognize.
                "layout_loading": not self.layout_ready.is_set(),
            }
            # Actual VRAM, not an estimate — llama.cpp, the layout model and the
            # CUDA context all live in this container.
            try:
                import subprocess

                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                used, total, util = (v.strip() for v in out.stdout.strip().split(","))
                info["vram_used_mib"] = int(used)
                info["vram_total_mib"] = int(total)
                info["vram_used_pct"] = round(int(used) / int(total) * 100, 1)
                info["gpu_util_pct"] = int(util)
            except Exception as e:
                info["vram_error"] = f"{type(e).__name__}: {e}"
            return info

        @api.post("/parse-page")
        def parse_page_route(payload: dict[str, Any]):
            import json

            from fastapi.responses import StreamingResponse

            # The detector loads in a background thread, so on a cold container
            # it may still be importing. Block here rather than 503-ing: this is
            # the only route that needs it, and the wait replaces time the caller
            # would otherwise have spent inside @modal.enter() anyway.
            if not self.layout_ready.wait(timeout=180):
                raise HTTPException(503, detail="Layout detector still loading.")
            if self.layout is None:
                raise HTTPException(
                    503, detail="Layout detector failed to load; see [INIT] logs."
                )
            if not any(
                payload.get(k)
                for k in ("image_url", "image_base64", "pdf_url", "pdf_base64")
            ):
                raise HTTPException(
                    400,
                    detail="Provide image_url, image_base64, pdf_url or pdf_base64.",
                )

            # Served from the GPU container itself, so this can just be a plain
            # generator — no .remote_gen() round trip like paddleocr-vl needs.
            def events():
                for chunk in self._parse_page(
                    image_url=payload.get("image_url"),
                    image_base64=payload.get("image_base64"),
                    pdf_url=payload.get("pdf_url"),
                    pdf_base64=payload.get("pdf_base64"),
                    pdf_dpi=int(payload.get("pdf_dpi", 200)),
                    conf=float(payload.get("conf", 0.25)),
                    iou=float(payload.get("iou", 0.45)),
                    max_new_tokens=int(payload.get("max_new_tokens", 512)),
                    table_format=str(payload.get("table_format", "html")),
                ):
                    yield json.dumps(chunk) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson")

        @api.post("/recognize")
        def recognize_route(payload: dict[str, Any]):
            """NDJSON, one object per update — same shape as /parse-page and as
            the transformers deployment, so one client reads both."""
            import json

            from fastapi.responses import StreamingResponse

            task = str(payload.get("task", "ocr"))
            if task not in PROMPTS:
                raise HTTPException(400, detail=f"Unknown task {task!r}.")
            if not (payload.get("image_url") or payload.get("image_base64")):
                raise HTTPException(400, detail="Provide image_url or image_base64.")

            def events():
                for chunk in self._recognize_stream(
                    image_url=payload.get("image_url"),
                    image_base64=payload.get("image_base64"),
                    task=task,
                    max_new_tokens=int(payload.get("max_new_tokens", 1024)),
                ):
                    yield json.dumps(chunk) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api

    def _detect_blocks(self, page, conf: float, iou: float) -> list[dict]:
        import torchvision

        with self.layout_lock:
            det = self.layout.predict(page, imgsz=1024, conf=conf, device="cuda")[0]
        boxes, cls, scores = det.boxes.xyxy, det.boxes.cls, det.boxes.conf
        keep = torchvision.ops.nms(
            boxes=boxes.float().cpu(), scores=scores.float().cpu(), iou_threshold=iou
        )
        blocks = []
        for i in keep.tolist():
            kind = LAYOUT_CLASSES.get(int(cls[i].item()), "plain text")
            if kind in SKIP_CLASSES:
                continue
            x1, y1, x2, y2 = (int(v) for v in boxes[i].tolist())
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            blocks.append({"type": kind, "bbox": [x1, y1, x2, y2], "score": float(scores[i])})
        return _reading_order(blocks, page.size[0])[:MAX_BLOCKS]

    def _llama_generate(
        self, data_url: str, task: str, max_new_tokens: int
    ) -> tuple[str, bool]:
        """Returns (text, truncated). truncated=True means the model hit the token
        cap without emitting a stop — for a table that means the OTSL ends
        mid-structure and the rendered result is incomplete."""
        import requests

        r = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": PROMPTS[task]},
                        ],
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            },
            timeout=900,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        return text, choice.get("finish_reason") == "length"

    def _parse_page(
        self,
        image_url=None, image_base64=None, pdf_url=None, pdf_base64=None,
        pdf_dpi: int = 200, conf: float = 0.25, iou: float = 0.45,
        max_new_tokens: int = 512, table_format: str = "html",
    ):
        """Detect layout blocks, then recognise each with the task that suits its
        type, all inside this container — a page can have 30+ blocks and a hop
        per block would dominate."""
        pages = _resolve_pages(image_url, image_base64, pdf_url, pdf_base64, pdf_dpi)

        t_det = time.time()
        per_page = [self._detect_blocks(p, conf, iou) for p in pages]
        total = sum(len(b) for b in per_page)
        det_ms = int((time.time() - t_det) * 1000)
        print(f"[LAYOUT] pages={len(pages)} blocks={total} detect_ms={det_ms}")

        yield {
            "blocks": [], "markdown": "", "total": total, "pages": len(pages),
            "detect_s": round(det_ms / 1000, 2), "elapsed_s": 0.0, "done": False,
        }

        t0 = time.time()
        done: list[dict] = []
        multi = len(pages) > 1

        def render() -> str:
            parts, cur = [], None
            for b in done:
                if multi and b["page"] != cur:
                    cur = b["page"]
                    parts.append(f"---\n\n**Page {cur + 1} / {len(pages)}**")
                md = _as_markdown(b["type"], b.get("text", ""), table_format)
                if md:
                    # Silent truncation of a table is worse than a visible gap —
                    # the OTSL simply ends and the converter renders the partial
                    # rows as if they were the whole table.
                    if b.get("truncated"):
                        md += (
                            f"\n\n> ⚠️ truncated at {b.get('max_tokens')} tokens — "
                            f"this {b['type']} is incomplete; raise max_new_tokens."
                        )
                    parts.append(md)
            return "\n\n".join(parts)

        # Flatten to one work list so blocks from different pages can share the
        # in-flight window; `done` is still appended in reading order because we
        # walk the ordered list and only collect a block once its future is done.
        work = [
            (page_idx, blk, page)
            for page_idx, (page, blocks) in enumerate(zip(pages, per_page))
            for blk in blocks
        ]

        from concurrent.futures import ThreadPoolExecutor

        def recognise(item):
            page_idx, blk, page = item
            x1, y1, x2, y2 = blk["bbox"]
            task = LAYOUT_TASK.get(blk["type"], "ocr")
            budget = max_new_tokens * TASK_TOKEN_SCALE.get(task, 1)
            truncated = False
            try:
                text, truncated = self._llama_generate(
                    _pil_data_url(page.crop((x1, y1, x2, y2))), task, budget
                )
            except Exception as e:
                text, blk["error"] = "", str(e)
            blk.update(
                page=page_idx, task=task, text=text,
                max_tokens=budget, truncated=truncated,
            )
            return blk

        with ThreadPoolExecutor(max_workers=max(1, PARSE_CONCURRENCY)) as pool:
            futures = [pool.submit(recognise, item) for item in work]
            for fut in futures:  # in submission order = reading order
                done.append(fut.result())
                yield {
                    "blocks": done, "markdown": render(), "total": total,
                    "pages": len(pages), "progress": len(done),
                    "detect_s": round(det_ms / 1000, 2),
                    "recognise_s": round(time.time() - t0, 2),
                    "elapsed_s": round(det_ms / 1000 + time.time() - t0, 2),
                    "done": False,
                }

        print(
            f"[PAGE] pages={len(pages)} blocks={len(done)} "
            f"detect_ms={det_ms} recognise_s={time.time() - t0:.1f}"
        )
        recognise_s = time.time() - t0
        truncated = [b for b in done if b.get("truncated")]
        if truncated:
            kinds = ", ".join(sorted({b["type"] for b in truncated}))
            print(f"[TRUNCATED] {len(truncated)}/{len(done)} blocks hit the cap ({kinds})")
        yield {
            "blocks": done, "markdown": render(), "total": total,
            "pages": len(pages), "progress": len(done),
            "detect_s": round(det_ms / 1000, 2),
            "recognise_s": round(recognise_s, 2),
            "elapsed_s": round(det_ms / 1000 + recognise_s, 2),
            "truncated_blocks": len(truncated),
            "done": True,
        }

    @modal.method()
    def recognize(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        return self._recognize(image_url, image_base64, task, max_new_tokens)

    # Minimum gap between emitted events. A large table runs to 600+ tokens and
    # forwarding each one would put 600 chunks on the wire for output a reader
    # cannot follow that fast; 80 ms still looks continuous.
    STREAM_COALESCE_S = 0.08

    def _recognize_stream(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
    ):
        """Yields partial results as llama-server produces them.

        This is the one place SSE genuinely lives in this app: llama-server's
        OpenAI-compatible endpoint answers `"stream": true` with `data: {...}`
        chunks. We consume that and re-emit as NDJSON, because the client talks
        to us over POST with a megabyte of base64 in the body and EventSource
        cannot do POST.

        /parse-page deliberately does NOT use this: it runs up to 64 blocks
        concurrently and assembles them in reading order, so partial blocks
        could not be placed in the document anyway.
        """
        import json as _json

        import requests

        if task not in PROMPTS:
            raise ValueError(f"Unknown task {task!r}. Expected one of: {', '.join(PROMPTS)}")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(image_url, image_base64)
                            },
                        },
                        {"type": "text", "text": PROMPTS[task]},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": True,
            # Without this the usage block never arrives and tokens/s has to be
            # inferred from the delta count, which misses the prompt side.
            "stream_options": {"include_usage": True},
        }

        t0 = time.time()
        parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason = None
        last_emit = 0.0

        with requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload, stream=True, timeout=900,
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = _json.loads(body)
                except ValueError:
                    continue  # keep-alive or a partial frame
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                now = time.time()
                if parts and now - last_emit >= self.STREAM_COALESCE_S:
                    last_emit = now
                    yield {"text": "".join(parts), "task": task, "done": False}

        elapsed = time.time() - t0
        text = "".join(parts).strip()
        completion_tokens = usage.get("completion_tokens")
        tok_s = (completion_tokens / elapsed) if (completion_tokens and elapsed) else None

        print(
            f"[GEN] task={task} tokens={completion_tokens} "
            f"elapsed_s={elapsed:.2f} tok_s={tok_s:.1f}" if tok_s else
            f"[GEN] task={task} elapsed_s={elapsed:.2f}"
        )
        yield {
            "text": text,
            "task": task,
            "elapsed_s": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "tokens_per_s": round(tok_s, 1) if tok_s else None,
            "truncated": finish_reason == "length",
            "done": True,
        }

    def _recognize(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Drain the stream to a single dict, for `modal run` and .remote()."""
        last: dict[str, Any] = {}
        for event in self._recognize_stream(
            image_url, image_base64, task, max_new_tokens
        ):
            last = event
        return last


@app.local_entrypoint()
def main(image_url: str, task: str = "ocr", max_new_tokens: int = 512):
    res = LlamaServer().recognize.remote(
        image_url=image_url, task=task, max_new_tokens=max_new_tokens
    )
    print(f"{res['completion_tokens']} tokens in {res['elapsed_s']}s "
          f"({res['tokens_per_s']} tok/s)")
    print(res["text"][:2000])
