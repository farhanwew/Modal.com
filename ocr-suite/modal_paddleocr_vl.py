from __future__ import annotations

import base64
import io
import os
from typing import Any, Iterator

import modal


APP_NAME = "paddleocr-vl"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
CACHE_DIR = "/cache/huggingface"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("paddleocr-vl-cache", create_if_missing=True)

# FlashAttention-2 was tried here and removed on purpose. transformers validates
# FA2 at load time and refuses on CPU, but the memory-snapshot phase *requires*
# a CPU load; switching after .to("cuda") is rejected by the same check. Forcing
# it would mean dropping memory snapshots and paying a full model load (~60s) on
# every cold start instead of ~1.4s, to chase a gain smaller than the run-to-run
# variance already seen on sdpa (26-36 tok/s) — decode here is memory-bandwidth
# bound, not attention bound.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        # The model card is explicit: "ensure the transformers v5 is installed".
        "transformers>=5.0.0",
        "accelerate",
        "Pillow",
        "requests",
        "fastapi[standard]",
        "huggingface_hub[hf_xet]",
        # Layout detection for /parse-page. This is a fork of ultralytics, not
        # ultralytics itself — the DocStructBench weights only load through it.
        "doclayout-yolo==0.0.3",
        "pymupdf",  # rasterises PDF pages for /parse-page
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)

# Hoisted so these land in the memory snapshot rather than being re-imported on
# every cold start.
with image.imports():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


# Transcribed from the model card's inference example.
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

# ---- Layout detection (/parse-page) ----------------------------------------
LAYOUT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_WEIGHTS = "doclayout_yolo_docstructbench_imgsz1024.pt"

# DocStructBench class ids, and which recognition task each one should be sent
# to. `abandon` is headers/footers/page numbers, and `figure` has no text worth
# transcribing — both are dropped rather than fed to the VLM.
LAYOUT_CLASSES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}
LAYOUT_TASK = {
    "title": "ocr",
    "plain text": "ocr",
    "figure_caption": "ocr",
    "table": "table",
    "table_caption": "ocr",
    "table_footnote": "ocr",
    "isolate_formula": "formula",
    "formula_caption": "ocr",
}
SKIP_CLASSES = {"abandon", "figure"}
MAX_BLOCKS = 60  # guard against a pathological detection flooding the GPU

# Spotting runs at a higher pixel budget and upscales small inputs; both numbers
# come straight from the model card.
SPOTTING_MAX_PIXELS = 2048 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
SPOTTING_UPSCALE_THRESHOLD = 1500

# The model card reads this off `processor.image_processor.min_pixels`, but that
# attribute doesn't exist on PaddleOCRVLImageProcessor (verified: it raises
# AttributeError at runtime). This is the value the model's own
# preprocessor_config.json ships, used as a fallback.
MIN_PIXELS = 112896


def _download_bytes(url: str) -> bytes:
    import requests

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    return response.content


def _load_image(image_url: str | None, image_base64: str | None):
    from PIL import Image

    if image_url:
        data = _download_bytes(image_url)
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
    else:
        raise ValueError("Provide image_url or image_base64.")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pdf_to_pages(pdf_bytes: bytes, dpi: int = 200) -> list:
    """Rasterise every PDF page to a PIL image."""
    import io as _io

    import fitz
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(Image.open(_io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return pages


def _resolve_pages(
    image_url: str | None,
    image_base64: str | None,
    pdf_url: str | None,
    pdf_base64: str | None,
    pdf_dpi: int,
) -> list:
    if pdf_url or pdf_base64:
        data = _download_bytes(pdf_url) if pdf_url else base64.b64decode(pdf_base64)
        return _pdf_to_pages(data, dpi=pdf_dpi)
    return [_load_image(image_url, image_base64)]


def _reading_order(blocks: list[dict], page_w: int) -> list[dict]:
    """Order detected blocks the way a human reads them.

    Blocks spanning most of the page width (titles, full-width figures) act as
    horizontal separators; between two of them, narrower blocks are read as
    left column top-to-bottom, then right column. That covers the common
    "full-width heading over two columns" layout.

    This is a heuristic, not a real XY-cut: three-or-more-column layouts,
    floating side notes and wrapped text around figures will come out in the
    wrong order.
    """
    full_width_ratio = 0.55
    mid = page_w / 2

    def by_column(band: list[dict]) -> list[dict]:
        left = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
        right = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
        return sorted(left, key=lambda b: b["bbox"][1]) + sorted(
            right, key=lambda b: b["bbox"][1]
        )

    wide = sorted(
        [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) >= full_width_ratio * page_w],
        key=lambda b: b["bbox"][1],
    )
    narrow = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) < full_width_ratio * page_w]

    ordered: list[dict] = []
    prev_y = float("-inf")
    for wb in wide:
        band = [b for b in narrow if prev_y <= b["bbox"][1] < wb["bbox"][1]]
        ordered.extend(by_column(band))
        ordered.append(wb)
        prev_y = wb["bbox"][1]
    ordered.extend(by_column([b for b in narrow if b["bbox"][1] >= prev_y]))
    return ordered


# Duplicated from gradio_app.py on purpose: that file is a standalone PEP 723
# script and must not import from this one. Keep the two in sync.
def _otsl_to_html(otsl: str) -> str:
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


def _as_markdown(kind: str, text: str) -> str:
    if not text:
        return ""
    if kind == "title":
        return f"## {text}"
    if kind == "table":
        return _otsl_to_html(text) or text
    if kind == "isolate_formula":
        return f"$$\n{text}\n$$"
    return text


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=900,
    scaledown_window=120,
    volumes={CACHE_DIR: hf_cache},
    enable_memory_snapshot=True,
)
# Unlike unlimited-ocr, this app streams via TextIteratorStreamer rather than by
# redirecting sys.stdout, so concurrent generations in one container can't clobber
# each other's output and it's safe to pack a couple of requests per container.
@modal.concurrent(max_inputs=2)
class PaddleOCRVLServer:
    # snap=True runs once on CPU and its memory image is reused by later cold
    # starts; snap=False reruns each time. Nothing here may touch CUDA, or the
    # snapshot becomes unrestorable.
    @modal.enter(snap=True)
    def load_to_cpu(self):
        from concurrent.futures import ThreadPoolExecutor

        def load_processor():
            return AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        def load_model():
            # Loaded plainly (no device_map/.to("cuda")) so it stays on CPU for
            # the snapshot, mirroring the model card's load-then-.to(DEVICE) split.
            #
            # Must be sdpa here, not flash_attention_2: transformers validates
            # FA2 at load time and rejects it on CPU ("FlashAttention2 is not
            # available on CPU"), while the snapshot phase requires CPU. The
            # upgrade to FA2 therefore happens in move_to_gpu(), once the weights
            # are actually on the device.
            return AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR,
                attn_implementation="sdpa",
            )

        def load_layout():
            # Weights load onto CPU here; the device is chosen per-call at
            # predict() time, so this stays snapshot-safe.
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=LAYOUT_REPO, filename=LAYOUT_WEIGHTS, cache_dir=CACHE_DIR
            )
            return YOLOv10(path)

        with ThreadPoolExecutor(max_workers=3) as executor:
            f_processor = executor.submit(load_processor)
            f_model = executor.submit(load_model)
            f_layout = executor.submit(load_layout)
            self.processor = f_processor.result()
            self.model = f_model.result()
            try:
                self.layout = f_layout.result()
            except Exception as e:
                # /recognize must keep working even if layout detection can't load;
                # only /parse-page depends on it.
                print(f"[INIT] layout detector unavailable: {e}")
                self.layout = None

        self.model.eval()

    @modal.enter(snap=False)
    def move_to_gpu(self):
        import time

        t0 = time.time()
        self.model = self.model.to("cuda")
        print(f"[COLD_START] move_to_gpu elapsed_ms={int((time.time() - t0) * 1000)}")

        print(
            f"[DEVICE] cuda_available={torch.cuda.is_available()} "
            f"name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} "
            f"model_device={next(self.model.parameters()).device} "
            f"dtype={next(self.model.parameters()).dtype} "
            f"attn={getattr(self.model.config, '_attn_implementation', None)}"
        )

    def _prepare(self, pil_image, task: str, max_pixels: int | None):
        """Turn a PIL image + task into model inputs. Shared by /recognize and
        by the per-region calls /parse-page makes, so the two transformers-v5
        workarounds below only exist in one place."""
        from PIL import Image

        # Spotting only: the model card upscales small images 2x before inference.
        if task == "spotting":
            w, h = pil_image.size
            if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
                pil_image = pil_image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

        # This is the single biggest lever on prefill cost: the image is cut into
        # 28x28-pixel tokens, so max_pixels/(28*28) is literally how many image
        # tokens the model must attend over. The model-card defaults below work
        # out to 1280 tokens (2048 for spotting); halving max_pixels halves that,
        # at the cost of resolving small text less reliably.
        if max_pixels is None:
            max_pixels = SPOTTING_MAX_PIXELS if task == "spotting" else DEFAULT_MAX_PIXELS

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": PROMPTS[task]},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers v5 requires processor kwargs to be nested under
            # `processor_kwargs`; passing `images_kwargs=` at the top level (as
            # the model card does) is silently ignored, so the pixel budget —
            # including spotting's larger one — never takes effect.
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": getattr(
                            self.processor.image_processor, "min_pixels", MIN_PIXELS
                        ),
                        "longest_edge": max_pixels,
                    }
                }
            },
        ).to(self.model.device)
        return inputs, max_pixels // (28 * 28)

    def _generate_text(self, pil_image, task: str, max_new_tokens: int, max_pixels=None) -> str:
        """Blocking single-shot generation, used for each detected region."""
        inputs, _ = self._prepare(pil_image, task, max_pixels)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, use_cache=True
            )
        # The trailing token is EOS; the model card's example trims it the same way.
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1] : -1]).strip()

    @modal.method()
    def recognize_stream(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
        max_pixels: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        from threading import Thread

        from PIL import Image

        if task not in PROMPTS:
            raise ValueError(f"Unknown task {task!r}. Expected one of: {', '.join(PROMPTS)}")

        pil_image = _load_image(image_url, image_base64)

        # Spotting only: the model card upscales small images 2x before inference.
        if task == "spotting":
            w, h = pil_image.size
            if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
                pil_image = pil_image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

        if max_pixels is None:
            max_pixels = SPOTTING_MAX_PIXELS if task == "spotting" else DEFAULT_MAX_PIXELS
        image_tokens = max_pixels // (28 * 28)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": PROMPTS[task]},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers v5 requires processor kwargs to be nested under
            # `processor_kwargs`; passing `images_kwargs=` at the top level (as
            # the model card does) is silently ignored, so the pixel budget —
            # including spotting's larger one — never takes effect.
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": getattr(
                            self.processor.image_processor, "min_pixels", MIN_PIXELS
                        ),
                        "longest_edge": max_pixels,
                    }
                }
            },
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        errors: list[str] = []

        def _generate():
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        # The model ships `"use_cache": false` in both config.json
                        # and generation_config.json. Left alone, every new token
                        # re-attends over the whole sequence — including ~1k image
                        # tokens — making decoding O(n^2) and ~10x slower than this
                        # 0.9B model should be on an L4. Re-enabling the KV cache is
                        # the single biggest win here.
                        use_cache=True,
                    )
            except Exception as e:  # surfaced after the stream drains
                errors.append(str(e))

        import time

        thread = Thread(target=_generate, daemon=True)
        thread.start()

        t0 = time.time()
        accumulated = ""
        chunks = 0
        for chunk in streamer:
            accumulated += chunk
            chunks += 1
            yield {"text": accumulated, "done": False, "task": task}
        thread.join()

        if errors:
            raise RuntimeError(f"Generation failed: {'; '.join(errors)}")

        elapsed = time.time() - t0
        print(
            f"[GEN] task={task} chunks={chunks} elapsed_s={elapsed:.1f} "
            f"chunks_per_s={chunks / elapsed if elapsed else 0:.1f} "
            f"image_tokens={image_tokens} prompt_tokens={inputs['input_ids'].shape[-1]}"
        )
        yield {"text": accumulated.strip(), "done": True, "task": task}

    def _detect_blocks(self, page, conf: float, iou: float) -> list[dict]:
        import torchvision

        det = self.layout.predict(page, imgsz=1024, conf=conf, device="cuda")[0]
        raw_boxes, raw_cls, raw_conf = det.boxes.xyxy, det.boxes.cls, det.boxes.conf
        keep = torchvision.ops.nms(
            boxes=raw_boxes.float().cpu(), scores=raw_conf.float().cpu(), iou_threshold=iou
        )

        blocks = []
        for i in keep.tolist():
            kind = LAYOUT_CLASSES.get(int(raw_cls[i].item()), "plain text")
            if kind in SKIP_CLASSES:
                continue
            x1, y1, x2, y2 = (int(v) for v in raw_boxes[i].tolist())
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            blocks.append({"type": kind, "bbox": [x1, y1, x2, y2], "score": float(raw_conf[i])})
        return _reading_order(blocks, page.size[0])[:MAX_BLOCKS]

    @modal.method()
    def parse_page_stream(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        pdf_url: str | None = None,
        pdf_base64: str | None = None,
        pdf_dpi: int = 200,
        conf: float = 0.25,
        iou: float = 0.45,
        max_new_tokens: int = 512,
        max_pixels: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Page-level parsing: detect layout blocks, then recognise each one with
        the task that suits its type. Detection and recognition deliberately run
        in the same container — a page can have dozens of blocks, and a network
        hop per block would dominate the cost.

        A PDF is rasterised here and every page goes through the same path."""
        import time

        if self.layout is None:
            raise RuntimeError(
                "Layout detector failed to load; /parse-page is unavailable. See [INIT] logs."
            )

        pages = _resolve_pages(image_url, image_base64, pdf_url, pdf_base64, pdf_dpi)

        t_det = time.time()
        per_page = [self._detect_blocks(p, conf, iou) for p in pages]
        total = sum(len(b) for b in per_page)
        det_ms = int((time.time() - t_det) * 1000)
        print(f"[LAYOUT] pages={len(pages)} blocks={total} detect_ms={det_ms}")

        yield {"blocks": [], "markdown": "", "total": total, "pages": len(pages), "done": False}

        t0 = time.time()
        done_blocks: list[dict] = []
        multi_page = len(pages) > 1

        def render() -> str:
            parts = []
            current_page = None
            for b in done_blocks:
                if multi_page and b["page"] != current_page:
                    current_page = b["page"]
                    parts.append(f"---\n\n**Page {current_page + 1} / {len(pages)}**")
                md = _as_markdown(b["type"], b.get("text", ""))
                if md:
                    parts.append(md)
            return "\n\n".join(parts)

        for page_idx, (page, blocks) in enumerate(zip(pages, per_page)):
            for blk in blocks:
                x1, y1, x2, y2 = blk["bbox"]
                task = LAYOUT_TASK.get(blk["type"], "ocr")
                try:
                    text = self._generate_text(
                        page.crop((x1, y1, x2, y2)), task, max_new_tokens, max_pixels
                    )
                except Exception as e:
                    text = ""
                    blk["error"] = str(e)
                blk.update(page=page_idx, task=task, text=text)
                done_blocks.append(blk)
                yield {
                    "blocks": done_blocks,
                    "markdown": render(),
                    "total": total,
                    "pages": len(pages),
                    "progress": len(done_blocks),
                    "done": False,
                }

        print(
            f"[PAGE] pages={len(pages)} blocks={len(done_blocks)} "
            f"detect_ms={det_ms} recognise_s={time.time() - t0:.1f}"
        )
        yield {
            "blocks": done_blocks,
            "markdown": render(),
            "total": total,
            "pages": len(pages),
            "progress": len(done_blocks),
            "done": True,
        }


# One ASGI app => one deployed base URL, path-routed. This container stays
# CPU-only/cheap and dispatches to the GPU class via .remote_gen().
@app.function(image=image, timeout=900)
@modal.asgi_app()
def web_app():
    import json

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    api = FastAPI(title=APP_NAME)

    @api.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID, "gpu": GPU_TYPE, "tasks": list(PROMPTS)}

    @api.post("/recognize")
    def recognize(payload: dict[str, Any]):
        task = str(payload.get("task", "ocr"))
        if task not in PROMPTS:
            raise HTTPException(400, detail=f"Unknown task {task!r}. Expected one of: {', '.join(PROMPTS)}")
        if not (payload.get("image_url") or payload.get("image_base64")):
            raise HTTPException(400, detail="Provide image_url or image_base64.")

        kwargs = dict(
            image_url=payload.get("image_url"),
            image_base64=payload.get("image_base64"),
            task=task,
            max_new_tokens=int(payload.get("max_new_tokens", 1024)),
            max_pixels=(
                int(payload["max_pixels"]) if payload.get("max_pixels") is not None else None
            ),
        )

        def event_stream():
            for chunk in PaddleOCRVLServer().recognize_stream.remote_gen(**kwargs):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @api.post("/parse-page")
    def parse_page(payload: dict[str, Any]):
        if not any(
            payload.get(k) for k in ("image_url", "image_base64", "pdf_url", "pdf_base64")
        ):
            raise HTTPException(
                400, detail="Provide one of image_url, image_base64, pdf_url or pdf_base64."
            )

        kwargs = dict(
            image_url=payload.get("image_url"),
            image_base64=payload.get("image_base64"),
            pdf_url=payload.get("pdf_url"),
            pdf_base64=payload.get("pdf_base64"),
            pdf_dpi=int(payload.get("pdf_dpi", 200)),
            conf=float(payload.get("conf", 0.25)),
            iou=float(payload.get("iou", 0.45)),
            max_new_tokens=int(payload.get("max_new_tokens", 512)),
            max_pixels=(
                int(payload["max_pixels"]) if payload.get("max_pixels") is not None else None
            ),
        )

        def event_stream():
            for chunk in PaddleOCRVLServer().parse_page_stream.remote_gen(**kwargs):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return api


@app.local_entrypoint()
def main(image_url: str, task: str = "ocr", max_new_tokens: int = 1024):
    text = ""
    for chunk in PaddleOCRVLServer().recognize_stream.remote_gen(
        image_url=image_url, task=task, max_new_tokens=max_new_tokens
    ):
        text = chunk["text"]
    print(text)
