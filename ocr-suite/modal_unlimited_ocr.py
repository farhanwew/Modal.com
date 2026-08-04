from __future__ import annotations

import base64
import os
import queue
import sys
import threading
from typing import Any, Iterator

import modal


APP_NAME = "unlimited-ocr"
MODEL_ID = "baidu/Unlimited-OCR"
CACHE_DIR = "/cache/huggingface"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("unlimited-ocr-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==4.57.1",
        "Pillow==12.1.1",
        "matplotlib==3.10.8",
        "einops==0.8.2",
        "addict==2.4.0",
        "easydict==1.13",
        "pymupdf==1.27.2.2",
        "psutil==7.2.2",
        "huggingface_hub[hf_xet]",
        "fastapi[standard]",
        "requests",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)

# Hoisted so these land in the memory snapshot instead of being re-imported on
# every cold start (importing torch + transformers alone costs several seconds).
with image.imports():
    import torch
    from transformers import AutoModel, AutoTokenizer


DEFAULT_SINGLE_PROMPT = "document parsing."


def _download_bytes(url: str) -> bytes:
    import requests

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    return response.content


def _pdf_to_page_pngs(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def _resolve_pages(
    out_dir: str,
    image_url: str | None,
    image_base64: str | None,
    image_urls: list[str] | None,
    images_base64: list[str] | None,
    pdf_url: str | None,
    pdf_base64: str | None,
    pdf_dpi: int,
) -> list[str]:
    page_bytes_list: list[bytes] = []
    if pdf_url or pdf_base64:
        pdf_bytes = _download_bytes(pdf_url) if pdf_url else base64.b64decode(pdf_base64)
        page_bytes_list = _pdf_to_page_pngs(pdf_bytes, dpi=pdf_dpi)
    elif image_urls or images_base64:
        for u in image_urls or []:
            page_bytes_list.append(_download_bytes(u))
        for b in images_base64 or []:
            page_bytes_list.append(base64.b64decode(b))
    elif image_url or image_base64:
        page_bytes_list = [
            _download_bytes(image_url) if image_url else base64.b64decode(image_base64)
        ]
    else:
        raise ValueError(
            "Provide one of image_url/image_base64, image_urls/images_base64, or pdf_url/pdf_base64."
        )
    if not page_bytes_list:
        raise ValueError("No pages resolved from the given input.")

    paths = []
    for i, data in enumerate(page_bytes_list):
        path = os.path.join(out_dir, f"page_{i:04d}.png")
        with open(path, "wb") as f:
            f.write(data)
        paths.append(path)
    return paths


def _read_result_md(out_dir: str) -> str:
    """save_results=True writes the *cleaned* markdown here (ref/det tags
    stripped, image crops turned into `![](images/N.jpg)` links) — this is
    what eval_mode=True's raw decoded string does NOT give you."""
    result = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith((".md", ".txt")):
            with open(os.path.join(out_dir, fname), encoding="utf-8") as f:
                result += f.read() + "\n"
    return result.strip()


class _ThreadTargetedStdout:
    """baidu/Unlimited-OCR's infer()/infer_multi() stream generated tokens by
    print()-ing them from inside model.generate() (see TPSTextStreamer in the
    model repo's modeling_unlimitedocr.py) instead of yielding/returning them.
    This redirects only `target_thread`'s writes into a queue so the request
    handler can turn those prints into streamed response chunks. Only safe
    because UnlimitedOCRServer does not use @modal.concurrent, so a container
    runs one generation at a time and stdout swaps stay properly nested."""

    def __init__(self, target_thread: threading.Thread, q: "queue.Queue[str]", original_stdout):
        self.target_thread = target_thread
        self.q = q
        self.original_stdout = original_stdout

    def write(self, data: str) -> int:
        self.original_stdout.write(data)
        if data and threading.current_thread() is self.target_thread:
            lower_data = data.lower()
            if "tps:" not in lower_data and "tokens/s" not in lower_data:
                self.q.put(data)
        return len(data)

    def flush(self) -> None:
        self.original_stdout.flush()

    def __getattr__(self, name):
        return getattr(self.original_stdout, name)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32 * 1024,
    timeout=1800,
    scaledown_window=120,
    volumes={CACHE_DIR: hf_cache},
    enable_memory_snapshot=True,
)
class UnlimitedOCRServer:
    # Cold start is split in two so Modal can snapshot the expensive half.
    # snap=True runs once, on CPU only, and its resulting memory image is reused
    # by every later cold start; snap=False is the only part that reruns.
    # Nothing here may touch CUDA — initializing it during the snap phase makes
    # the snapshot unrestorable.
    @modal.enter(snap=True)
    def load_to_cpu(self):
        from concurrent.futures import ThreadPoolExecutor

        def load_tokenizer():
            return AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            )

        def load_model():
            # No device_map/.cuda() — this deliberately stays on CPU, mirroring
            # the model card's own load-then-.cuda() two-step.
            return AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_tokenizer = executor.submit(load_tokenizer)
            f_model = executor.submit(load_model)
            self.tokenizer = f_tokenizer.result()
            self.model = f_model.result()

        self.model.eval()

    @modal.enter(snap=False)
    def move_to_gpu(self):
        import time

        t0 = time.time()
        self.model = self.model.cuda()
        print(f"[COLD_START] move_to_gpu elapsed_ms={int((time.time() - t0) * 1000)}")

    @modal.method()
    def parse_stream(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_urls: list[str] | None = None,
        images_base64: list[str] | None = None,
        pdf_url: str | None = None,
        pdf_base64: str | None = None,
        pdf_dpi: int = 200,
        prompt: str | None = None,
        mode: str = "gundam",
        max_length: int = 8192,
        temperature: float = 0.0,
        no_repeat_ngram_size: int = 35,
        ngram_window: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        import shutil
        import tempfile
        from threading import Thread

        out_dir = tempfile.mkdtemp(prefix="ocr-")
        try:
            paths = _resolve_pages(
                out_dir, image_url, image_base64, image_urls, images_base64,
                pdf_url, pdf_base64, pdf_dpi,
            )

            errors: list[str] = []
            returned: dict[str, Any] = {}
            single = len(paths) == 1

            if mode == "base":
                base_size, image_size, crop_mode = 1024, 1024, False
            else:
                base_size, image_size, crop_mode = 1024, 640, True
            window = ngram_window if ngram_window is not None else 128
            effective_prompt = f"<image>{prompt or DEFAULT_SINGLE_PROMPT}"

            # Multi-page/PDF input is parsed one page at a time via infer(), the
            # same way the official baidu/Unlimited-OCR Space's app.py does it —
            # infer_multi() exists in the model repo but that reference demo
            # never calls it, so this sticks to the code path that's actually
            # been exercised/tuned.
            def _run():
                try:
                    texts = []
                    for idx, page_path in enumerate(paths, start=1):
                        self.model.infer(
                            self.tokenizer,
                            prompt=effective_prompt,
                            image_file=page_path,
                            output_path=out_dir,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=crop_mode,
                            max_length=max_length,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            ngram_window=window,
                            temperature=temperature,
                            save_results=True,
                        )
                        page_text = _read_result_md(out_dir)
                        texts.append(
                            page_text if single else f"── PAGE {idx} / {len(paths)} ──\n{page_text}"
                        )
                    returned["text"] = "\n\n".join(texts).strip()
                except Exception as e:
                    errors.append(str(e))

            q: "queue.Queue[str]" = queue.Queue()
            thread = Thread(target=_run, daemon=True)
            original_stdout = sys.stdout
            sys.stdout = _ThreadTargetedStdout(thread, q, original_stdout)

            accumulated = ""
            try:
                thread.start()
                while thread.is_alive() or not q.empty():
                    try:
                        chunk = q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    accumulated += chunk
                    yield {"text": accumulated, "done": False, "pages": len(paths)}
            finally:
                sys.stdout = original_stdout
                thread.join()

            if errors:
                raise RuntimeError(f"Inference failed: {'; '.join(errors)}")

            final_text = returned.get("text") or _read_result_md(out_dir) or accumulated.strip()
            yield {"text": final_text, "done": True, "pages": len(paths)}
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# A single ASGI app (one deployed URL, path-routed) instead of one
# @modal.fastapi_endpoint per function, so a client only needs one base URL.
# This function itself stays CPU-only/cheap — /parse dispatches to the GPU
# UnlimitedOCRServer class via .remote_gen(), /explode-pdf does its CPU work
# (PyMuPDF rasterization) directly, without ever spinning up a GPU container.
@app.function(image=image, timeout=1800)
@modal.asgi_app()
def web_app():
    import json

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    api = FastAPI(title=APP_NAME)

    @api.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID, "gpu": GPU_TYPE}

    @api.post("/parse")
    def parse(payload: dict[str, Any]):
        kwargs = dict(
            image_url=payload.get("image_url"),
            image_base64=payload.get("image_base64"),
            image_urls=payload.get("image_urls"),
            images_base64=payload.get("images_base64"),
            pdf_url=payload.get("pdf_url"),
            pdf_base64=payload.get("pdf_base64"),
            pdf_dpi=int(payload.get("pdf_dpi", 200)),
            prompt=payload.get("prompt"),
            mode=str(payload.get("mode", "gundam")),
            max_length=int(payload.get("max_length", 8192)),
            temperature=float(payload.get("temperature", 0.0)),
            no_repeat_ngram_size=int(payload.get("no_repeat_ngram_size", 35)),
            ngram_window=int(payload["ngram_window"]) if payload.get("ngram_window") is not None else None,
        )

        def event_stream():
            for chunk in UnlimitedOCRServer().parse_stream.remote_gen(**kwargs):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @api.post("/explode-pdf")
    def explode_pdf_route(payload: dict[str, Any]):
        """CPU-only: split a PDF into per-page PNGs so a caller can drive OCR
        one page at a time (progress reporting, cancellation) without
        spinning up the GPU just to rasterize pages."""
        pdf_url = payload.get("pdf_url")
        pdf_base64 = payload.get("pdf_base64")
        if not (pdf_url or pdf_base64):
            raise HTTPException(400, detail="Provide pdf_url or pdf_base64.")

        pdf_bytes = _download_bytes(pdf_url) if pdf_url else base64.b64decode(pdf_base64)
        pages = _pdf_to_page_pngs(pdf_bytes, dpi=int(payload.get("pdf_dpi", 200)))
        return {"pages": [base64.b64encode(p).decode("ascii") for p in pages]}

    return api


@app.local_entrypoint()
def main(
    image_url: str,
    prompt: str | None = None,
    mode: str = "gundam",
):
    last_text = ""
    for chunk in UnlimitedOCRServer().parse_stream.remote_gen(
        image_url=image_url, prompt=prompt, mode=mode
    ):
        last_text = chunk["text"]
        if chunk["done"]:
            print("\n--- final ---")
            print(last_text)
        else:
            print(last_text[-200:], end="\r")
