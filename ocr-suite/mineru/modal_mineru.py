from __future__ import annotations

import base64
import io
import os
from typing import Any, Iterator

import modal


APP_NAME = "mineru-vl"
MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

app = modal.App(APP_NAME)


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "mineru-vl-utils[transformers]",
        "fastapi[standard]",
        "pymupdf",
        "requests",
    )
    .run_function(_download_model, timeout=1800)
)


def _source(payload: dict[str, Any]) -> tuple[str, bool]:
    if payload.get("image_url"):
        return str(payload["image_url"]), False
    if payload.get("image_base64"):
        return str(payload["image_base64"]), False
    if payload.get("pdf_url"):
        return str(payload["pdf_url"]), True
    if payload.get("pdf_base64"):
        return str(payload["pdf_base64"]), True
    raise ValueError("Provide image_url, image_base64, pdf_url or pdf_base64.")


def _markdown(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        content = block.get("content")
        if not content:
            continue
        if block.get("type") == "title":
            parts.append(f"## {content}")
        elif block.get("type") == "equation":
            parts.append(f"$$\n{content}\n$$")
        else:
            parts.append(str(content))
    return "\n\n".join(parts)


def _embedded_images(doc) -> dict[str, str]:
    images = {}
    for page_number, page in enumerate(doc, 1):
        for image_number, image in enumerate(page.get_images(full=True), 1):
            xref = image[0]
            extracted = doc.extract_image(xref)
            name = f"page_{page_number:04d}_embedded_{image_number:04d}.{extracted['ext']}"
            images[name] = base64.b64encode(extracted["image"]).decode("ascii")
    return images


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=1800,
    scaledown_window=300,
    enable_memory_snapshot=True,
)
class MinerU:
    @modal.enter(snap=True)
    def load_to_cpu(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from mineru_vl_utils import MinerUClient
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        with ThreadPoolExecutor(max_workers=2) as executor:
            model_future = executor.submit(
                Qwen2VLForConditionalGeneration.from_pretrained,
                MODEL,
                dtype="bfloat16",
                device_map="cpu",
            )
            processor_future = executor.submit(
                AutoProcessor.from_pretrained, MODEL, use_fast=True
            )
            self.model = model_future.result()
            processor = processor_future.result()

        self.model.eval()
        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=processor,
            image_analysis=False,
        )

    @modal.enter(snap=False)
    def move_to_gpu(self) -> None:
        import time

        import torch

        started = time.time()
        self.model.to("cuda", dtype=torch.bfloat16)
        self.device = str(next(self.model.parameters()).device)
        self.dtype = str(next(self.model.parameters()).dtype)
        print(f"[COLD_START] move_to_gpu elapsed_ms={int((time.time() - started) * 1000)}")

    @modal.method()
    def extract(self, source: str) -> list[dict[str, Any]]:
        return self._extract(source)

    def _extract(self, source: str) -> list[dict[str, Any]]:
        from PIL import Image

        raw = self._read(source)
        blocks = self.client.two_step_extract(Image.open(io.BytesIO(raw)).convert("RGB"))
        return [dict(block) for block in blocks]

    @staticmethod
    def _read(source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            import requests

            response = requests.get(source, timeout=120)
            response.raise_for_status()
            return response.content
        return base64.b64decode(source.split(",", 1)[-1])

    @modal.asgi_app()
    def serve(self):
        import json

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse

        api = FastAPI(title=APP_NAME)

        @api.get("/health")
        def health():
            return {
                "ok": True,
                "model": MODEL,
                "gpu": GPU_TYPE,
                "device": self.device,
                "dtype": self.dtype,
            }

        @api.post("/parse")
        def parse(payload: dict[str, Any]):
            try:
                source, is_pdf = _source(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events() -> Iterator[str]:
                yield json.dumps({"text": "", "pages": 0, "done": False}) + "\n"
                if not is_pdf:
                    blocks = self._extract(source)
                    yield json.dumps(
                        {
                            "text": _markdown(blocks),
                            "blocks": blocks,
                            "pages": 1,
                            "done": True,
                        }
                    ) + "\n"
                    return

                import fitz
                from PIL import Image

                dpi = max(72, min(int(payload.get("pdf_dpi", 200)), 300))
                doc = fitz.open(stream=self._read(source), filetype="pdf")
                blocks = []
                embedded_images = _embedded_images(doc)
                try:
                    for page_number, page in enumerate(doc, 1):
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                        page_blocks = [
                            dict(block, page=page_number)
                            for block in self.client.two_step_extract(image)
                        ]
                        blocks.extend(page_blocks)
                        yield json.dumps(
                            {
                                "text": _markdown(blocks),
                                "blocks": blocks,
                                "pages": page_number,
                                "embedded_images": embedded_images if page_number == len(doc) else {},
                                "done": page_number == len(doc),
                            }
                        ) + "\n"
                finally:
                    doc.close()

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api


@app.local_entrypoint()
def main(image_url: str = "https://cdn.farhan-wicaksono.me/table%20.png"):
    blocks = MinerU().extract.remote(image_url)
    print(blocks)
