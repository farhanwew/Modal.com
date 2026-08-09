from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Iterator

import modal


QUANT = os.environ.get("MINERU_GGUF_QUANT", "Q8_0")
if QUANT not in {"Q8_0", "Q4_K_M"}:
    raise ValueError("MINERU_GGUF_QUANT must be Q8_0 or Q4_K_M")
APP_NAME = f"mineru-vl-gguf-{QUANT.lower().replace('_', '-')}"
REPO_ID = "mradermacher/MinerU2.5-Pro-2604-1.2B-GGUF"
MODEL_FILE = f"MinerU2.5-Pro-2604-1.2B.{QUANT}.gguf"
MMPROJ_FILE = "MinerU2.5-Pro-2604-1.2B.mmproj-Q8_0.gguf"
MODEL_DIR = "/models"
PORT = 8080
PARALLEL = 4
PDF_BATCH_SIZE = 2
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

app = modal.App(APP_NAME)
models = modal.Volume.from_name("mineru-vl-gguf-models", create_if_missing=True)


def _fetch_weights() -> None:
    from huggingface_hub import hf_hub_download

    for filename in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)


image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "huggingface_hub[hf_xet]",
        "mineru-vl-utils",
        "fastapi[standard]",
        "pymupdf",
        "requests",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "MINERU_GGUF_QUANT": QUANT})
    .run_function(_fetch_weights, volumes={MODEL_DIR: models}, timeout=1800)
)


def _source(payload: dict[str, Any]) -> tuple[str, bool]:
    for key, is_pdf in (
        ("image_url", False),
        ("image_base64", False),
        ("pdf_url", True),
        ("pdf_base64", True),
    ):
        if payload.get(key):
            return str(payload[key]), is_pdf
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


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=1800,
    scaledown_window=120,
    volumes={MODEL_DIR: models},
)
@modal.concurrent(max_inputs=2)
class MinerUGGUF:
    @modal.enter()
    def load(self) -> None:
        import shutil
        import subprocess

        import requests

        self.quant = os.environ["MINERU_GGUF_QUANT"]
        self.model_file = f"MinerU2.5-Pro-2604-1.2B.{self.quant}.gguf"
        binary = shutil.which("llama-server") or "/app/llama-server"
        self.process = subprocess.Popen(
            [
                binary,
                "-m", f"{MODEL_DIR}/{self.model_file}",
                "--mmproj", f"{MODEL_DIR}/{MMPROJ_FILE}",
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "-ngl", "999",
                "-c", str(8192 * PARALLEL),
                "--parallel", str(PARALLEL),
                "--temp", "0",
                "--seed", "42",
                "-fa", "on",
                "--special",
                "--image-min-tokens", "1024",
            ]
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy in 600s")

        from mineru_vl_utils import MinerUClient

        self.client = MinerUClient(
            backend="http-client",
            server_url=f"http://127.0.0.1:{PORT}/",
            image_analysis=False,
            max_concurrency=PARALLEL,
            skip_model_name_checking=True,
            model_name=self.model_file,
            use_tqdm=False,
        )
        parse_layout = self.client.helper.parse_layout_output

        def parse_llama_layout(output: str):
            if "<|box_start|>" not in output:
                import re

                output = re.sub(
                    r"(?m)^(\d{3})\s+(\d{3})\s+(\d{3})\s+(\d{3})(\w+)\s*$",
                    r"<|box_start|>\1 \2 \3 \4<|box_end|>"
                    r"<|ref_start|>\5<|ref_end|><|rotate_up|>",
                    output,
                )
            return parse_layout(output)

        self.client.helper.parse_layout_output = parse_llama_layout

    @staticmethod
    def _read(source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            import requests

            response = requests.get(source, timeout=120)
            response.raise_for_status()
            return response.content
        return base64.b64decode(source.split(",", 1)[-1])

    def _extract_image(self, image) -> list[dict[str, Any]]:
        return [dict(block) for block in self.client.two_step_extract(image)]

    @modal.method()
    def extract(self, source: str) -> list[dict[str, Any]]:
        from PIL import Image

        image = Image.open(io.BytesIO(self._read(source))).convert("RGB")
        return self._extract_image(image)

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
                "model": REPO_ID,
                "quant": self.quant,
                "gpu": GPU_TYPE,
                "parallel": PARALLEL,
            }

        @api.post("/parse")
        def parse(payload: dict[str, Any]):
            try:
                source, is_pdf = _source(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events() -> Iterator[str]:
                from PIL import Image

                yield json.dumps({"text": "", "pages": 0, "done": False}) + "\n"
                if not is_pdf:
                    blocks = self._extract_image(
                        Image.open(io.BytesIO(self._read(source))).convert("RGB")
                    )
                    yield json.dumps({
                        "text": _markdown(blocks), "blocks": blocks,
                        "pages": 1, "done": True,
                    }) + "\n"
                    return

                import pymupdf

                dpi = max(72, min(int(payload.get("pdf_dpi", 200)), 300))
                doc = pymupdf.open(stream=self._read(source), filetype="pdf")
                blocks = []
                try:
                    for start in range(0, len(doc), PDF_BATCH_SIZE):
                        page_images = []
                        for index in range(start, min(start + PDF_BATCH_SIZE, len(doc))):
                            page = doc[index]
                            pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
                            mode = "RGBA" if pix.alpha else "RGB"
                            page_image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                            page_images.append(page_image.convert("RGB") if mode != "RGB" else page_image)

                        # ponytail: two-page chunks cap raster memory; tune only if
                        # profiling shows more parallel pages fit and improve throughput.
                        results = self.client.batch_two_step_extract(page_images)
                        for offset, result in enumerate(results, 1):
                            page_number = start + offset
                            blocks.extend(dict(block, page=page_number) for block in result)
                            yield json.dumps({
                                "text": _markdown(blocks), "blocks": blocks,
                                "pages": page_number, "done": page_number == len(doc),
                            }) + "\n"
                finally:
                    doc.close()

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api


@app.local_entrypoint()
def main(image_url: str = "https://cdn.farhan-wicaksono.me/table%20.png"):
    print(MinerUGGUF().extract.remote(image_url))
