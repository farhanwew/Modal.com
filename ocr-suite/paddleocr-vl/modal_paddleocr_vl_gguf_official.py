from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "paddleocr-vl-gguf-official"
REPO_ID = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"
MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
MODEL_DIR = "/models"
PORT = 8080
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
PARALLEL = int(os.environ.get("LLAMA_PARALLEL", "32"))
CTX_TOTAL = int(os.environ.get("LLAMA_CTX", str(6144 * PARALLEL)))

app = modal.App(APP_NAME)
models = modal.Volume.from_name("paddleocr-vl-gguf-models", create_if_missing=True)
paddlex_cache = modal.Volume.from_name(
    "paddleocr-vl-official-paddlex-cache", create_if_missing=True
)


def _fetch_weights() -> None:
    from huggingface_hub import hf_hub_download

    for filename in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)


# The official PaddleOCR pipeline supplies layout analysis and page assembly;
# llama-server remains the VLM backend for GGUF recognition.
image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --no-cache-dir paddlepaddle==3.2.1 "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
    )
    .uv_pip_install(
        "huggingface_hub[hf_xet]",
        "requests",
        "paddleocr[doc-parser]>=3.6.0",
        "fastapi[standard]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_fetch_weights, volumes={MODEL_DIR: models}, timeout=60 * 30)
)


def _source_from_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    if payload.get("pdf_url"):
        return str(payload["pdf_url"]), None
    if payload.get("image_url"):
        return str(payload["image_url"]), None
    if payload.get("pdf_base64"):
        return str(payload["pdf_base64"]), ".pdf"
    if payload.get("image_base64"):
        return str(payload["image_base64"]), ".png"
    raise ValueError("Provide image_url, image_base64, pdf_url or pdf_base64.")


def _result_to_dict(result: Any) -> dict[str, Any]:
    # PaddleOCR results can contain PIL objects in their structured payload;
    # normalize them before Modal serializes the method return value.
    data = json.loads(json.dumps(result.json, default=str))
    raw_markdown = result.markdown
    markdown = json.loads(json.dumps(raw_markdown, default=str))
    images = {}
    for name, image in (raw_markdown.get("markdown_images") or {}).items():
        image = image.get("img") if isinstance(image, dict) else image
        if image is None:
            continue
        if not hasattr(image, "save"):
            from PIL import Image

            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        images[Path(name).name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "json": data,
        "markdown": markdown.get("markdown_texts", "") if isinstance(markdown, dict) else str(markdown),
        "official_markdown": markdown,
        "images": images,
    }


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=1800,
    scaledown_window=120,
    volumes={MODEL_DIR: models, "/root/.paddlex": paddlex_cache},
)
@modal.concurrent(max_inputs=2)
class OfficialPaddleOCRVL:
    @modal.enter()
    async def start(self) -> None:
        from paddleocr import PaddleOCRVL

        binary = "/app/llama-server"
        self.process = subprocess.Popen(
            [
                binary,
                "-m", f"{MODEL_DIR}/{MODEL_FILE}",
                "--mmproj", f"{MODEL_DIR}/{MMPROJ_FILE}",
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "-ngl", "999",
                "-c", str(CTX_TOTAL),
                "--parallel", str(PARALLEL),
                "--temp", "0",
            ]
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}")
            try:
                import requests

                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy in 600s")

        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            # Paddle handles layout on CPU; llama-server keeps VLM inference on GPU.
            device="cpu",
            vl_rec_backend="llama-cpp-server",
            vl_rec_server_url=f"http://127.0.0.1:{PORT}/v1",
            vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
            vl_rec_max_concurrency=PARALLEL,
        )
        await paddlex_cache.commit.aio()

    @modal.exit()
    def stop(self) -> None:
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)

    @staticmethod
    def _prepare_source(source: str, suffix: str | None) -> tuple[str, str | None]:
        if suffix is None:
            return source, None
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            handle.write(base64.b64decode(source.split(",", 1)[-1]))
            return handle.name, handle.name
        finally:
            handle.close()

    @modal.method()
    def parse(
        self,
        source: str,
        source_suffix: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        layout_threshold: float = 0.5,
        layout_unclip_ratio: float = 1.05,
        layout_merge_bboxes_mode: str = "large",
    ) -> dict[str, Any]:
        return self._parse(
            source,
            source_suffix,
            max_new_tokens,
            temperature,
            layout_threshold,
            layout_unclip_ratio,
            layout_merge_bboxes_mode,
        )

    def _parse(
        self,
        source: str,
        source_suffix: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        layout_threshold: float = 0.5,
        layout_unclip_ratio: float = 1.05,
        layout_merge_bboxes_mode: str = "large",
    ) -> dict[str, Any]:
        source, temporary = self._prepare_source(source, source_suffix)
        try:
            pages = list(
                self.pipeline.predict(
                    input=source,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    layout_threshold=layout_threshold,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                    use_queues=True,
                )
            )
            results = list(
                self.pipeline.restructure_pages(
                    pages,
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                )
            )
            return {
                "pages": len(pages),
                "results": [_result_to_dict(result) for result in results],
            }
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)

    @modal.method()
    def recognize(
        self,
        source: str,
        source_suffix: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        return self._recognize(source, source_suffix, task, max_new_tokens)

    def _recognize(
        self,
        source: str,
        source_suffix: str | None = None,
        task: str = "ocr",
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        source, temporary = self._prepare_source(source, source_suffix)
        try:
            result = next(
                iter(
                    self.pipeline.predict(
                        input=source,
                        use_layout_detection=False,
                        prompt_label=task,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )
                )
            )
            return _result_to_dict(result)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse

        api = FastAPI(title=APP_NAME)

        @api.get("/health")
        def health():
            return {"ok": True, "backend": "official-paddleocr-vl", "gpu": GPU_TYPE}

        @api.post("/parse-page")
        def parse_page(payload: dict[str, Any]):
            try:
                source, suffix = _source_from_payload(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events():
                yield json.dumps({"done": False, "pages": 0, "total": 0, "progress": 0}) + "\n"
                result = self._parse(
                    source=source,
                    source_suffix=suffix,
                    max_new_tokens=int(payload.get("max_new_tokens", 4096)),
                    temperature=float(payload.get("temperature", 0.0)),
                    layout_threshold=float(payload.get("layout_threshold", 0.5)),
                    layout_unclip_ratio=float(payload.get("layout_unclip_ratio", 1.05)),
                    layout_merge_bboxes_mode=str(payload.get("layout_merge_bboxes_mode", "large")),
                )
                final = result["results"][-1] if result["results"] else {}
                images = {
                    name: data
                    for item in result["results"]
                    for name, data in item.get("images", {}).items()
                }
                official_results = [
                    {key: value for key, value in item.items() if key != "images"}
                    for item in result["results"]
                ]
                yield json.dumps(
                    {
                        "done": True,
                        "pages": result["pages"],
                        "total": result["pages"],
                        "progress": result["pages"],
                        "markdown": final.get("markdown", ""),
                        "official_results": official_results,
                        "new_images": images,
                    },
                    default=str,
                ) + "\n"
            return StreamingResponse(events(), media_type="application/x-ndjson")

        @api.post("/recognize")
        def recognize(payload: dict[str, Any]):
            if payload.get("pdf_url") or payload.get("pdf_base64"):
                raise HTTPException(400, detail="PDF input is not supported by /recognize.")
            try:
                source, suffix = _source_from_payload(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc
            return self._recognize(
                source=source,
                source_suffix=suffix,
                task=str(payload.get("task", "ocr")),
                max_new_tokens=int(payload.get("max_new_tokens", 1024)),
            )

        return api


@app.local_entrypoint()
def main(image_url: str, max_new_tokens: int = 4096):
    source, suffix = _source_from_payload({"image_url": image_url})
    result = OfficialPaddleOCRVL().parse.remote(
        source, source_suffix=suffix, max_new_tokens=max_new_tokens
    )
    print(result["results"][-1]["markdown"] if result["results"] else "")
