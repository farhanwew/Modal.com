from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import modal


# Quant baked into APP_NAME (same convention as mineru/modal_mineru_gguf.py) so
# `MODAL_GLM_OCR_GGUF_QUANT=Q8_0 modal deploy` and the Q4_K_M default coexist as
# two separate apps/URLs instead of overwriting each other.
QUANT = os.environ.get("GLM_OCR_GGUF_QUANT", "Q4_K_M")
if QUANT not in {"Q8_0", "Q4_K_M"}:
    raise ValueError("GLM_OCR_GGUF_QUANT must be Q8_0 or Q4_K_M")
APP_NAME = f"glm-ocr-gguf-{QUANT.lower().replace('_', '-')}"
REPO_ID = "mradermacher/GLM-OCR-GGUF"
MODEL_FILE = f"GLM-OCR.{QUANT}.gguf"
# Kept at Q8_0 regardless of the main model's quant: the vision projector is
# small (484 MB) and precision there is cheap next to a Q4 main model.
MMPROJ_FILE = "GLM-OCR.mmproj-Q8_0.gguf"
MODEL_DIR = "/models"
SERVED_MODEL_NAME = "glm-ocr"
PORT = 8080
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
# Ollama's own llama-server backend divides total context by parallel slots
# (n_ctx_seq = total / parallel); at the default 4096 total it produced
# blank output for every region because a crop's image tokens alone filled
# the slot before any text could be generated — see CLAUDE.md's GLM-OCR
# gotcha. 16384/slot was confirmed sufficient there without ballooning KV
# cache, so PARALLEL stays modest (not paddleocr-vl-gguf's 32) to keep
# PARALLEL * 16384 tokens of KV cache inside a 24 GB L4 alongside a <1 GB
# GGUF model.
PARALLEL = int(os.environ.get("LLAMA_PARALLEL", "8"))
CTX_TOTAL = int(os.environ.get("LLAMA_CTX", str(16384 * PARALLEL)))
MAX_WORKERS = os.environ.get("GLM_OCR_MAX_WORKERS", str(PARALLEL))
# /root/.cache/huggingface is already non-empty in the llama.cpp base image,
# and Modal refuses to mount a Volume on a non-empty path — same sidestep
# paddleocr-vl/modal_paddleocr_vl.py uses.
HF_CACHE_DIR = "/cache/huggingface"  # glmocr's own PP-DocLayoutV3 download

app = modal.App(APP_NAME)
models = modal.Volume.from_name("glm-ocr-gguf-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("glm-ocr-gguf-hf-cache", create_if_missing=True)


def _fetch_weights() -> None:
    """Downloads into the Volume at build time, not baked into the image
    layer — `modal deploy` runs two builders concurrently against the same
    image id, and the one that saves first can win with an empty /models
    (see paddleocr-vl/modal_paddleocr_vl_gguf.py). A Volume is immune: it
    lives outside the image, so whichever builder wins, the weights are
    already there."""
    from huggingface_hub import hf_hub_download

    for fname in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=MODEL_DIR)


# llama.cpp's own prebuilt CUDA server image — same base as
# paddleocr-vl/modal_paddleocr_vl_gguf.py, for the same reason: building
# llama.cpp from source with CUDA takes 10-20 minutes per rebuild.
image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12"
    )
    .entrypoint([])  # the base image starts llama-server; we manage it ourselves
    .apt_install("curl")
    .uv_pip_install(
        "glmocr[selfhosted]", "fastapi[standard]", "requests", "huggingface_hub[hf_xet]"
    )
    # HF_HOME is set at runtime (in start()), not baked in here: hf_hub_download
    # writes cache bookkeeping under HF_HOME even with local_dir= set, so baking
    # it would populate /cache/huggingface during this build step and the
    # container's own volume mount there would then hit the same
    # "non-empty path" conflict MODEL_DIR did before local_dir fixed it.
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_fetch_weights, volumes={MODEL_DIR: models}, timeout=60 * 30)
)


def _source(payload: dict[str, Any]) -> tuple[str, str | None]:
    if payload.get("image_url"):
        return str(payload["image_url"]), None
    if payload.get("pdf_url"):
        return str(payload["pdf_url"]), None
    if payload.get("image_base64"):
        return str(payload["image_base64"]), ".png"
    if payload.get("pdf_base64"):
        return str(payload["pdf_base64"]), ".pdf"
    raise ValueError("Provide image_url, image_base64, pdf_url or pdf_base64.")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=1800,
    scaledown_window=120,
    volumes={MODEL_DIR: models, HF_CACHE_DIR: hf_cache},
)
class GLMOCR:
    @modal.enter()
    def start(self) -> None:
        os.environ["HF_HOME"] = HF_CACHE_DIR
        from glmocr import GlmOcr

        binary = shutil.which("llama-server") or "/app/llama-server"
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

        import requests

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

        config = Path("/tmp/glmocr.yaml")
        config.write_text(
            f"""pipeline:
  max_workers: {MAX_WORKERS}
  maas:
    enabled: false
  layout:
    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
    label_task_mapping:
      text: [abstract, algorithm, content, doc_title, figure_title, paragraph_title, reference_content, text, vertical_text, vision_footnote, seal, formula_number]
      table: [table]
      formula: [display_formula, inline_formula]
      skip: [chart, image]
      abandon: [header, footer, number, footnote, aside_text, reference, footer_image, header_image]
  ocr_api:
    api_host: 127.0.0.1
    api_port: {PORT}
    model: {SERVED_MODEL_NAME}
    connect_timeout: 600
    request_timeout: 900
""",
            encoding="utf-8",
        )
        self.parser = GlmOcr(config_path=str(config), layout_device="cpu")
        models.commit()
        hf_cache.commit()

    @modal.exit()
    def stop(self) -> None:
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)

    @modal.method()
    def parse(
        self,
        source: str,
        suffix: str | None = None,
    ) -> str:
        return self._parse(source, suffix)

    def _parse(self, source: str, suffix: str | None = None) -> str:
        temporary = None
        if suffix or source.startswith(("http://", "https://")):
            import tempfile
            from urllib.parse import urlparse

            if not suffix:
                suffix = Path(urlparse(source).path).suffix or ".png"
            handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                if source.startswith(("http://", "https://")):
                    import requests

                    response = requests.get(source, timeout=120)
                    response.raise_for_status()
                    handle.write(response.content)
                else:
                    handle.write(base64.b64decode(source.split(",", 1)[-1]))
                temporary = handle.name
            finally:
                handle.close()
            source = temporary
        try:
            return self.parser.parse(source).markdown_result
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
            return {
                "ok": True, "model": SERVED_MODEL_NAME, "quant": QUANT,
                "layout": "PP-DocLayoutV3", "gpu": GPU_TYPE,
            }

        @api.post("/parse")
        def parse(payload: dict[str, Any]):
            try:
                source, suffix = _source(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events() -> Iterator[str]:
                yield json.dumps({"text": "", "pages": 0, "done": False}) + "\n"
                text = self._parse(source, suffix)
                yield json.dumps({"text": text, "pages": 1, "done": True}) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api


@app.local_entrypoint()
def main(image_url: str):
    source, suffix = _source({"image_url": image_url})
    print(GLMOCR().parse.remote(source, suffix=suffix))
