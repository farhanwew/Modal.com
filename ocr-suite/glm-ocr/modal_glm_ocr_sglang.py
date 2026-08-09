from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import modal


APP_NAME = "glm-ocr-sglang"
MODEL_REPO = "zai-org/GLM-OCR"
SERVED_MODEL_NAME = "glm-ocr"
PORT = 8080
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
# glmocr's own default is 32 ("lower to reduce 503s on a busy server"); SGLang's
# continuous batching (unlike Ollama's llama-server, which divides one fixed
# context across n_parallel slots — see the Ollama attempt logged in CLAUDE.md)
# shares one KV-cache pool across requests, so raising this doesn't shrink
# anyone else's context budget.
MAX_WORKERS = os.environ.get("GLM_OCR_MAX_WORKERS", "16")
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("glm-ocr-sglang-hf-cache", create_if_missing=True)


# Every lmsysorg/sglang docker image tag tried (default v0.5.10 and
# v0.5.10.post1-cu130-runtime) ships a sgl_kernel binary with only an sm100
# (Hopper/Blackwell) build of common_ops — no sm89 (Ada/L4), so it ABI-fails
# on this GPU regardless of the torch/CUDA version match. A fresh PyPI
# install resolves sgl-kernel 0.4.5, which does ship sm89 and gets past model
# load and GPU memory setup — so build sglang + glmocr from a plain CUDA base
# instead of any prebuilt sglang image.
image = (
    # flashinfer's quantization op JIT-compiles a real .cu file via nvcc, not
    # just a C/C++ kernel — the "runtime" CUDA base has no CUDA compiler at
    # all (nvcc ships only with "devel"), so build-essential alone wasn't
    # enough; ninja's ninja: build stopped: subcommand failed was nvcc itself
    # missing, confirmed by "/usr/local/cuda/bin/nvcc: not found".
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "ninja-build")
    .uv_pip_install("sglang>=0.5.10", "glmocr[selfhosted]", "fastapi[standard]", "requests")
    .env({"HF_HOME": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
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
    memory=24 * 1024,
    timeout=1800,
    scaledown_window=120,
    volumes={HF_CACHE_DIR: hf_cache},
)
class GLMOCR:
    @modal.enter()
    def start(self) -> None:
        from glmocr import GlmOcr

        # Launch command per GLM-OCR's own README ("Using SGLang" section).
        # NEXTN speculative decoding needs SGLANG_ENABLE_SPEC_V2=1. Module
        # form rather than the `sglang serve` CLI wrapper — works regardless
        # of whether a console script happens to be on PATH.
        self.process = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", MODEL_REPO,
                "--served-model-name", SERVED_MODEL_NAME,
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--speculative-algorithm", "NEXTN",
                "--speculative-num-steps", "3",
                "--speculative-eagle-topk", "1",
                "--speculative-num-draft-tokens", "4",
            ],
            env={**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"},
        )

        import requests

        # First cold start downloads the BF16 weights from HF into the cached
        # volume; generous deadline to cover that on top of server warmup.
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"sglang serve exited with {self.process.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(2)
        else:
            raise RuntimeError("sglang serve did not become healthy in 900s")

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
        # api_mode/api_path default to "openai" / "/v1/chat/completions" — the
        # mode confirmed (via the scrapped Ollama attempt) to actually generate
        # real tokens, unlike "ollama_generate" which silently returned blank.
        self.parser = GlmOcr(config_path=str(config), layout_device="cpu")
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
        import json

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse

        api = FastAPI(title=APP_NAME)

        @api.get("/health")
        def health():
            return {"ok": True, "model": SERVED_MODEL_NAME, "layout": "PP-DocLayoutV3", "gpu": GPU_TYPE}

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
