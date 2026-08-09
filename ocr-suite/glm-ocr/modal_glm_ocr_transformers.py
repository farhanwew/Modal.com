from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import modal


APP_NAME = "glm-ocr-transformers"
MODEL_REPO = "zai-org/GLM-OCR"
SERVED_MODEL_NAME = "glm-ocr"
INNER_PORT = 8080
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
MAX_WORKERS = os.environ.get("GLM_OCR_MAX_WORKERS", "16")
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("glm-ocr-transformers-hf-cache", create_if_missing=True)


# No vLLM/SGLang: both spend minutes on engine startup (CUDA graph capture,
# JIT kernel compilation, multiprocess spawn) that only pays off under
# sustained load, which a cold-start-per-request serverless container never
# has. Plain transformers loads this 0.9B model's weights in under a second
# (measured via vLLM's own loader) and skips all of that — same tradeoff this
# repo already makes for PaddleOCR-VL (modal_paddleocr_vl.py vs the GGUF app).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "torchvision",
        "transformers>=5.3.0",  # GLM-OCR's own README requirement
        "accelerate", "Pillow", "requests",
        "glmocr[selfhosted]", "fastapi[standard]",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)

# Hoisted so these land in the memory snapshot rather than being re-imported on
# every cold start — same convention as modal_paddleocr_vl.py.
with image.imports():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor


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
    volumes={HF_CACHE_DIR: hf_cache},
    enable_memory_snapshot=True,
)
class GLMOCR:
    # snap=True runs once on CPU and its memory image is reused by later cold
    # starts; snap=False reruns each time. Nothing here may touch CUDA or open
    # a network socket, or the snapshot becomes unrestorable — the inner
    # server (which does both) is started in the snap=False phase instead.
    @modal.enter(snap=True)
    def load_to_cpu(self) -> None:
        self.processor = AutoProcessor.from_pretrained(MODEL_REPO, cache_dir=HF_CACHE_DIR)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_REPO, dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR
        )
        self.model.eval()

    @modal.enter(snap=False)
    def start(self) -> None:
        import threading

        import requests
        import uvicorn
        from fastapi import FastAPI
        from glmocr import GlmOcr

        self.model = self.model.to("cuda")
        # A single model instance serving `max_workers` concurrent requests
        # needs its GPU calls serialized — transformers' generate() isn't
        # safe to run from multiple threads at once on the same weights.
        lock = threading.Lock()

        # glmocr's OCRClient always talks HTTP (no in-process mode), so this
        # stands in for vLLM/SGLang — same role, plain transformers
        # underneath. Runs as a background thread in this same process
        # (rather than a subprocess) so its weights are the ones the
        # snap=True phase already loaded and snapshotted.
        inner = FastAPI()

        @inner.get("/health")
        def health():
            return {"ok": True}

        @inner.post("/v1/chat/completions")
        def chat_completions(payload: dict[str, Any]):
            import io

            from PIL import Image

            messages = payload.get("messages", [])
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            text_prompt, pil_image = "", None
            for item in (last_user or {}).get("content", []):
                if item.get("type") == "text":
                    text_prompt = item.get("text", "")
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {})
                    url = url.get("url", url) if isinstance(url, dict) else url
                    b64 = url.split(",", 1)[-1] if url.startswith("data:") else url
                    pil_image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

            content = [{"type": "text", "text": text_prompt}]
            if pil_image is not None:
                content = [{"type": "image", "image": pil_image}, *content]
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)

            max_new_tokens = int(payload.get("max_tokens", 512))
            with lock, torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, use_cache=True
                )
            text = self.processor.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()

            return {"choices": [{"message": {"content": text}}]}

        config = uvicorn.Config(inner, host="127.0.0.1", port=INNER_PORT, log_level="info")
        threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if requests.get(f"http://127.0.0.1:{INNER_PORT}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            raise RuntimeError("inner server did not become healthy in 120s")

        config_path = Path("/tmp/glmocr.yaml")
        config_path.write_text(
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
    api_port: {INNER_PORT}
    model: {SERVED_MODEL_NAME}
    connect_timeout: 600
    request_timeout: 900
""",
            encoding="utf-8",
        )
        self.parser = GlmOcr(config_path=str(config_path), layout_device="cpu")
        hf_cache.commit()

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
