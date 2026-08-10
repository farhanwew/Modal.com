from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import modal


APP_NAME = "mineru-pipeline"
MINERU_REF = os.environ.get("MINERU_REF", "master")
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
MODEL_DIR = "/models"

app = modal.App(APP_NAME)
models = modal.Volume.from_name("mineru-pipeline-models", create_if_missing=True)


def _download_models() -> None:
    import subprocess

    subprocess.run(
        ["mineru-models-download", "-s", "huggingface", "-m", "pipeline"],
        check=True,
        env={**os.environ, "MINERU_MODEL_SOURCE": "huggingface"},
    )


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "fonts-noto-cjk")
    .uv_pip_install(
        f"mineru[pipeline] @ git+https://github.com/opendatalab/MinerU.git@{MINERU_REF}",
        "onnxruntime-gpu==1.23.2",
        "six",
        "fastapi[standard]",
        "requests",
    )
    .env({"MINERU_MODEL_SOURCE": "local", "MINERU_API_ENABLE_FASTAPI_DOCS": "0"})
    .run_function(_download_models, volumes={MODEL_DIR: models}, timeout=1800)
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


def _read(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, timeout=120)
        response.raise_for_status()
        return response.content
    return base64.b64decode(source.split(",", 1)[-1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_files(root: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    markdown = next(root.rglob("*.md"), Path())
    images: dict[str, str] = {}
    extras: dict[str, Any] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
            images[path.name] = base64.b64encode(path.read_bytes()).decode("ascii")
        elif path.name.endswith(("_middle.json", "_content_list.json", "_content_list_v2.json", "_model.json")):
            try:
                extras[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return (
        markdown.read_text(encoding="utf-8") if markdown.exists() else "",
        images,
        extras,
    )


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=24 * 1024,
    timeout=1800,
    scaledown_window=300,
    volumes={MODEL_DIR: models},
)
@modal.concurrent(max_inputs=1)
class MinerUPipeline:
    @modal.enter()
    def load(self) -> None:
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    def _parse(self, source: str, is_pdf: bool, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        from mineru.cli.common import do_parse, read_fn

        root = Path(tempfile.mkdtemp(prefix="mineru-pipeline-"))
        input_path = root / ("input.pdf" if is_pdf else "input.png")
        input_path.write_bytes(_read(source))
        output = root / "output"
        output.mkdir()
        try:
            yield {"text": "", "pages": 0, "done": False}
            do_parse(
                output_dir=str(output),
                pdf_file_names=["document"],
                pdf_bytes_list=[read_fn(input_path)],
                p_lang_list=[str(payload.get("lang", "ch"))],
                backend="pipeline",
                parse_method=str(payload.get("parse_method", "auto")),
                formula_enable=bool(payload.get("formula_enable", True)),
                table_enable=bool(payload.get("table_enable", True)),
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=True,
                f_dump_model_output=True,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
                start_page_id=0,
                end_page_id=(
                    max(0, int(payload["max_pages"]) - 1)
                    if payload.get("max_pages")
                    else None
                ),
            )
            markdown, images, extras = _result_files(output)
            if is_pdf:
                import pypdfium2 as pdfium

                page_count = len(pdfium.PdfDocument(str(input_path)))
            else:
                page_count = 1
            yield {
                "text": markdown,
                "pages": min(page_count, int(payload["max_pages"])) if payload.get("max_pages") else page_count,
                "done": True,
                "new_images": images,
                "official_results": extras,
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse

        api = FastAPI(title=APP_NAME)

        @api.get("/health")
        def health():
            return {"ok": True, "backend": APP_NAME, "gpu": GPU_TYPE, "device": self.device}

        @api.post("/parse")
        def parse(payload: dict[str, Any]):
            try:
                source, is_pdf = _source(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events():
                for event in self._parse(source, is_pdf, payload):
                    yield json.dumps(event, ensure_ascii=False, default=_jsonable) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api


@app.local_entrypoint()
def main(image_url: str):
    for event in MinerUPipeline()._parse.remote_gen(image_url, False, {}):
        if event["done"]:
            print(event["text"])
