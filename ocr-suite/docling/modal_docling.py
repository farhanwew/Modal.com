from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import modal


APP_NAME = "docling"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(
        "quay.io/docling-project/docling-serve-cu128:v1.30.0"
    )
    .entrypoint([])
    .env({"DOCLING_DEVICE": "cuda", "DOCLING_NUM_THREADS": "4"})
)


def _source(payload: dict[str, Any]) -> tuple[str, str | None]:
    for key, suffix in (
        ("image_url", None),
        ("pdf_url", None),
        ("image_base64", ".png"),
        ("pdf_base64", ".pdf"),
    ):
        if payload.get(key):
            return str(payload[key]), suffix
    raise ValueError("Provide image_url, image_base64, pdf_url or pdf_base64.")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=1800,
    scaledown_window=120,
)
class Docling:
    @modal.enter()
    def load(self) -> None:
        import torch

        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )
        from docling.pipeline.threaded_standard_pdf_pipeline import (
            ThreadedStandardPdfPipeline,
        )

        torch.set_float32_matmul_precision("high")
        options = ThreadedPdfPipelineOptions(
            artifacts_path=Path("/opt/app-root/src/.cache/docling/models"),
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            ocr_batch_size=4,
            layout_batch_size=64,
            table_batch_size=4,
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=options,
                ),
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=options,
                ),
            }
        )
        self.converter.initialize_pipeline(InputFormat.PDF)

    @modal.method()
    def parse(self, source: str, suffix: str | None = None) -> tuple[str, int]:
        return self._parse(source, suffix)

    def _parse(self, source: str, suffix: str | None = None) -> tuple[str, int]:
        temporary = None
        if suffix:
            handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                handle.write(base64.b64decode(source.split(",", 1)[-1]))
                temporary = handle.name
            finally:
                handle.close()
            source = temporary

        try:
            result = self.converter.convert(source)
            return result.document.export_to_markdown(), len(result.document.pages)
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
            return {"ok": True, "gpu": GPU_TYPE}

        @api.post("/parse")
        def parse(payload: dict[str, Any]):
            try:
                source, suffix = _source(payload)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            def events() -> Iterator[str]:
                yield json.dumps({"text": "", "pages": 0, "done": False}) + "\n"
                text, pages = self._parse(source, suffix)
                yield json.dumps({"text": text, "pages": pages, "done": True}) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson")

        return api


@app.local_entrypoint()
def main(url: str = "https://arxiv.org/pdf/2501.17887"):
    source, suffix = _source({"pdf_url": url})
    text, pages = Docling().parse.remote(source, suffix=suffix)
    print(f"{pages} pages\n{text[:2000]}")
