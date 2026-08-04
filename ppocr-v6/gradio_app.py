# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=5.0,<6",
#     "requests>=2,<3",
#     "Pillow>=10",
#     "pymupdf>=1.24",
# ]
# ///
"""Thin Gradio client for the ppocr-v6 Modal deployment.

Loads no model — it calls the deployed endpoint (modal_app.py's `web_app`:
one base URL with /ocr and /health) over HTTP.

Run with:
    PPOCR_V6_BASE_URL=https://<workspace>--ppocr-v6-web-app.modal.run uv run gradio_app.py
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import gradio as gr
import requests
from PIL import Image, ImageDraw

BASE_URL = os.environ.get("PPOCR_V6_BASE_URL", "http://localhost:8000").rstrip("/")
OCR_URL = f"{BASE_URL}/ocr"


def _b64_file(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _first_page_image(path: str) -> Image.Image:
    """Render page 1 so detected boxes have something to be drawn on. PDFs are
    rasterised here at the same 200 dpi the server uses, so the polygon
    coordinates it returns line up with this bitmap."""
    if not _is_pdf(path):
        return Image.open(path).convert("RGB")

    import fitz

    doc = fitz.open(path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def _draw_boxes(img: Image.Image, lines: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for ln in lines:
        poly = ln.get("poly")
        if not poly:
            continue
        pts = [(float(x), float(y)) for x, y in poly]
        # Green = confident, amber = borderline. Makes a bad min_score visible.
        colour = (0, 200, 90) if (ln.get("score") or 0) >= 0.9 else (245, 160, 30)
        draw.line(pts + [pts[0]], fill=colour, width=2)
    return out


def run_ocr(file_path: str | None, min_score: float, reading_order: bool):
    if not file_path:
        raise gr.Error("Please upload an image or PDF first.")

    payload = {
        ("pdf_base64" if _is_pdf(file_path) else "image_base64"): _b64_file(file_path),
        "min_score": float(min_score),
        "reading_order": bool(reading_order),
    }
    try:
        resp = requests.post(OCR_URL, json=payload, timeout=900)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Request to {OCR_URL} failed: {e}")

    lines = data.get("lines", [])
    pages = data.get("pages", 1)
    status = (
        f"**{data.get('lines_kept', 0)}** lines kept of "
        f"{data.get('lines_detected', 0)} detected · {pages} page(s) · "
        f"**{data.get('elapsed_s', '?')} s** on tier `{data.get('tier', '?')}`"
    )

    # Boxes are only drawn for page 1 — a multi-page PDF returns coordinates per
    # page, and stacking them onto one bitmap would be meaningless.
    try:
        base_img = _first_page_image(file_path)
        page0 = [ln for ln in lines if ln.get("page", 0) == 0]
        vis = _draw_boxes(base_img, page0)
    except Exception as e:
        vis = None
        status += f"\n\n_(visualisation unavailable: {e})_"

    return data.get("text", ""), vis, status


with gr.Blocks(theme=gr.themes.Soft(), title="PP-OCRv6") as demo:
    gr.Markdown(f"# PP-OCRv6\nTalking to `{BASE_URL}`")
    gr.Markdown(
        "_Plain-text OCR: fast and small (~133 MB, whole page in a few seconds). "
        "It returns **text lines only** — no table structure, LaTeX or charts. "
        "For those, use the `paddleocr-vl` deployment instead._"
    )

    with gr.Row():
        with gr.Column(scale=5):
            file_input = gr.File(
                label="Upload image or PDF",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf"],
                type="filepath",
            )
            min_score = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="Min confidence",
                info="Detection emits low-confidence single characters; this drops them.",
            )
            reading_order = gr.Checkbox(
                value=True,
                label="Sort into reading order",
                info="Off = raw detection order, which interleaves columns.",
            )
            run_btn = gr.Button("Run OCR", variant="primary")
            status = gr.Markdown("")
        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.Tab("Text"):
                    text_out = gr.Textbox(label="", lines=26, show_copy_button=True)
                with gr.Tab("Detected boxes (page 1)"):
                    vis_out = gr.Image(label="", height=620)

    run_btn.click(
        run_ocr,
        inputs=[file_input, min_score, reading_order],
        outputs=[text_out, vis_out, status],
    )

if __name__ == "__main__":
    demo.queue().launch()
