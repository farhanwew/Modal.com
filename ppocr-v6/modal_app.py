from __future__ import annotations

import base64
import io
import os
from typing import Any

import modal


APP_NAME = "ppocr-v6"
CACHE_DIR = "/cache"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

# Three tiers, det+rec sizes from the official pipeline docs:
#   medium  59.4MB + 73.3MB   86.2% Hmean / 83.2% acc
#   small    9.6MB + 20.4MB   84.1%      / 81.3%
#   tiny     1.9MB +  4.4MB   80.6%      / 73.5%
# For comparison, paddleocr-vl/ runs a 1.9 GB VLM.
TIER = os.environ.get("PPOCR_TIER", "medium")

# SLANeXt_* is the accurate default (69.65%) but its transformers backend warns
# "Resampling is not supported in SLANeXt". SLANet_plus (1.8M, 63.69%) is the
# lighter fallback. Override with PPOCR_TABLE_MODEL.
TABLE_STRUCT_MODEL = os.environ.get("PPOCR_TABLE_MODEL", "SLANet_plus")


app = modal.App(APP_NAME)

cache_vol = modal.Volume.from_name("ppocr-v6-cache", create_if_missing=True)

# `paddleocr` here is the pure-Python orchestration package. Verified locally
# that installing it pulls no `paddle` module (61 deps, `paddle installed? False`),
# so with engine="transformers" the actual inference runs on torch. That is the
# whole reason this app can exist alongside a torch-only stack.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers>=5.0.0",
        # [all], not bare paddleocr: the table pipeline needs shapely/pyclipper/
        # premailer/openpyxl and fails with a generic "dependency error" without
        # them. Verified locally that [all] still resolves paddle-free (118 pkgs,
        # `paddle present? False`); it does drag in unused openai/langchain bloat.
        "paddleocr[all]>=3.7.0",
        "Pillow",
        "requests",
        "fastapi[standard]",
        "huggingface_hub[hf_xet]",
        "pymupdf",  # rasterises PDF pages
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "PADDLE_PDX_CACHE_HOME": f"{CACHE_DIR}/paddlex",
        }
    )
)


def _download_bytes(url: str) -> bytes:
    import requests

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.content


def _load_image(image_url: str | None, image_base64: str | None):
    from PIL import Image

    if image_url:
        data = _download_bytes(image_url)
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
    else:
        raise ValueError("Provide image_url or image_base64.")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _resolve_pages(
    image_url: str | None,
    image_base64: str | None,
    pdf_url: str | None,
    pdf_base64: str | None,
    pdf_dpi: int,
) -> list:
    """One image, or every page of a PDF rasterised."""
    if pdf_url or pdf_base64:
        import fitz
        from PIL import Image

        data = _download_bytes(pdf_url) if pdf_url else base64.b64decode(pdf_base64)
        doc = fitz.open(stream=data, filetype="pdf")
        mat = fitz.Matrix(pdf_dpi / 72, pdf_dpi / 72)
        pages = [
            Image.open(io.BytesIO(p.get_pixmap(matrix=mat).tobytes("png"))).convert("RGB")
            for p in doc
        ]
        doc.close()
        return pages
    return [_load_image(image_url, image_base64)]


def _order_lines(lines: list[dict], page_w: int) -> list[dict]:
    """Sort detected text lines into reading order.

    Lines come back in detection order, which interleaves columns. A fixed
    left-half/right-half split is not enough — a newspaper page can hold four
    narrow columns, so "left half" still contains two of them. Instead the left
    edges are clustered: a gap wider than `gap_ratio` of the page starts a new
    column. Lines are then read column by column, top to bottom.

    Still a heuristic. Headlines spanning several columns get filed into
    whichever column their left edge falls in, and text wrapped around a figure
    will be wrong.
    """
    if not lines:
        return lines

    def span(ln) -> tuple[int, int]:
        poly = ln.get("poly") or [[0, 0]]
        xs = [int(p[0]) for p in poly]
        return max(0, min(xs)), min(page_w, max(xs))

    def top_of(ln) -> float:
        poly = ln.get("poly") or [[0, 0]]
        return min(p[1] for p in poly)

    # Clustering left edges doesn't work: within a column they spread out enough
    # that no single gap stands out. Find the gutters instead — vertical bands no
    # text box covers at all. Lines spanning over half the page (headlines, rules)
    # are left out of the coverage map, since one of them would otherwise bridge
    # every gutter and collapse the page back to a single column.
    covered = bytearray(page_w)
    for ln in lines:
        x1, x2 = span(ln)
        if x2 - x1 > page_w * 0.5:
            continue
        for x in range(x1, max(x1 + 1, x2)):
            covered[x] = 1

    min_gutter = max(8, int(page_w * 0.015))
    boundaries: list[int] = []
    run_start = None
    for x in range(page_w):
        if not covered[x]:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None and x - run_start >= min_gutter:
                boundaries.append((run_start + x) // 2)
            run_start = None
    if run_start is not None and page_w - run_start >= min_gutter:
        boundaries.append((run_start + page_w) // 2)

    def column_of(ln) -> int:
        x1, x2 = span(ln)
        centre = (x1 + x2) / 2
        return sum(1 for b in boundaries if centre > b)

    return sorted(lines, key=lambda ln: (column_of(ln), top_of(ln)))


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16 * 1024,
    timeout=900,
    scaledown_window=120,
    volumes={CACHE_DIR: cache_vol},
)
class PPOCRv6Server:
    # No memory snapshot yet, deliberately. Whether PaddleOCR's pipeline can be
    # built without touching CUDA is unknown, and a snapshot that can't restore
    # fails at runtime. Get it working first, measure, then optimise.
    @modal.enter()
    def load(self):
        import time

        import numpy as np
        import torch
        from paddleocr import PaddleOCR

        t0 = time.time()
        self.np = np
        device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.ocr = PaddleOCR(
            text_detection_model_name=f"PP-OCRv6_{TIER}_det",
            text_recognition_model_name=f"PP-OCRv6_{TIER}_rec",
            engine="transformers",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        import importlib.util

        print(
            f"[INIT] tier={TIER} device={device} load_s={time.time() - t0:.1f} "
            f"paddle_present={importlib.util.find_spec('paddle') is not None} "
            f"cuda={torch.cuda.is_available()}"
        )

        # Table structure is a separate pipeline: classify wired/wireless, detect
        # cells, predict structure, then fill cells using the same det+rec models.
        # Built lazily and defensively — /ocr must keep working if this can't load.
        self.table = None
        self._table_error = None
        self._table_device = device

    def _ensure_table(self):
        if self.table is not None or self._table_error is not None:
            return
        import time

        from paddleocr import TableRecognitionPipelineV2

        t0 = time.time()
        try:
            # use_layout_detection=False is required, not just an optimisation:
            # the pipeline otherwise instantiates PP-DocLayout-L, which has no
            # safetensors build and rejects engine="transformers" outright
            # ("Supported engines: ['paddle_static', 'hpi', 'onnxruntime']").
            # Callers are expected to pass an already-cropped table region.
            #
            # The det/rec names must be pinned too: the pipeline defaults to
            # PP-OCRv4_server_det, which likewise has no safetensors build.
            self.table = TableRecognitionPipelineV2(
                engine="transformers",
                use_layout_detection=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                text_detection_model_name=f"PP-OCRv6_{TIER}_det",
                text_recognition_model_name=f"PP-OCRv6_{TIER}_rec",
                wired_table_structure_recognition_model_name=TABLE_STRUCT_MODEL,
                wireless_table_structure_recognition_model_name=TABLE_STRUCT_MODEL,
                device=self._table_device,
            )
            print(f"[INIT] table pipeline load_s={time.time() - t0:.1f}")
        except Exception as e:
            self._table_error = f"{type(e).__name__}: {e}"
            print(f"[INIT] table pipeline unavailable: {self._table_error}")

    def _ocr_one(self, pil, min_score: float, reading_order: bool) -> list[dict]:
        arr = self.np.array(pil)[:, :, ::-1]  # PaddleOCR expects BGR
        results = self.ocr.predict(arr)

        lines = []
        for res in results:
            d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
            texts = d.get("rec_texts", []) or []
            scores = d.get("rec_scores", []) or []
            polys = d.get("rec_polys", []) or d.get("dt_polys", []) or []
            for i, text in enumerate(texts):
                poly = polys[i] if i < len(polys) else None
                lines.append(
                    {
                        "text": text,
                        "score": float(scores[i]) if i < len(scores) else None,
                        "poly": poly.tolist() if hasattr(poly, "tolist") else poly,
                    }
                )

        # Detection produces a tail of low-confidence single characters (observed:
        # "1" at 0.57, "a" at 0.14 on the demo page). They are noise, not text.
        kept = [ln for ln in lines if (ln["score"] or 0) >= min_score]
        if reading_order:
            kept = _order_lines(kept, pil.size[0])
        return kept, len(lines)

    @modal.method()
    def ocr_image(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        pdf_url: str | None = None,
        pdf_base64: str | None = None,
        pdf_dpi: int = 200,
        min_score: float = 0.5,
        reading_order: bool = True,
    ) -> dict[str, Any]:
        import time

        pages = _resolve_pages(image_url, image_base64, pdf_url, pdf_base64, pdf_dpi)

        t0 = time.time()
        all_lines: list[dict] = []
        raw_count = 0
        chunks: list[str] = []
        for page_idx, pil in enumerate(pages):
            kept, raw = self._ocr_one(pil, min_score, reading_order)
            raw_count += raw
            for ln in kept:
                ln["page"] = page_idx
            all_lines.extend(kept)
            if len(pages) > 1:
                chunks.append(f"--- Page {page_idx + 1} / {len(pages)} ---")
            chunks.extend(ln["text"] for ln in kept)
        elapsed = time.time() - t0

        print(
            f"[OCR] tier={TIER} pages={len(pages)} lines={len(all_lines)}/{raw_count} "
            f"min_score={min_score} elapsed_s={elapsed:.2f}"
        )
        return {
            "tier": TIER,
            "pages": len(pages),
            "lines": all_lines,
            "text": "\n".join(chunks),
            "lines_kept": len(all_lines),
            "lines_detected": raw_count,
            "elapsed_s": round(elapsed, 3),
        }


    @modal.method()
    def table_image(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        import time

        self._ensure_table()
        if self.table is None:
            return {"error": f"Table pipeline unavailable: {self._table_error}"}

        pil = _load_image(image_url, image_base64)
        arr = self.np.array(pil)[:, :, ::-1]

        t0 = time.time()
        try:
            results = self.table.predict(arr)
        except Exception as e:
            # Known-broken upstream as of paddleocr 3.7.0: the pipeline *builds*
            # fine on the transformers backend, but predicting raises
            # "KMeans n_clusters must be in range [1, inf). Got 0" from the cell
            # clustering step — cell detection comes back empty. Reproduced on
            # PaddleOCR's own table_recognition.jpg demo image, with both
            # SLANeXt and SLANet_plus. Returned as data, not a 500.
            print(f"[TABLE] predict failed: {type(e).__name__}: {e}")
            return {
                "error": f"{type(e).__name__}: {e}",
                "hint": (
                    "PaddleOCR's table pipeline loads but does not run on the "
                    "transformers backend. Use paddleocr-vl's `table` task instead."
                ),
            }
        elapsed = time.time() - t0

        tables = []
        for res in results:
            d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
            for tbl in d.get("table_res_list", []) or []:
                tables.append(
                    {
                        "html": tbl.get("pred_html", ""),
                        "bbox": tbl.get("table_region_id") or tbl.get("cell_box_list"),
                    }
                )
            if not d.get("table_res_list") and d.get("pred_html"):
                tables.append({"html": d["pred_html"], "bbox": None})

        print(f"[TABLE] tables={len(tables)} elapsed_s={elapsed:.2f}")
        return {"tables": tables, "count": len(tables), "elapsed_s": round(elapsed, 3)}


@app.function(image=image, timeout=900)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException

    api = FastAPI(title=APP_NAME)

    @api.get("/health")
    def health():
        return {"ok": True, "tier": TIER, "gpu": GPU_TYPE}

    @api.post("/ocr")
    def ocr(payload: dict[str, Any]):
        if not any(
            payload.get(k) for k in ("image_url", "image_base64", "pdf_url", "pdf_base64")
        ):
            raise HTTPException(
                400, detail="Provide one of image_url, image_base64, pdf_url or pdf_base64."
            )
        return PPOCRv6Server().ocr_image.remote(
            image_url=payload.get("image_url"),
            image_base64=payload.get("image_base64"),
            pdf_url=payload.get("pdf_url"),
            pdf_base64=payload.get("pdf_base64"),
            pdf_dpi=int(payload.get("pdf_dpi", 200)),
            min_score=float(payload.get("min_score", 0.5)),
            reading_order=bool(payload.get("reading_order", True)),
        )

    @api.post("/table")
    def table(payload: dict[str, Any]):
        if not (payload.get("image_url") or payload.get("image_base64")):
            raise HTTPException(400, detail="Provide image_url or image_base64.")
        return PPOCRv6Server().table_image.remote(
            image_url=payload.get("image_url"),
            image_base64=payload.get("image_base64"),
        )

    return api


@app.local_entrypoint()
def main(image_url: str):
    result = PPOCRv6Server().ocr_image.remote(image_url=image_url)
    print(f"{len(result['lines'])} lines in {result['elapsed_s']}s")
    print(result["text"][:2000])
