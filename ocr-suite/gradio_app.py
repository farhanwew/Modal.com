# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=5.0,<6",
#     "requests>=2,<3",
#     "Pillow>=10",
#     "pymupdf>=1.24",
#     "opendataloader-pdf",
# ]
# ///
"""One Gradio client for the OCR backends in this folder.

Four run on Modal and are reached over HTTP; the fifth runs in this process.

    PADDLEOCR_VL_GGUF_BASE_URL   modal_paddleocr_vl_gguf.py   llama.cpp, fastest
    PADDLEOCR_VL_GGUF_OFFICIAL_BASE_URL  official PaddleOCR pipeline + llama.cpp
    PADDLEOCR_VL_BASE_URL        modal_paddleocr_vl.py        transformers
    UNLIMITED_OCR_BASE_URL       modal_unlimited_ocr.py       DeepSeek-OCR-style VLM
    (no variable)                opendataloader-pdf           local, no model, no GPU

Only the backends whose env var is set are selectable; the rest are listed as
unconfigured rather than failing at click time. OpenDataLoader needs no variable
— just a Java 11+ runtime on the machine.

Run with:
    PADDLEOCR_VL_GGUF_BASE_URL=https://… \
    PADDLEOCR_VL_BASE_URL=https://… \
    UNLIMITED_OCR_BASE_URL=https://… \
    uv run gradio_app.py
"""

from __future__ import annotations

import base64
import html
import importlib.util
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr
import requests

def _load_dotenv(path: Path) -> None:
    """Read .env sitting next to this script, so `uv run gradio_app.py` needs no
    shell setup and matches what `docker run --env-file .env` already uses.

    Hand-rolled rather than python-dotenv: the format is KEY=VALUE and this is a
    dozen lines, against a dependency in a PEP 723 block people copy around. The
    real environment wins — an explicit export still overrides the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).with_name(".env"))


# --- backends ---------------------------------------------------------------
# `parse` takes whole pages/PDFs and streams NDJSON. `recognize` takes one image
# plus a task and is PaddleOCR-VL-only — Unlimited-OCR has no task concept, its
# behaviour is steered by a free-text prompt instead.
BACKENDS: dict[str, dict] = {
    "PaddleOCR-VL · GGUF": {
        "env": "PADDLEOCR_VL_GGUF_BASE_URL",
        "parse": "/parse-page",
        "recognize": "/recognize",
        "family": "paddle",
        "repo": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
    },
    "PaddleOCR-VL · transformers": {
        "env": "PADDLEOCR_VL_BASE_URL",
        "parse": "/parse-page",
        "recognize": "/recognize",
        "family": "paddle",
        "repo": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6",
    },
    "PaddleOCR-VL · GGUF official": {
        "env": "PADDLEOCR_VL_GGUF_OFFICIAL_BASE_URL",
        "parse": "/parse-page",
        "recognize": "/recognize",
        "family": "paddle",
        "repo": "https://github.com/PaddlePaddle/PaddleOCR",
    },
    "Docling": {
        "env": "DOCLING_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "docling",
        "repo": "https://github.com/docling-project/docling",
    },
    "MinerU 2.5 Pro · transformers": {
        "env": "MINERU_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "mineru",
        "repo": "https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B",
    },
    "MinerU 2.5 Pro · GGUF Q8": {
        "env": "MINERU_GGUF_Q8_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "mineru",
        "repo": "https://huggingface.co/mradermacher/MinerU2.5-Pro-2604-1.2B-GGUF",
    },
    "MinerU 2.5 Pro · GGUF Q4": {
        "env": "MINERU_GGUF_Q4_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "mineru",
        "repo": "https://huggingface.co/mradermacher/MinerU2.5-Pro-2604-1.2B-GGUF",
    },
    "Unlimited-OCR": {
        "env": "UNLIMITED_OCR_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "unlimited",
        "repo": "https://huggingface.co/baidu/Unlimited-OCR",
    },
    # /parse's NDJSON shape (text/pages/done) is generic — like Docling above,
    # this needs no family-specific payload fields or option group, only the
    # existing else-branch response parsing in run_parse.
    "GLM-OCR · transformers": {
        "env": "GLM_OCR_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "glm_ocr",
        "repo": "https://huggingface.co/zai-org/GLM-OCR",
    },
    "GLM-OCR · GGUF Q4": {
        "env": "GLM_OCR_GGUF_Q4_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "glm_ocr",
        "repo": "https://huggingface.co/mradermacher/GLM-OCR-GGUF",
    },
    "GLM-OCR · GGUF Q8": {
        "env": "GLM_OCR_GGUF_Q8_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "glm_ocr",
        "repo": "https://huggingface.co/mradermacher/GLM-OCR-GGUF",
    },
    # Runs in this process, not on Modal: a deterministic Java parser that reads
    # the PDF's own text and structure. No model, no GPU, no upload — and no OCR
    # either, so it is blank on scanned pages. Measured 0.92 s for a page that
    # the GPU pipeline takes seconds on, because it never guesses at pixels.
    "OpenDataLoader (local)": {
        "env": "—",
        "parse": "local",
        "recognize": None,
        "family": "local",
        "repo": "https://github.com/opendataloader-project/opendataloader-pdf",
    },
}


def _find_java() -> str | None:
    """opendataloader-pdf spawns a JVM. Java is often installed but absent from
    PATH on Windows, so look where it actually lands before giving up."""
    exe = shutil.which("java")
    if exe:
        return exe
    for root in filter(None, [
        os.environ.get("JAVA_HOME"),
        *(str(p) for p in Path(os.environ.get("ProgramFiles", "C:/Program Files"))
          .glob("*/jdk*") if p.is_dir()),
    ]):
        candidate = Path(root) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.exists():
            os.environ.setdefault("JAVA_HOME", root)
            os.environ["PATH"] = f"{candidate.parent}{os.pathsep}" + os.environ["PATH"]
            return str(candidate)
    return None


def _odl_available() -> tuple[bool, str]:
    if importlib.util.find_spec("opendataloader_pdf") is None:
        return False, "opendataloader-pdf is not installed"
    if not _find_java():
        return False, "no Java 11+ runtime found (set JAVA_HOME)"
    return True, ""


ODL_OK, ODL_WHY = _odl_available()

for _cfg in BACKENDS.values():
    if _cfg["family"] == "local":
        _cfg["base"] = "local" if ODL_OK else ""
    else:
        _cfg["base"] = os.environ.get(_cfg["env"], "").rstrip("/")

CONFIGURED = [name for name, c in BACKENDS.items() if c["base"]]
PARSE_CHOICES = CONFIGURED or list(BACKENDS)
RECOGNIZE_CHOICES = [n for n in PARSE_CHOICES if BACKENDS[n]["recognize"]]

MODE_MAP = {"Long": "gundam", "Base": "base"}

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

TASK_BY_LABEL = {
    "Text Recognition": "ocr",
    "Formula Recognition": "formula",
    "Table Recognition": "table",
    "Chart Recognition": "chart",
    "Seal Recognition": "seal",
    "Spotting": "spotting",
}

# Long documents used to push the controls off-screen because the output pane
# grew with the page. The scrollbox comes from gr.Markdown's own `max_height`,
# NOT from CSS here: setting overflow on the wrapper as well produced two nested
# scrollbars, since Gradio already scrolls the inner content.
#
# Layout only — no colour or token overrides. A previous attempt at hand-theming
# this UI set background tokens without the matching text tokens and produced
# dark-on-dark text, so colours stay with gr.themes.Soft(). The one exception is
# the fullscreen background, which reads Gradio's own variable rather than a
# literal, so it follows whichever theme is active.
PANE_MAX_H = os.environ.get("PANE_MAX_HEIGHT", "68vh")

CSS = """
.ocr-pane table { border-collapse: collapse; }
.ocr-pane td, .ocr-pane th { padding: 2px 6px; }

/* Native fullscreen: the element is lifted out of the page, so it must bring
   its own background and reclaim the height that max_height caps. */
.ocr-pane:fullscreen {
    max-height: 100vh !important;
    height: 100vh;
    overflow-y: auto;
    padding: 2rem 3rem;
    background: var(--body-background-fill);
}
.ocr-pane:fullscreen::backdrop { background: var(--body-background-fill); }
.fs-btn { max-width: 8rem; }

/* Extracted figures sit inline in the preview; cap them so one large chart
   does not push the rest of the document out of view. */
.ocr-pane img { max-width: 100%; height: auto; max-height: 45vh; }

/* Gallery grid cells default to a fixed square-ish aspect ratio regardless of
   object_fit — object_fit only shapes the image inside the cell, not the cell
   itself. That left large empty top/bottom gaps on landscape PDF pages (short
   relative to their width). Letting cell height follow the image instead of a
   forced ratio removes the letterboxing for both orientations. */
#src-gallery .thumbnail-item, #src-gallery .grid-container .thumbnail-item {
    aspect-ratio: unset !important;
    height: auto !important;
}
"""


def _fullscreen_js(elem_id: str) -> str:
    """Toggle native fullscreen on one pane. Gradio has no fullscreen for
    Markdown, and `js=` runs in the browser with no server round trip."""
    return (
        "() => {"
        f"  const el = document.getElementById({elem_id!r});"
        "   if (!el) return;"
        "   if (document.fullscreenElement) { document.exitFullscreen(); }"
        "   else { el.requestFullscreen && el.requestFullscreen(); }"
        "}"
    )

_OTSL_TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)")


def otsl_to_html(otsl: str) -> str:
    """OTSL -> HTML with real colspan AND rowspan.

    The `table` task returns OTSL markup, not HTML or Markdown; a browser
    silently drops the unknown tags, so feeding it to gr.Markdown renders as
    garbled text.

    The tags encode a grid, not a list of cells:
        fcel  a cell with content        ecel  an empty cell
        lcel  merged with the cell LEFT  ucel  merged with the cell ABOVE
        xcel  merged both ways           nl    end of row

    An earlier version handled lcel and treated ucel as just another empty
    cell, which silently dropped every vertical merge — a header spanning two
    rows came out as one header plus a blank. Resolving each position to the
    cell that owns it gets both spans right.
    """
    grid: list[list[tuple[str, str]]] = []
    for raw_row in otsl.split("<nl>"):
        tokens = _OTSL_TOKEN.findall(raw_row)
        if tokens:
            grid.append([(tag, text.strip()) for tag, text in tokens])
    if not grid:
        return ""

    width = max(len(r) for r in grid)
    grid = [r + [("ecel", "")] * (width - len(r)) for r in grid]

    # owner[r][c] = coordinates of the cell this position belongs to.
    owner: list[list[tuple[int, int]]] = [[(r, c) for c in range(width)]
                                          for r in range(len(grid))]
    for r, row in enumerate(grid):
        for c, (tag, _) in enumerate(row):
            if tag == "lcel" and c > 0:
                owner[r][c] = owner[r][c - 1]
            elif tag == "ucel" and r > 0:
                owner[r][c] = owner[r - 1][c]
            elif tag == "xcel":
                if r > 0:
                    owner[r][c] = owner[r - 1][c]
                elif c > 0:
                    owner[r][c] = owner[r][c - 1]

    spans: dict[tuple[int, int], list[int]] = {}
    for r in range(len(grid)):
        for c in range(width):
            a = owner[r][c]
            s = spans.setdefault(a, [0, 0, set()])
            if r == a[0]:
                s[0] += 1              # columns covered on the anchor's own row
            s[2].add(r)                # every row this anchor reaches
    for a, s in spans.items():
        s[1] = len(s[2])

    out = ['<table border="1" style="border-collapse:collapse">']
    for r in range(len(grid)):
        out.append("<tr>")
        for c in range(width):
            if owner[r][c] != (r, c):
                continue               # a continuation, already emitted
            colspan, rowspan, _ = spans[(r, c)]
            attr = ""
            if colspan > 1:
                attr += f' colspan="{colspan}"'
            if rowspan > 1:
                attr += f' rowspan="{rowspan}"'
            out.append(f"<td{attr}>{html.escape(grid[r][c][1])}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _render(task: str, text: str) -> str:
    return (otsl_to_html(text) or text) if task == "table" else text


# Payload size, not inference, dominated wall clock: the same 1524x1368 page took
# 18.0 s as a 3029 KB base64 PNG and 7.5 s as a 929 KB base64 JPEG — 2.4x with
# the model untouched. Quality 92 rather than 88, since OCR of small text is
# exactly where JPEG artefacts bite. Set JPEG_ABOVE_KB=0 to send bytes verbatim.
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))
JPEG_ABOVE_KB = int(os.environ.get("JPEG_ABOVE_KB", "400"))


def _b64_file(path: str, *, reencode: bool = True) -> str:
    raw = Path(path).read_bytes()
    if reencode and JPEG_ABOVE_KB and len(raw) > JPEG_ABOVE_KB * 1024:
        try:
            import io as _io

            from PIL import Image

            buf = _io.BytesIO()
            Image.open(_io.BytesIO(raw)).convert("RGB").save(
                buf, "JPEG", quality=JPEG_QUALITY
            )
            if buf.tell() < len(raw):
                raw = buf.getvalue()
        except Exception:
            pass  # unreadable or already optimal — send the original
    return base64.b64encode(raw).decode("ascii")


def _payload_for_file(path: str) -> dict:
    """PDFs go over verbatim; re-encoding one as JPEG would be nonsense."""
    is_pdf = Path(path).suffix.lower() == ".pdf"
    key = "pdf_base64" if is_pdf else "image_base64"
    return {key: _b64_file(path, reencode=not is_pdf)}


def _looks_pdf(url: str) -> bool:
    """Extension only. A URL that serves a PDF without saying so in the path
    will be sent as an image and rejected server-side with a clear error, which
    is better than guessing by downloading it here — the whole point of the URL
    path is that this machine never touches the bytes."""
    return Path(url.split("?", 1)[0]).suffix.lower() == ".pdf"


def _payload_for_url(url: str) -> dict:
    return {"pdf_url" if _looks_pdf(url) else "image_url": url.strip()}


def _payload_for_source(file_path: str | None, url: str | None) -> dict:
    """URL wins when both are set.

    Sending a URL is not a convenience — it is the single biggest lever on wall
    clock. A 27-page PDF measured 51 s of server execution against 144 s total,
    the difference being base64 upload from this machine. With a URL, Modal
    fetches it over datacentre network and that gap disappears.
    """
    if url and url.strip():
        return _payload_for_url(url)
    if file_path:
        return _payload_for_file(file_path)
    raise gr.Error("Upload a file or paste a URL first.")


# --- source preview ----------------------------------------------------------
PREVIEW_MAX_PAGES = int(os.environ.get("PREVIEW_MAX_PAGES", "0"))
PREVIEW_DPI = int(os.environ.get("PREVIEW_DPI", "80"))


def _preview_pages(path: str) -> list:
    """Thumbnails of the input, for the gallery. Deliberately capped: a 319-page
    PDF rendered in full would take longer to preview than to OCR."""
    from PIL import Image

    if Path(path).suffix.lower() != ".pdf":
        return [Image.open(path).convert("RGB")]

    import fitz

    doc = fitz.open(path)
    mat = fitz.Matrix(PREVIEW_DPI / 72, PREVIEW_DPI / 72)
    out = []
    for i, page in enumerate(doc):
        if PREVIEW_MAX_PAGES and i >= PREVIEW_MAX_PAGES:
            break
        pix = page.get_pixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        out.append(img.convert("RGB") if mode != "RGB" else img)
    doc.close()
    return out


def preview_file(path: str | None):
    if not path:
        return None, ""
    try:
        pages = _preview_pages(path)
    except Exception as e:
        return None, f"_preview failed: {type(e).__name__}: {e}_"
    note = f"{len(pages)} page(s) shown"
    if PREVIEW_MAX_PAGES and len(pages) == PREVIEW_MAX_PAGES:
        note += f" (capped at {PREVIEW_MAX_PAGES}; the parse still uses all of them)"
    return pages, f"_{note}_"


def preview_url(url: str | None):
    """Downloading here defeats the point of the URL path, so it is behind its
    own button rather than firing as you type."""
    if not url or not url.strip():
        return None, "_paste a URL first_"
    try:
        raw = requests.get(url.strip(), timeout=120,
                           headers={"User-Agent": "Mozilla/5.0"}).content
    except requests.exceptions.RequestException as e:
        return None, f"_fetch failed: {e}_"
    suffix = ".pdf" if _looks_pdf(url) else Path(url.split("?", 1)[0]).suffix or ".bin"
    tmp = Path(tempfile.mkdtemp(prefix="preview-")) / f"src{suffix}"
    tmp.write_bytes(raw)
    pages, note = preview_file(str(tmp))
    return pages, note + f" · fetched {len(raw) / 1e6:.1f} MB locally for preview only"


# --- extracted figures -------------------------------------------------------
# The backend crops `figure` blocks and references them from the markdown as
# ![](images/pN_bX_bY.jpg), sending the bytes once in `new_images`. The run
# directory is ephemeral: a fresh one per parse, wiped when the next parse
# starts, so nothing accumulates on disk between runs. What the user keeps is
# the ZIP.
RUN_ROOT = Path(tempfile.gettempdir()) / "ocr-suite-runs"
_IMG_REF = re.compile(r"!\[([^\]]*)\]\(images/([^)]+)\)")


def _new_run_dir() -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=RUN_ROOT))
    (run / "images").mkdir()
    return run


def _write_images(run: Path, images: dict[str, str]) -> None:
    for name, b64 in (images or {}).items():
        # basename() so a crafted name cannot escape the run directory.
        target = run / "images" / Path(name).name
        try:
            target.write_bytes(base64.b64decode(b64))
        except Exception:
            pass  # a bad crop should not abort a whole document


def _mineru_artifacts(run: Path, file_path: str | None, source_url: str | None,
                       blocks: list[dict], is_pdf: bool) -> None:
    """Write MinerU JSON, bbox preview, and detected visual crops into the ZIP."""
    import pymupdf
    from PIL import Image, ImageDraw

    extras = run / "extras"
    extras.mkdir(exist_ok=True)
    (extras / "result.json").write_text(
        json.dumps({"pages": max((b.get("page", 1) for b in blocks), default=0),
                    "blocks": blocks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if source_url:
        response = requests.get(source_url.strip(), timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        raw = response.content
    elif file_path:
        raw = Path(file_path).read_bytes()
    else:
        return

    colors = {
        "title": (0.95, 0.35, 0.15), "table": (0.15, 0.55, 0.95),
        "image": (0.15, 0.75, 0.35), "chart": (0.55, 0.25, 0.85),
        "equation": (0.85, 0.55, 0.1),
    }
    visual_types = {"image", "chart", "image_block"}

    if is_pdf:
        doc = pymupdf.open(stream=raw, filetype="pdf")
        try:
            for index, block in enumerate(blocks, 1):
                page_index = int(block.get("page", 1)) - 1
                bbox = block.get("bbox")
                if not (0 <= page_index < len(doc) and isinstance(bbox, list) and len(bbox) == 4):
                    continue
                page = doc[page_index]
                rect = pymupdf.Rect(
                    bbox[0] * page.rect.width, bbox[1] * page.rect.height,
                    bbox[2] * page.rect.width, bbox[3] * page.rect.height,
                )
                color = colors.get(block.get("type"), (0.8, 0.15, 0.15))
                if block.get("type") in visual_types and not rect.is_empty:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=rect, alpha=False)
                    pix.save(str(run / "images" / f"page_{page_index + 1:04d}_{block['type']}_{index:04d}.png"))
                page.draw_rect(rect, color=color, width=1)
                page.insert_text((rect.x0, max(7, rect.y0 - 2)), str(block.get("type", "block")),
                                 fontsize=6, color=color)
            doc.save(str(extras / "document-bboxes.pdf"))
        finally:
            doc.close()
        return

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, block in enumerate(blocks, 1):
        bbox = block.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        box = tuple(round(v * size) for v, size in zip(bbox, (image.width, image.height) * 2))
        color = tuple(round(c * 255) for c in colors.get(block.get("type"), (0.8, 0.15, 0.15)))
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 2, box[1] + 2), str(block.get("type", "block")), fill=color)
        if block.get("type") in visual_types:
            image.crop(box).save(run / "images" / f"image_{block['type']}_{index:04d}.png")
    image.save(extras / "image-bboxes.png")


def _inline_images(markdown: str, run: Path) -> str:
    """Rewrite images/NAME to a data URI, for the preview pane only.

    The markdown that goes into the ZIP keeps the relative path so the bundle
    resolves offline; only what gr.Markdown renders needs to be self-contained,
    since it has no file server rooted at the run directory."""
    def sub(m: re.Match) -> str:
        path = run / "images" / Path(m.group(2)).name
        if not path.exists():
            return m.group(0)
        # Derive the type from the extension. Labelling a PNG as image/jpeg
        # happens to render — browsers sniff the bytes — but it is wrong, and
        # OpenDataLoader writes PNG while the Modal backends write JPEG.
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"![{m.group(1)}](data:{mime};base64,{b64})"

    return _IMG_REF.sub(sub, markdown)


HTML_SHELL = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  td, th {{ border: 1px solid #999; padding: 4px 8px; vertical-align: top; }}
  img {{ max-width: 100%; height: auto; }}
  pre {{ overflow-x: auto; }}
</style>
{body}
"""


def _strip_wrapping_fence(text: str) -> str:
    """Drop a code fence that wraps the whole answer.

    VLMs like to return the page inside ```…```, and one leading fence turns the
    entire document — tables included — into a single code block. Observed with
    Docling's gemma preset, whose prompt does not forbid it (ours does, and is
    still ignored often enough to need this).

    Only a fence on the very first line counts, so a document that legitimately
    contains fenced code further down keeps it.
    """
    lines = text.strip().splitlines()
    if not lines or not lines[0].lstrip().startswith("```"):
        return text
    if len(lines) > 1 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _md_to_html(markdown: str, title: str) -> str:
    """Markdown -> a standalone HTML file.

    `html=True` matters: the table blocks are already raw <table> markup with
    colspan and rowspan (that is the whole point of table_format=html), and a
    converter that escapes embedded HTML would turn them back into visible tags.
    markdown-it-py arrives with gradio, so this costs no new dependency.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    return HTML_SHELL.format(title=html.escape(title), body=md.render(markdown))


def _build_zip(run: Path, markdown: str, source_name: str,
               html_body: str | None = None) -> str:
    """Bundle markdown, HTML, extracted images and anything else in `extras/`.

    html_body is passed when a backend produced HTML itself; everything else
    gets the markdown converted here.
    """
    run.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name or "document").stem or "document"

    (run / "document.md").write_text(markdown, encoding="utf-8")
    (run / "document.html").write_text(
        html_body or _md_to_html(markdown, stem), encoding="utf-8"
    )

    zip_path = run / f"{stem}-ocr.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run / "document.md", "document.md")
        z.write(run / "document.html", "document.html")
        for img in sorted((run / "images").glob("*")):
            z.write(img, f"images/{img.name}")
        for extra in sorted((run / "extras").glob("*")):
            z.write(extra, extra.name)
    return str(zip_path)


# --- OpenDataLoader, in-process ----------------------------------------------
# Its markdown references extracted images as ![](<stem_images/imageFileN.png>) —
# angle-bracketed, relative to output_dir, and undocumented; found by running it
# and reading the output. Rewrite to images/NAME so the ZIP and the preview work
# exactly as they do for the Modal backends.
_ODL_IMG = re.compile(r"!\[([^\]]*)\]\(<?([^)>]+)>?\)")


def _odl_run(pdf_path: str, run: Path, struct_tree: bool, sanitize: bool,
             keep_breaks: bool, table_method: str = "default",
             formats: list[str] | None = None,
             ) -> tuple[str, int, str | None]:
    import opendataloader_pdf

    # markdown and html are always produced: the UI panes and the bundle are
    # built from them regardless of what else the user ticked.
    wanted = sorted({"markdown", "html", *(formats or [])})

    out = run / "odl"
    out.mkdir(parents=True, exist_ok=True)
    opendataloader_pdf.convert(
        input_path=[pdf_path],
        output_dir=str(out),
        format=",".join(wanted),
        image_output="external",
        image_format="png",
        use_struct_tree=struct_tree,
        sanitize=sanitize,
        keep_line_breaks=keep_breaks,
        table_method=table_method,
        # Without this, markdown is plain pipe tables and every merged cell is
        # flattened to a blank. With it, a header spanning two columns comes out
        # as <td colspan="2"> — measured on a table whose grid genuinely omits
        # the internal borders. The .html output keeps the spans either way.
        markdown_with_html=True,
        quiet=True,
        # include_header_footer stays at its default False: running headers and
        # page numbers are noise in an LLM context, and this drops them for free.
    )

    md_files = sorted(out.rglob("*.md"))
    if not md_files:
        raise gr.Error(
            "OpenDataLoader produced no markdown. A scanned PDF has no text "
            "layer to read — use one of the OCR backends for that."
        )
    text = md_files[0].read_text(encoding="utf-8", errors="replace")

    n = 0
    extras = run / "extras"
    for src in out.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(src, run / "images" / src.name)
            n += 1
        elif src.suffix.lower() not in {".md", ".html"}:
            # json / text / pdf / tagged-pdf go into the bundle as-is; md and
            # html are rebuilt below from the rewritten copies.
            extras.mkdir(exist_ok=True)
            shutil.copy2(src, extras / src.name)

    def fix(m: re.Match) -> str:
        return f"![{m.group(1)}](images/{Path(m.group(2)).name})"

    # Its HTML references the same stem_images/ directory, via src="…" rather
    # than markdown syntax, so it needs its own rewrite.
    html_files = sorted(out.rglob("*.html"))
    html_body = None
    if html_files:
        raw = html_files[0].read_text(encoding="utf-8", errors="replace")
        html_body = re.sub(
            r'(src\s*=\s*["\'])([^"\']+)(["\'])',
            lambda m: f"{m.group(1)}images/{Path(m.group(2)).name}{m.group(3)}",
            raw,
        )

    return _ODL_IMG.sub(fix, text), n, html_body


def _local_pdf_path(file_path: str | None, source_url: str | None, run: Path) -> str:
    """OpenDataLoader reads local files only, so a URL is fetched here — there is
    no Modal in this path to fetch it for us."""
    if source_url and source_url.strip():
        raw = requests.get(source_url.strip(), timeout=300,
                           headers={"User-Agent": "Mozilla/5.0"}).content
        target = run / (Path(source_url.split("?", 1)[0]).name or "input.pdf")
        target.write_bytes(raw)
        return str(target)
    if not file_path:
        raise gr.Error("Upload a PDF or paste a URL first.")
    return file_path


def _resolve(name: str, route: str) -> tuple[dict, str]:
    cfg = BACKENDS[name]
    if not cfg["base"]:
        raise gr.Error(f"{name} is not configured — set {cfg['env']}.")
    if not cfg[route]:
        raise gr.Error(f"{name} has no /{route} endpoint.")
    return cfg, cfg["base"] + cfg[route]


# --- document parsing --------------------------------------------------------
def run_parse(
    file_path: str | None,
    source_url: str | None,
    backend: str,
    tokens: int,
    conf: float,
    table_format: str,
    mode_label: str,
    ngram_on: bool,
    prompt: str,
    odl_table: str = "default",
    odl_struct: bool = False,
    odl_sanitize: bool = False,
    odl_breaks: bool = False,
    odl_formats: list[str] | None = None,
):
    """Yields (preview, source, status, zip_path, run_dir). Streams NDJSON."""
    if BACKENDS[backend]["family"] == "local":
        if not ODL_OK:
            raise gr.Error(f"OpenDataLoader unavailable — {ODL_WHY}.")
        run = _new_run_dir()
        t0 = time.time()
        yield "", "", "_reading the PDF's own text and structure…_", None, str(run)

        pdf = _local_pdf_path(file_path, source_url, run)
        source, n_img, odl_html = _odl_run(pdf, run, odl_struct, odl_sanitize,
                                           odl_breaks, odl_table, odl_formats)
        bundle = _build_zip(run, source, Path(pdf).name, odl_html)
        how = "deterministic parse, no GPU"
        if odl_struct:
            how += " · tag tree"
        yield (
            _inline_images(source, run), source,
            f"done — {how}"
            + (f" · {n_img} images" if n_img else "")
            + f" · wall **{time.time() - t0:.1f} s**",
            bundle, str(run),
        )
        return

    cfg, url = _resolve(backend, "parse")
    payload = _payload_for_source(file_path, source_url)
    via_url = "url" in next(iter(payload))

    run = _new_run_dir()

    # Emit before touching the network. The first server event only arrives after
    # upload + rasterise + layout detection, which on a 27-page PDF measured over
    # 90 s of blank screen — indistinguishable from a hung UI.
    t0 = time.time()
    yield (
        "", "",
        f"_sending to **{backend}** — "
        + ("Modal is fetching the URL…_" if via_url else "upload and page setup…_"),
        None, str(run),
    )
    if cfg["family"] == "paddle":
        # ponytail: official PaddleOCR treats a whole table as one block, so keep
        # a 4096-token ceiling; expose per-table budgeting if larger tables need it.
        official_tokens = 4096 if backend == "PaddleOCR-VL · GGUF official" else int(tokens)
        payload |= {
            "max_new_tokens": official_tokens,
            "conf": float(conf),
            "table_format": table_format,
        }
    elif cfg["family"] == "unlimited":
        payload["mode"] = MODE_MAP.get(mode_label, "gundam")
        if prompt.strip():
            payload["prompt"] = prompt.strip()
        if not ngram_on:
            payload |= {"no_repeat_ngram_size": 0, "ngram_window": 0}

    content, completed = "", False
    try:
        with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                ev = json.loads(line)
                completed = bool(ev.get("done"))
                wall = time.time() - t0
                _write_images(run, ev.get("new_images"))
                if ev.get("done") and ev.get("official_results"):
                    extras = run / "extras"
                    extras.mkdir(exist_ok=True)
                    (extras / "paddleocr-official.json").write_text(
                        json.dumps(ev["official_results"], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                if cfg["family"] == "paddle":
                    content = ev.get("markdown", "")
                    total, prog = ev.get("total", 0), ev.get("progress", 0)
                    pages = ev.get("pages", 1)
                    where = f" across {pages} pages" if pages > 1 else ""
                    # Client wall includes upload; the server figures don't.
                    # Showing both is the point — the gap is the upload.
                    timing = (
                        f"wall **{wall:.1f} s** · detect {ev.get('detect_s', '?')} s"
                        f" · recognise {ev.get('recognise_s', 0)} s"
                    )
                    if prog:
                        timing += f" · {ev.get('recognise_s', 0) / prog:.2f} s/block"
                    figs = ev.get("figure_count") or 0
                    status = (
                        f"done — {total} blocks{where}"
                        + (f" · {figs} figures" if figs else "")
                        + f" · {timing}"
                        if ev.get("done")
                        else f"{total} blocks{where} — {prog}/{total} · {timing}"
                    )
                else:
                    content = ev.get("text", "")
                    pages = ev.get("pages", 1)
                    where = f" · {pages} pages" if pages > 1 else ""
                    status = (
                        f"done · wall **{wall:.1f} s**{where}"
                        if ev.get("done")
                        else f"streaming{where} · {len(content)} chars · wall {wall:.1f} s"
                    )

                # Preview gets data URIs so images render with no file server;
                # Source keeps images/NAME so the ZIP resolves offline. The ZIP
                # is only built once, on the final event.
                name = (
                    Path((source_url or "").split("?", 1)[0]).name
                    if via_url else Path(file_path).name
                )
                if ev.get("done") and cfg["family"] == "mineru":
                    try:
                        _mineru_artifacts(
                            run, file_path, source_url, ev.get("blocks") or [],
                            next(iter(payload)).startswith("pdf_"),
                        )
                    except Exception as artifact_error:
                        status += f" · artifact warning: {artifact_error}"
                bundle = _build_zip(run, content, name) if ev.get("done") else None
                yield _inline_images(content, run), content, status, bundle, str(run)
        if not completed:
            shutil.rmtree(run, ignore_errors=True)
            raise gr.Error(f"{backend} closed the stream before sending done=true.")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Request to {url} failed: {e}")


# --- single-image recognition ------------------------------------------------
def run_recognize(file_path: str | None, backend: str, label: str, tokens: int):
    """Yields (preview, raw, status).

    The two PaddleOCR-VL deployments answer differently — the GGUF one returns a
    single JSON object, the transformers one streams NDJSON — so read line by
    line and treat the last complete object as current either way.
    """
    if not file_path:
        raise gr.Error("Upload an image first.")
    cfg, url = _resolve(backend, "recognize")
    task = TASK_BY_LABEL[label]

    t0 = time.time()
    yield "", "", f"_sending to **{backend}**…_"

    payload = _payload_for_file(file_path) | {
        "task": task,
        "max_new_tokens": int(tokens),
    }

    text, data = "", {}
    try:
        with requests.post(url, json=payload, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                text = data.get("text") or data.get("markdown", "")
                if not data.get("done"):
                    yield (
                        _render(task, text),
                        text,
                        f"streaming · {len(text)} chars · wall {time.time() - t0:.1f} s",
                    )
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Request to {url} failed: {e}")

    stats = f"wall **{time.time() - t0:.1f} s**"
    if data.get("elapsed_s") is not None:
        stats += f" · generation **{data['elapsed_s']} s**"
    if data.get("tokens_per_s"):
        stats += f" · **{data['tokens_per_s']} tok/s**"
    if data.get("completion_tokens"):
        stats += f" · {data['completion_tokens']} tokens"
    yield _render(task, text), text, stats


# --- side-by-side ------------------------------------------------------------
def run_compare(file_path: str | None, source_url: str | None,
                backends: list[str], tokens: int, conf: float):
    """Yields (pane_a, pane_b, pane_c, status).

    Runs the selected backends one after another rather than concurrently: they
    are separate Modal apps on separate GPUs, so running them at once would make
    the timings measure contention for this machine's uplink instead of the
    models.
    """
    if not backends:
        raise gr.Error("Pick at least one backend.")
    _payload_for_source(file_path, source_url)  # fail fast if neither is set

    panes = ["", "", ""]
    rows = ["| Backend | Wall | Chars |", "| --- | --- | --- |"]
    for slot, name in enumerate(backends[:3]):
        last, raw, run_dir, t0 = "", "", None, time.time()
        try:
            for preview, source, _, _, generated_run in run_parse(
                file_path, source_url, name, tokens, conf, "html", "Long", True, ""
            ):
                last, raw, run_dir = preview, source, generated_run
                panes[slot] = preview
                yield (*panes, "\n".join(rows) + f"\n\n_running **{name}**…_")
        except gr.Error as e:
            panes[slot] = f"**failed:** {e}"
            rows.append(f"| {name} | — | failed |")
            yield (*panes, "\n".join(rows))
            continue
        rows.append(f"| {name} | {time.time() - t0:.1f} s | {len(raw)} |")
        if run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
        yield (*panes, "\n".join(rows))

    yield (*panes, "\n".join(rows) + "\n\n_done._")


# --- destructive actions -----------------------------------------------------
# Both of these throw away a result that can cost minutes of GPU time to
# reproduce, so neither fires on a single click. The confirm row is plain Gradio
# state rather than a browser confirm() via `js=`: an aborted js handler
# surfaces as an error toast, which reads like a failure rather than a
# cancellation.
WARNINGS = {
    "run": "⚠️ **Ganti hasil?** Inferensi baru akan menimpa hasil yang sekarang, "
           "termasuk gambar yang sudah diekstrak. Unduh ZIP-nya dulu kalau masih "
           "dibutuhkan.",
    "clear": "⚠️ **Hapus hasil?** Teks, gambar hasil ekstraksi, dan ZIP-nya akan "
             "dibuang. Tidak bisa dikembalikan.",
}


def _ask(kind: str, current_markdown: str | None):
    """Returns (warn, confirm_row, pending, go).

    With nothing on screen there is nothing to lose, so the first parse of a
    session runs on one click; the prompt only appears once a result exists.
    """
    if not (current_markdown or "").strip():
        return gr.update(visible=False), gr.update(visible=False), None, True
    return (
        gr.update(visible=True, value=WARNINGS[kind]),
        gr.update(visible=True),
        kind,
        False,
    )


# --- API guide ---------------------------------------------------------------
def _api_snippets(backend: str):
    """Snippets for whichever deployment is selected, with its real base URL."""
    cfg = BACKENDS[backend]
    base = cfg["base"] or f"https://<{cfg['env']}>"
    parse = base + cfg["parse"]
    paddle = cfg["family"] == "paddle"

    body = (
        '{"image_base64":"'"'"'"$B64"'"'"'","max_new_tokens":512,'
        '"conf":0.25,"table_format":"html"}'
        if paddle else
        '{"image_base64":"'"'"'"$B64"'"'"'","mode":"gundam"}'
    )
    curl = f"""# health — cheap, and wakes the container
curl -s {base}/health

# parse a page or PDF. NDJSON: one JSON object per line, last one has done:true
B64=$(base64 -w0 page.jpg)          # macOS: base64 -i page.jpg
curl -N -s -X POST {parse} \\
  -H 'Content-Type: application/json' \\
  -d '{body}'

# a PDF goes in the same way, under a different key
B64=$(base64 -w0 doc.pdf)
curl -N -s -X POST {parse} \\
  -H 'Content-Type: application/json' \\
  -d '{{"pdf_base64":"'"$B64"'","pdf_dpi":200}}'

# skip the upload entirely when the file is already reachable by URL —
# Modal fetches it over datacentre network instead of your uplink
curl -N -s -X POST {parse} \\
  -H 'Content-Type: application/json' \\
  -d '{{"image_url":"https://example.com/page.jpg"}}'
"""
    if paddle:
        curl += f"""
# single image, single task (ocr|table|formula|chart|spotting|seal)
curl -N -s -X POST {base}/recognize \\
  -H 'Content-Type: application/json' \\
  -d '{{"image_url":"https://example.com/table.jpg","task":"table"}}'
"""

    extra = (
        '"max_new_tokens": 512, "conf": 0.25, "table_format": "html",'
        if paddle else '"mode": "gundam",'
    )
    key_note = (
        "images/pN_bX_bY.jpg — write them next to the markdown and it resolves"
        if paddle else "plain text, no figure extraction"
    )
    py = f'''import base64, json, requests

BASE = "{base}"
path = "page.jpg"                      # or "doc.pdf"
key = "pdf_base64" if path.endswith(".pdf") else "image_base64"

payload = {{
    key: base64.b64encode(open(path, "rb").read()).decode(),
    {extra}
}}

images = {{}}                            # {key_note}
markdown = ""

with requests.post(f"{{BASE}}{cfg['parse']}", json=payload,
                   stream=True, timeout=1800) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        ev = json.loads(line)
        images.update(ev.get("new_images") or {{}})
        markdown = ev.get("markdown") or ev.get("text") or markdown
        if not ev.get("done"):
            print(f"{{ev.get('progress', 0)}}/{{ev.get('total', 0)}}", end="\\r")

print(markdown)

# figure crops arrive once each, base64, under new_images
import pathlib
out = pathlib.Path("out"); (out / "images").mkdir(parents=True, exist_ok=True)
(out / "document.md").write_text(markdown, encoding="utf-8")
for name, b64 in images.items():
    (out / "images" / name).write_bytes(base64.b64decode(b64))
'''

    js = f'''const BASE = "{base}";

const file = document.querySelector("input[type=file]").files[0];
const b64 = await new Promise(res => {{
  const fr = new FileReader();
  fr.onload = () => res(fr.result.split(",")[1]);
  fr.readAsDataURL(file);
}});

const resp = await fetch(BASE + "{cfg['parse']}", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{
    [file.name.endsWith(".pdf") ? "pdf_base64" : "image_base64"]: b64,
    {"max_new_tokens: 512, conf: 0.25, table_format: \"html\"," if paddle else "mode: \"gundam\","}
  }}),
}});

// NDJSON, so split on newlines and keep the trailing partial line
const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
let buf = "", markdown = "";
for (;;) {{
  const {{ value, done }} = await reader.read();
  if (done) break;
  buf += value;
  const lines = buf.split("\\n");
  buf = lines.pop();                    // incomplete line, wait for more
  for (const line of lines) {{
    if (!line.trim()) continue;
    const ev = JSON.parse(line);
    markdown = ev.markdown ?? ev.text ?? markdown;
  }}
}}
console.log(markdown);
'''

    if paddle:
        fields = """| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_base64` / `image_url` | str | — | one image |
| `pdf_base64` / `pdf_url` | str | — | a PDF, rasterised server-side |
| `pdf_dpi` | int | 200 | higher costs time on every later step |
| `conf` | float | 0.25 | layout-detection threshold |
| `iou` | float | 0.45 | NMS overlap |
| `max_new_tokens` | int | 512 | **per block**; tables get 6×, charts 4×, formulas 2× |
| `table_format` | str | `html` | `html` keeps colspan — safest for LLM context. `markdown` is ~40% fewer characters but has no colspan, so merged cells collapse to blanks. `otsl` is the model's raw output |

**Response objects**

| Key | Notes |
| --- | --- |
| `markdown` | assembled document so far, in reading order |
| `blocks` | every finished block: `bbox`, `type`, `text`, `truncated` |
| `new_images` | `{name: base64 jpeg}` — figure crops, **sent once each**, not repeated |
| `total` / `progress` | block counts |
| `detect_s` / `recognise_s` | server-side timings; the gap against your own wall clock is upload |
| `truncated_blocks` | blocks that hit the token cap — their content is incomplete |
| `done` | `true` on the final object only |

**`/recognize`** takes `image_base64`/`image_url`, `task`, `max_new_tokens` and streams the
text as it is generated, ending with `elapsed_s`, `completion_tokens`, `tokens_per_s`.

Tasks: `ocr`, `table`, `formula`, `chart`, `spotting`, `seal`. `table` returns **OTSL**
(`<fcel>`/`<lcel>`/`<ecel>`/`<nl>`), not HTML — a browser silently drops those tags, so
convert before rendering.
"""
    else:
        fields = """| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_base64` / `image_url` | str | — | one image |
| `images_base64` / `image_urls` | list | — | several pages |
| `pdf_base64` / `pdf_url` | str | — | rasterised server-side |
| `pdf_dpi` | int | 200 | |
| `prompt` | str | model default | free text — this is how behaviour is steered; there is no task list |
| `mode` | str | `gundam` | `gundam` (Long) or `base` |
| `max_length` | int | 8192 | |
| `no_repeat_ngram_size` | int | 35 | set to 0 with `ngram_window: 0` to disable the repeat guard |

**Response objects**: `text` (accumulated so far), `pages`, `done`.

No layout detection and no figure extraction — output is one text stream per page.
`/explode-pdf` splits a PDF into page PNGs on CPU, without touching the GPU.
"""

    return curl, py, js, fields


# --- UI ----------------------------------------------------------------------
EXTRAS_MAX_CHARS = 400_000


def _show_extras(run_dir: str | None):
    """Preview whatever landed in extras/, reading from the run directory
    instead of threading a sixth value through every yield in run_parse."""
    run = Path(run_dir) if run_dir else None
    extras = sorted((run / "extras").glob("*")) if run else []
    files = (
        [path for path in (run / "document.md", run / "document.html") if path.exists()]
        + sorted((run / "images").glob("*"))
        + extras
        if run else []
    )
    if not files:
        return "_no generated files yet_", "", None

    listing = " · ".join(f"`{f.name}` ({f.stat().st_size / 1024:.0f} KB)"
                          for f in files)
    # Only json and text are worth showing as text; pdf and tagged-pdf are
    # binary and live in the ZIP.
    readable = next((f for f in extras if f.suffix.lower() in {".json", ".txt"}), None)
    if not readable:
        return listing, "", [str(f) for f in files]

    body = readable.read_text(encoding="utf-8", errors="replace")
    if len(body) > EXTRAS_MAX_CHARS:
        body = body[:EXTRAS_MAX_CHARS] + "\n… truncated; full file is in the ZIP"
    return f"{listing} — showing `{readable.name}`", body, [str(f) for f in files]


def _toggle_opts(backend: str):
    fam = BACKENDS[backend]["family"]
    return (
        gr.update(visible=fam == "paddle"),
        gr.update(visible=fam == "unlimited"),
        gr.update(visible=fam == "local"),
    )


_configured = [(n, c) for n, c in BACKENDS.items() if c["base"]]
_missing = [f"`{c['env']}`" for n, c in BACKENDS.items() if not c["base"]]

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OCR Suite") as demo:
    gr.Markdown("# OCR Suite — PaddleOCR-VL · GGUF · Unlimited-OCR")
    with gr.Accordion(f"Configured backends ({len(_configured)})", open=False):
        gr.Markdown(
            "\n".join([
                "| Model | URL | Source |",
                "| --- | --- | --- |",
                *(
                    f"| **{n}** | `{c['base']}` | "
                    f"[{'HF' if 'huggingface.co' in c['repo'] else 'GitHub'}]({c['repo']}) |"
                    for n, c in _configured
                ),
            ])
            or "_No backend configured._"
        )
        if _missing:
            gr.Markdown(f"_Unconfigured: {', '.join(_missing)}_")

    with gr.Tabs():
        # ---------------------------------------------------------------- parse
        with gr.Tab("Document Parsing"):
            gr.Markdown(
                "_Whole pages and PDFs. The custom PaddleOCR-VL backends run "
                "DocLayout-YOLO; the official variant runs PaddleOCR's PP-DocLayoutV3. "
                "Unlimited-OCR parses each page in one pass. Output streams as it is produced._"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    backend_p = gr.Dropdown(
                        PARSE_CHOICES,
                        value=PARSE_CHOICES[0],
                        label="Model",
                        info="Which deployment handles this document",
                    )
                    file_p = gr.File(
                        label="Upload page or PDF",
                        file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf"],
                        type="filepath",
                    )
                    url_p = gr.Textbox(
                        label="…or a URL",
                        placeholder="https://example.com/report.pdf",
                        info=(
                            "Takes precedence over the upload. Modal fetches it "
                            "over datacentre network, which removes the base64 "
                            "upload — on a 27-page PDF that gap measured 93 s."
                        ),
                    )
                    with gr.Row():
                        prev_file_b = gr.Button("Preview upload", size="sm")
                        prev_url_b = gr.Button("Preview URL", size="sm")
                    with gr.Group(visible=True) as paddle_opts:
                        tokens_p = gr.Slider(
                            128, 4096, value=512, step=128,
                            label="Max new tokens per block",
                        )
                        conf_p = gr.Slider(
                            0.05, 0.9, value=0.25, step=0.05, label="Layout confidence"
                        )
                        fmt_p = gr.Radio(
                            ["html", "markdown", "otsl"],
                            value="html",
                            label="Table format",
                            info=(
                                "html keeps merged cells (colspan) — safest for LLM "
                                "context. markdown is ~40% fewer characters but has no "
                                "colspan, so merged cells collapse to blanks. otsl is "
                                "the model's raw output: most compact, but LLMs don't "
                                "know the format."
                            ),
                        )
                    with gr.Group(visible=False) as odl_opts:
                        odl_table_c = gr.Radio(
                            ["default", "cluster"], value="default",
                            label="Table method",
                            info=(
                                "Both read ruled tables exactly. Measured on a "
                                "borderless table, neither produced a table at "
                                "all and cluster was slower — the clustering is "
                                "over border fragments, not text columns. For "
                                "borderless tables use PaddleOCR-VL."
                            ),
                        )
                        odl_struct_c = gr.Checkbox(
                            value=False, label="Use PDF's own tag tree",
                            info="Trust the document's native structure where it "
                                 "has one; falls back to layout analysis when it "
                                 "does not.",
                        )
                        odl_sanitize_c = gr.Checkbox(
                            value=False, label="Sanitize",
                            info="Filters content-safety risks, including prompt "
                                 "injection — worth turning on when the output "
                                 "feeds an LLM.",
                        )
                        odl_breaks_c = gr.Checkbox(
                            value=False, label="Keep line breaks",
                        )
                        odl_formats_c = gr.CheckboxGroup(
                            ["json", "text", "pdf", "tagged-pdf"],
                            value=["json"],
                            label="Extra formats in the ZIP",
                            info=(
                                "markdown and html are always produced — the "
                                "panes and the bundle need them. These land in "
                                "extras/. json is the only one that carries "
                                "table cell coordinates."
                            ),
                        )
                    with gr.Group(visible=False) as unlimited_opts:
                        mode_p = gr.Radio(
                            ["Long", "Base"], value="Long", label="Mode"
                        )
                        ngram_p = gr.Checkbox(value=True, label="NGRAM repeat guard")
                        prompt_p = gr.Textbox(
                            label="Prompt (optional)",
                            placeholder="Leave empty for the model's default parsing prompt",
                            lines=2,
                        )
                    with gr.Row():
                        btn_p = gr.Button("Parse Document", variant="primary")
                        clear_p = gr.Button("Hapus hasil", variant="stop")
                with gr.Column(scale=7):
                    warn_p = gr.Markdown(visible=False)
                    with gr.Row(visible=False) as confirm_row:
                        yes_p = gr.Button("Lanjutkan", variant="stop", size="sm")
                        no_p = gr.Button("Batal", size="sm")
                    # Tab ids so events can switch panes. Without this the source
                    # gallery filled up behind whichever tab was in front and
                    # pressing Preview looked like it did nothing at all.
                    with gr.Tabs() as out_tabs:
                        with gr.Tab("Preview", id="pane-out"):
                            fs_p = gr.Button("⛶ Fullscreen", size="sm",
                                             elem_classes="fs-btn")
                            md_p = gr.Markdown(
                                latex_delimiters=LATEX_DELIMS,
                                max_height=PANE_MAX_H,
                                elem_classes="ocr-pane",
                                elem_id="pane-parse",
                            )
                        with gr.Tab("Source", id="pane-src"):
                            raw_p = gr.Code(
                                language="markdown", lines=26, max_lines=26
                            )
                        with gr.Tab("Extras", id="pane-extras"):
                            extras_note = gr.Markdown("_nothing yet_")
                            extras_code = gr.Code(language="json", lines=24,
                                                  max_lines=24)
                            extras_files = gr.File(
                                label="Generated files", file_count="multiple",
                                interactive=False,
                            )
                        with gr.Tab("Input document", id="pane-input"):
                            src_note = gr.Markdown("_nothing loaded yet_")
                            src_gallery = gr.Gallery(
                                label="Input pages", columns=3, height="60vh",
                                object_fit="contain", preview=True,
                                elem_id="src-gallery",
                            )
                    # Under the preview, not beside the controls: both only mean
                    # anything once there is a result, and the result is here.
                    status_p = gr.Markdown("")
                    zip_p = gr.File(
                        label="Download ZIP (MD + HTML + JSON + bbox + images)",
                        interactive=False,
                    )

            # run_dir survives between events so "Hapus hasil" knows what to
            # delete; pending holds which action the confirm row is asking about.
            run_dir_st = gr.State(None)
            pending_st = gr.State(None)

            fs_p.click(fn=None, js=_fullscreen_js("pane-parse"))
            backend_p.change(
                _toggle_opts, inputs=backend_p,
                outputs=[paddle_opts, unlimited_opts, odl_opts],
            )

            # Uploading a file previews it straight away — the bytes are already
            # local, so it costs nothing. A URL does not: fetching it here would
            # undo the very saving the URL path exists for, so it waits for the
            # button.
            def _show_input():
                return gr.Tabs(selected="pane-input")

            def _show_output():
                return gr.Tabs(selected="pane-out")

            file_p.change(preview_file, inputs=file_p,
                          outputs=[src_gallery, src_note]).then(
                _show_input, outputs=out_tabs)
            prev_file_b.click(preview_file, inputs=file_p,
                              outputs=[src_gallery, src_note]).then(
                _show_input, outputs=out_tabs)
            prev_url_b.click(preview_url, inputs=url_p,
                             outputs=[src_gallery, src_note]).then(
                _show_input, outputs=out_tabs)

            PARSE_IN = [file_p, url_p, backend_p, tokens_p, conf_p, fmt_p,
                        mode_p, ngram_p, prompt_p,
                        odl_table_c, odl_struct_c, odl_sanitize_c, odl_breaks_c,
                        odl_formats_c]
            PARSE_OUT = [md_p, raw_p, status_p, zip_p, run_dir_st]

            # Two flags rather than one, because "run" and "clear" write to
            # different output sets and Gradio wires outputs per handler.
            go_run_st = gr.State(False)
            go_clear_st = gr.State(False)

            # Both work functions are generators so that "not confirmed" can be
            # expressed as `return` — yielding nothing leaves every output as it
            # was. Returning None from a plain function would instead try to
            # unpack None across five outputs.
            def do_run(go, *args):
                if go:
                    yield from run_parse(*args)

            def do_clear(go, run_dir):
                if not go:
                    return
                if run_dir:
                    shutil.rmtree(run_dir, ignore_errors=True)
                yield "", "", "", None, None

            btn_p.click(
                lambda md: _ask("run", md), inputs=md_p,
                outputs=[warn_p, confirm_row, pending_st, go_run_st],
            ).then(
                # Come back to the output pane, otherwise a parse started right
                # after a preview streams into a tab nobody is looking at.
                _show_output, outputs=out_tabs
            ).then(
                do_run, inputs=[go_run_st, *PARSE_IN], outputs=PARSE_OUT
            ).then(
                _show_extras, inputs=run_dir_st,
                outputs=[extras_note, extras_code, extras_files]
            )

            clear_event = clear_p.click(
                lambda md: _ask("clear", md), inputs=md_p,
                outputs=[warn_p, confirm_row, pending_st, go_clear_st],
            )
            clear_event.then(
                do_clear, inputs=[go_clear_st, run_dir_st], outputs=PARSE_OUT
            ).then(
                _show_extras, inputs=run_dir_st,
                outputs=[extras_note, extras_code, extras_files]
            )

            # Confirmed: turn the pending action into whichever flag it was, and
            # let both .then() chains fire — the one whose flag is False is a
            # no-op.
            yes_p.click(
                lambda pending: (
                    gr.update(visible=False), gr.update(visible=False),
                    pending == "run", pending == "clear",
                ),
                inputs=pending_st,
                outputs=[warn_p, confirm_row, go_run_st, go_clear_st],
            ).then(
                do_run, inputs=[go_run_st, *PARSE_IN], outputs=PARSE_OUT
            ).then(
                do_clear, inputs=[go_clear_st, run_dir_st], outputs=PARSE_OUT
            ).then(
                _show_extras, inputs=run_dir_st,
                outputs=[extras_note, extras_code, extras_files]
            )

            no_p.click(
                lambda: (gr.update(visible=False), gr.update(visible=False), False, False),
                outputs=[warn_p, confirm_row, go_run_st, go_clear_st],
            )

        # ------------------------------------------------------------ recognize
        with gr.Tab("Recognize"):
            gr.Markdown(
                "_One image, one task, no layout detection. PaddleOCR-VL only — "
                "Unlimited-OCR has no task selector._"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    backend_a = gr.Dropdown(
                        RECOGNIZE_CHOICES or ["(none configured)"],
                        value=(RECOGNIZE_CHOICES or ["(none configured)"])[0],
                        label="Model",
                    )
                    file_a = gr.Image(label="Upload image", type="filepath", height=320)
                    task_a = gr.Radio(
                        list(TASK_BY_LABEL), value="Text Recognition", label="Task"
                    )
                    tokens_a = gr.Slider(
                        128, 4096, value=1024, step=128, label="Max new tokens"
                    )
                    btn_a = gr.Button("Run", variant="primary")
                    stats_a = gr.Markdown("")
                with gr.Column(scale=7):
                    with gr.Tabs():
                        with gr.Tab("Result"):
                            fs_a = gr.Button("⛶ Fullscreen", size="sm",
                                             elem_classes="fs-btn")
                            md_a = gr.Markdown(
                                latex_delimiters=LATEX_DELIMS,
                                max_height=PANE_MAX_H,
                                elem_classes="ocr-pane",
                                elem_id="pane-recognize",
                            )
                        with gr.Tab("Raw Output"):
                            raw_a = gr.Code(
                                language="markdown", lines=26, max_lines=26
                            )

            fs_a.click(fn=None, js=_fullscreen_js("pane-recognize"))
            btn_a.click(
                run_recognize,
                inputs=[file_a, backend_a, task_a, tokens_a],
                outputs=[md_a, raw_a, stats_a],
            )

        # -------------------------------------------------------------- compare
        with gr.Tab("Compare"):
            gr.Markdown(
                "_The same document through several backends, one after another. "
                "This is the tab for the question the benchmarks never answered: "
                "whether GGUF's speed costs accuracy on tables and formulas._"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    file_c = gr.File(
                        label="Upload page or PDF",
                        file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf"],
                        type="filepath",
                    )
                    url_c = gr.Textbox(
                        label="…or a URL",
                        placeholder="https://example.com/report.pdf",
                        info="Takes precedence over the upload.",
                    )
                    backends_c = gr.Dropdown(
                        PARSE_CHOICES,
                        value=PARSE_CHOICES[:2],
                        multiselect=True,
                        max_choices=3,
                        label="Models",
                        info="Pick up to 3; they run one after another",
                    )
                    tokens_c = gr.Slider(
                        128, 4096, value=512, step=128, label="Max new tokens per block"
                    )
                    conf_c = gr.Slider(
                        0.05, 0.9, value=0.25, step=0.05, label="Layout confidence"
                    )
                    btn_c = gr.Button("Compare", variant="primary")
                with gr.Column(scale=7):
                    status_c = gr.Markdown("")
            with gr.Row():
                fs_c = [
                    gr.Button(f"⛶ Pane {i + 1}", size="sm", elem_classes="fs-btn")
                    for i in range(3)
                ]
            with gr.Row():
                panes_c = [
                    gr.Markdown(
                        latex_delimiters=LATEX_DELIMS,
                        max_height=PANE_MAX_H,
                        elem_classes="ocr-pane",
                        elem_id=f"pane-compare-{i}",
                    )
                    for i in range(3)
                ]

            for i, btn in enumerate(fs_c):
                btn.click(fn=None, js=_fullscreen_js(f"pane-compare-{i}"))

            btn_c.click(
                run_compare,
                inputs=[file_c, url_c, backends_c, tokens_c, conf_c],
                outputs=[*panes_c, status_c],
            )

        # ------------------------------------------------------------------ api
        with gr.Tab("API"):
            gr.Markdown(
                "_Call the deployments directly, without this UI. Snippets are "
                "filled in with the base URLs this instance is configured with, "
                "so they are copy-pasteable as-is._"
            )
            api_backend = gr.Dropdown(
                PARSE_CHOICES, value=PARSE_CHOICES[0], label="Model"
            )
            gr.Markdown(
                "Every endpoint takes **POST + JSON** and answers "
                "**`application/x-ndjson`** — one complete JSON object per line, "
                "emitted as work finishes. Read it line by line; the last object "
                "carries `\"done\": true`. It is *not* SSE: the request body holds "
                "megabytes of base64, and `EventSource` cannot do POST."
            )
            with gr.Tabs():
                with gr.Tab("curl"):
                    api_curl = gr.Code(language="shell", lines=22)
                with gr.Tab("Python"):
                    api_py = gr.Code(language="python", lines=30)
                with gr.Tab("JavaScript"):
                    api_js = gr.Code(language="javascript", lines=28)
                with gr.Tab("Fields"):
                    api_fields = gr.Markdown()

            api_backend.change(
                _api_snippets, inputs=api_backend,
                outputs=[api_curl, api_py, api_js, api_fields],
            )
            demo.load(
                _api_snippets, inputs=api_backend,
                outputs=[api_curl, api_py, api_js, api_fields],
            )

if __name__ == "__main__":
    demo.queue().launch()
