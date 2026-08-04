# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=5.0,<6",
#     "requests>=2,<3",
#     "Pillow>=10",
# ]
# ///
"""One Gradio client for all three OCR deployments in this folder.

Loads no model — every tab is an HTTP call to a deployed Modal app. Each backend
is selected per tab, so the same document can be pushed through any of them
without restarting the UI:

    PADDLEOCR_VL_GGUF_BASE_URL   modal_paddleocr_vl_gguf.py   llama.cpp, fastest
    PADDLEOCR_VL_BASE_URL        modal_paddleocr_vl.py        transformers
    UNLIMITED_OCR_BASE_URL       modal_unlimited_ocr.py       DeepSeek-OCR-style VLM

Only the backends whose env var is set are selectable; the rest are listed as
unconfigured rather than failing at click time.

Run with:
    PADDLEOCR_VL_GGUF_BASE_URL=https://… \
    PADDLEOCR_VL_BASE_URL=https://… \
    UNLIMITED_OCR_BASE_URL=https://… \
    uv run gradio_app.py
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import requests

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
    },
    "PaddleOCR-VL · transformers": {
        "env": "PADDLEOCR_VL_BASE_URL",
        "parse": "/parse-page",
        "recognize": "/recognize",
        "family": "paddle",
    },
    "Unlimited-OCR": {
        "env": "UNLIMITED_OCR_BASE_URL",
        "parse": "/parse",
        "recognize": None,
        "family": "unlimited",
    },
}
for _cfg in BACKENDS.values():
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
    """The `table` task returns OTSL markup, not HTML or Markdown. A browser
    silently drops the unknown tags, so feeding it to gr.Markdown renders as
    garbled text. /parse-page already converts server-side; /recognize does not,
    which is why this still lives here."""
    rows = []
    for raw_row in otsl.split("<nl>"):
        tokens = _OTSL_TOKEN.findall(raw_row)
        if not tokens:
            continue
        cells: list[list] = []
        for tag, text in tokens:
            if tag in ("lcel", "xcel") and cells:
                cells[-1][1] += 1
            elif tag in ("ecel", "ucel"):
                cells.append(["", 1])
            else:
                cells.append([text.strip(), 1])
        rows.append(cells)
    if not rows:
        return ""
    out = ['<table border="1" style="border-collapse:collapse">']
    for cells in rows:
        out.append("<tr>")
        for text, span in cells:
            attr = f' colspan="{span}"' if span > 1 else ""
            out.append(f"<td{attr}>{html.escape(text)}</td>")
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
    backend: str,
    tokens: int,
    conf: float,
    table_format: str,
    mode_label: str,
    ngram_on: bool,
    prompt: str,
):
    """Yields (preview, source, status). Streams NDJSON from whichever backend."""
    if not file_path:
        raise gr.Error("Upload an image or PDF first.")
    cfg, url = _resolve(backend, "parse")

    # Emit before touching the network. The first server event only arrives after
    # upload + rasterise + layout detection, which on a 27-page PDF measured over
    # 90 s of blank screen — indistinguishable from a hung UI.
    t0 = time.time()
    yield "", "", f"_sending to **{backend}** — upload and page setup…_"

    payload = _payload_for_file(file_path)
    if cfg["family"] == "paddle":
        payload |= {
            "max_new_tokens": int(tokens),
            "conf": float(conf),
            "table_format": table_format,
        }
    else:
        payload["mode"] = MODE_MAP.get(mode_label, "gundam")
        if prompt.strip():
            payload["prompt"] = prompt.strip()
        if not ngram_on:
            payload |= {"no_repeat_ngram_size": 0, "ngram_window": 0}

    content = ""
    try:
        with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                ev = json.loads(line)
                wall = time.time() - t0

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
                    status = (
                        f"done — {total} blocks{where} · {timing}"
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

                yield content, content, status
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
                text = data.get("text", "")
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
def run_compare(file_path: str | None, backends: list[str], tokens: int, conf: float):
    """Yields (pane_a, pane_b, pane_c, status).

    Runs the selected backends one after another rather than concurrently: they
    are separate Modal apps on separate GPUs, so running them at once would make
    the timings measure contention for this machine's uplink instead of the
    models.
    """
    if not file_path:
        raise gr.Error("Upload an image or PDF first.")
    if not backends:
        raise gr.Error("Pick at least one backend.")

    panes = ["", "", ""]
    rows = ["| Backend | Wall | Chars |", "| --- | --- | --- |"]
    for slot, name in enumerate(backends[:3]):
        last, t0 = "", time.time()
        try:
            for content, _, _ in run_parse(
                file_path, name, tokens, conf, "html", "Long", True, ""
            ):
                last = content
                panes[slot] = content
                yield (*panes, "\n".join(rows) + f"\n\n_running **{name}**…_")
        except gr.Error as e:
            panes[slot] = f"**failed:** {e}"
            rows.append(f"| {name} | — | failed |")
            yield (*panes, "\n".join(rows))
            continue
        rows.append(f"| {name} | {time.time() - t0:.1f} s | {len(last)} |")
        yield (*panes, "\n".join(rows))

    yield (*panes, "\n".join(rows) + "\n\n_done._")


# --- UI ----------------------------------------------------------------------
def _toggle_opts(backend: str):
    paddle = BACKENDS[backend]["family"] == "paddle"
    return gr.update(visible=paddle), gr.update(visible=not paddle)


_missing = [f"`{c['env']}`" for n, c in BACKENDS.items() if not c["base"]]

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OCR Suite") as demo:
    gr.Markdown("# OCR Suite — PaddleOCR-VL · GGUF · Unlimited-OCR")
    gr.Markdown(
        "<br>".join(
            f"**{n}** — `{c['base']}`" for n, c in BACKENDS.items() if c["base"]
        )
        or "_No backend configured._"
    )
    if _missing:
        gr.Markdown(f"_Unconfigured: {', '.join(_missing)}_")

    with gr.Tabs():
        # ---------------------------------------------------------------- parse
        with gr.Tab("Document Parsing"):
            gr.Markdown(
                "_Whole pages and PDFs. The PaddleOCR-VL backends run DocLayout-YOLO "
                "first and route each block to the task that fits it; Unlimited-OCR "
                "parses each page in one pass. Output streams as it is produced._"
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
                    with gr.Group(visible=True) as paddle_opts:
                        tokens_p = gr.Slider(
                            128, 2048, value=512, step=128,
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
                    btn_p = gr.Button("Parse Document", variant="primary")
                    status_p = gr.Markdown("")
                with gr.Column(scale=7):
                    with gr.Tabs():
                        with gr.Tab("Preview"):
                            fs_p = gr.Button("⛶ Fullscreen", size="sm",
                                             elem_classes="fs-btn")
                            md_p = gr.Markdown(
                                latex_delimiters=LATEX_DELIMS,
                                max_height=PANE_MAX_H,
                                elem_classes="ocr-pane",
                                elem_id="pane-parse",
                            )
                        with gr.Tab("Source"):
                            raw_p = gr.Code(
                                language="markdown", lines=26, max_lines=26
                            )

            fs_p.click(fn=None, js=_fullscreen_js("pane-parse"))

            backend_p.change(
                _toggle_opts, inputs=backend_p, outputs=[paddle_opts, unlimited_opts]
            )
            btn_p.click(
                run_parse,
                inputs=[file_p, backend_p, tokens_p, conf_p, fmt_p,
                        mode_p, ngram_p, prompt_p],
                outputs=[md_p, raw_p, status_p],
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
                    backends_c = gr.Dropdown(
                        PARSE_CHOICES,
                        value=PARSE_CHOICES[:2],
                        multiselect=True,
                        max_choices=3,
                        label="Models",
                        info="Pick up to 3; they run one after another",
                    )
                    tokens_c = gr.Slider(
                        128, 2048, value=512, step=128, label="Max new tokens per block"
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
                inputs=[file_c, backends_c, tokens_c, conf_c],
                outputs=[*panes_c, status_c],
            )

if __name__ == "__main__":
    demo.queue().launch()
