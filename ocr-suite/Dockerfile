# Container for the merged Gradio client (gradio_app.py).
#
# The three Modal-hosted models are not in here: they run remotely and their
# images are defined in code via `modal.Image` inside the modal_*.py scripts.
# The fourth backend, OpenDataLoader, is a Java parser that runs in-process —
# that is the only thing this image carries beyond the HTTP client itself.
# No GPU, no model weights, no CUDA, no torch.

FROM python:3.12-slim

# OpenDataLoader is a Java program with a Python wrapper; without a JRE that
# backend hides itself with "no Java 11+ runtime found". headless keeps the
# image from dragging in an X stack it will never use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Keep in sync with the PEP 723 dependency block at the top of gradio_app.py —
# that block is what `uv run gradio_app.py` uses locally.
RUN pip install --no-cache-dir \
        "gradio>=5.0,<6" \
        "requests>=2,<3" \
        "Pillow>=10" \
        "pymupdf>=1.24" \
        "opendataloader-pdf"

WORKDIR /app
COPY gradio_app.py .

# Gradio binds 127.0.0.1 by default, which is unreachable from outside the
# container; GRADIO_SERVER_NAME is how it's told to listen on all interfaces.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Supply at run time (--env-file .env is easiest). A backend whose variable is
# empty shows as unconfigured in the UI rather than failing on click, so the
# container is usable with only one of them set.
ENV PADDLEOCR_VL_GGUF_BASE_URL="" \
    PADDLEOCR_VL_GGUF_OFFICIAL_BASE_URL="" \
    PADDLEOCR_VL_BASE_URL="" \
    UNLIMITED_OCR_BASE_URL="" \
    GLM_OCR_BASE_URL="" \
    DOCLING_BASE_URL="" \
    MINERU_BASE_URL="" \
    MINERU_GGUF_Q8_BASE_URL="" \
    MINERU_GGUF_Q4_BASE_URL=""

# Optional tuning, same defaults as the script.
ENV JPEG_QUALITY=92 \
    JPEG_ABOVE_KB=400 \
    PANE_MAX_HEIGHT=68vh \
    PREVIEW_MAX_PAGES=0

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Java's temp handling and huggingface_hub both want a writable HOME; the
# non-root user has one, but be explicit so a read-only mount does not surprise.
ENV HOME=/home/app

EXPOSE 7860

# Deliberately probes only Gradio. The Modal backends scale to zero, and a
# health poll every 30 s would hold a GPU container alive and billing — which
# is exactly the trap this project already fell into once.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/').read(1)" || exit 1

CMD ["python", "gradio_app.py"]
