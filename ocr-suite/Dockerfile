# Container for the merged Gradio client (gradio_app.py) — all three backends.
#
# No model is in here. The models run on Modal, and their images are defined in
# code via `modal.Image` inside the modal_*.py scripts, not via a Dockerfile.
# This container is a thin HTTP frontend: no GPU, no weights, no CUDA.
#
# Supersedes Dockerfile.paddleocr-vl and Dockerfile.unlimited-ocr, which each
# built the same client with only one backend wired up.

FROM python:3.12-slim

# Keep in sync with the PEP 723 dependency block at the top of gradio_app.py —
# that block is what `uv run gradio_app.py` uses locally. Pillow is needed for
# the JPEG re-encode that keeps large page uploads from dominating wall clock.
RUN pip install --no-cache-dir \
        "gradio>=5.0,<6" \
        "requests>=2,<3" \
        "Pillow>=10"

WORKDIR /app
COPY gradio_app.py .

# Gradio binds 127.0.0.1 by default, which is unreachable from outside the
# container; GRADIO_SERVER_NAME is how it's told to listen on all interfaces.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Supply at run time. A backend left empty is shown as unconfigured in the UI
# rather than failing on click, so the container is usable with only one set.
ENV PADDLEOCR_VL_GGUF_BASE_URL="" \
    PADDLEOCR_VL_BASE_URL="" \
    UNLIMITED_OCR_BASE_URL=""

# Optional tuning, same defaults as the script.
ENV JPEG_QUALITY=92 \
    JPEG_ABOVE_KB=400 \
    PANE_MAX_HEIGHT=68vh

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 7860

# No HEALTHCHECK on the backends on purpose: they are Modal apps that scale to
# zero, and polling them would keep GPU containers alive and billing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/').read(1)" || exit 1

CMD ["python", "gradio_app.py"]
