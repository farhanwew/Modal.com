from __future__ import annotations

import base64
from typing import Any

import modal


APP_NAME = "whisper-transcriber"
CACHE_ROOT = "/model-cache"
MODEL_CACHE_DIR = "/model-cache/whisper"
DEFAULT_MODEL = "turbo"


def _download_model() -> None:
    import whisper

    whisper.load_model(DEFAULT_MODEL, download_root=MODEL_CACHE_DIR)


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "fastapi[standard]",
        "openai-whisper",
        "requests",
    )
    .env({"XDG_CACHE_HOME": MODEL_CACHE_DIR})
    .run_function(_download_model)
)


def _download_audio(url: str) -> bytes:
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=120,
    )
    response.raise_for_status()
    return response.content


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16 * 1024,
    timeout=60 * 30,
    scaledown_window=300,
)
class WhisperTranscriber:
    @modal.enter()
    def load_model(self) -> None:
        import whisper

        self.model = whisper.load_model(DEFAULT_MODEL, download_root=MODEL_CACHE_DIR)

    def _transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, Any]:
        import os
        import tempfile

        suffix = ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
            audio_file.write(audio_bytes)
            audio_path = audio_file.name

        try:
            result = self.model.transcribe(
                audio_path,
                language=language or None,
                task=task,
                fp16=True,
            )
        finally:
            os.unlink(audio_path)

        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": result.get("segments", []),
        }

    @modal.method()
    def transcribe_url(
        self,
        audio_url: str,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, Any]:
        return self._transcribe_bytes(
            _download_audio(audio_url),
            language=language,
            task=task,
        )

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, Any]:
        return self._transcribe_bytes(
            audio_bytes,
            language=language,
            task=task,
        )


@app.function(image=image, timeout=60 * 30)
@modal.fastapi_endpoint(method="POST", docs=True)
def transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    audio_url = payload.get("audio_url") or payload.get("url")
    audio_base64 = payload.get("audio_base64")

    if bool(audio_url) == bool(audio_base64):
        raise ValueError("Provide exactly one of audio_url or audio_base64.")

    task = str(payload.get("task") or "transcribe")
    language = payload.get("language")

    if audio_url:
        return WhisperTranscriber().transcribe_url.remote(
            str(audio_url),
            language=language,
            task=task,
        )

    audio_bytes = base64.b64decode(str(audio_base64))
    return WhisperTranscriber().transcribe.remote(
        audio_bytes,
        language=language,
        task=task,
    )


@app.local_entrypoint()
def main(
    audio_url: str,
    language: str | None = None,
    task: str = "transcribe",
) -> None:
    result = WhisperTranscriber().transcribe_url.remote(
        audio_url,
        language=language,
        task=task,
    )
    print(result["text"])
