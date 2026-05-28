# Modal Whisper Transcriber

Deploy OpenAI Whisper `turbo` on Modal and expose it as a POST endpoint for audio or video transcription.

## Configuration

- App name: `whisper-transcriber`
- Model: Whisper `turbo`
- GPU: `L4`
- CPU: `4`
- Memory: `16 GB`
- Timeout: `30 minutes`
- Warm idle window: `300 seconds`
- Model cache path in image: `/model-cache/whisper`

The model is downloaded during Modal image build with `image.run_function(_download_model)`, so cold starts load the bundled model instead of downloading it on each new container.

## Deploy

```bash
modal deploy transcribe.py
```

## Run from local CLI

```bash
modal run transcribe.py --audio-url "https://example.com/audio.mp3"
```

Video URLs also work as long as `ffmpeg` can decode the audio track:

```bash
modal run transcribe.py --audio-url "https://example.com/video.mp4"
```

## Call the deployed endpoint

Send exactly one of `audio_url` or `audio_base64`:

```bash
curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"https://example.com/video.mp4","task":"transcribe"}'
```

Optional fields:

- `language`: language code such as `en`, `id`, or `ja`
- `task`: `transcribe` or `translate`

Example response includes `text`, detected `language`, and Whisper `segments` with timestamps.
