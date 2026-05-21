# Modal deployment for `aitf-its-tim3-dfk/qwen3.5-0.8B-ws3`

This deploys the Hugging Face model on Modal with a SITA-style DFK VLM prompt. The API accepts `title`, `context`/`text`, and `image_url` or `image_base64`, then asks the model to return `Label:` and `Analisis:`.

This deploys the Hugging Face model on Modal with:

- `modal.App("qwen35-ws3")`
- one `modal.Cls` that loads the model once per warm GPU container
- one POST web endpoint with interactive docs enabled
- a Modal volume cache for Hugging Face downloads

## Setup

Install and authenticate Modal locally:

```bash
pip install modal
modal setup
```

If the Hugging Face repo is private or gated, create a Modal secret named
`huggingface-secret` containing `HF_TOKEN`, then add this argument to both
decorators in `modal_app.py`:

```python
secrets=[modal.Secret.from_name("huggingface-secret")]
```

Create the secret with:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

## Test once

```bash
cd /home/farhan/modal-qwen35-ws3
modal run modal_app.py --title "Judul unggahan" --context "Teks/konteks unggahan"
```

For an image URL:

```bash
modal run modal_app.py --title "Judul unggahan" --context "Teks/konteks unggahan" --image-url "https://example.com/image.jpg"
```

## Serve during development

```bash
modal serve modal_app.py
```

Modal will print a temporary endpoint URL. Send requests like:

```bash
curl -X POST "$MODAL_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Judul unggahan media sosial",
    "context": "Teks/konteks unggahan yang menyertai gambar",
    "image_url": "https://example.com/image.jpg",
    "max_new_tokens": 256,
    "temperature": 0.0
  }'
```

## Deploy

```bash
cd /home/farhan/modal-qwen35-ws3
modal deploy modal_app.py
```

After deployment, stream logs with:

```bash
modal app logs qwen35-ws3
```
