import base64
import io
from typing import Any

import modal


APP_NAME = "qwen35-ws3"
MODEL_ID = "aitf-its-tim3-dfk/qwen3.5-0.8B-ws3"
CACHE_DIR = "/cache/huggingface"


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("qwen35-ws3-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "accelerate",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
        "peft",
        "pillow",
        "sentencepiece",
        "torch",
        "torchvision",
        "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)


def _decode_image(image_base64: str):
    from PIL import Image

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")



DFK_INSTRUCTION = (
    "Anda adalah seorang analis konten media sosial ahli. "
    "Diberikan tangkapan layar dari sebuah unggahan media sosial dan metadata berupa "
    "ringkasan, klaim, serta fakta pembanding. Tentukan label kategori pelanggaran "
    "dan berikan analisis detail mengenai pelanggaran yang ditemukan. "
    "Jawab hanya dengan format: Label: <label> lalu Analisis: <analisis>."
)


def build_dfk_content(
    title: str = "",
    context: str = "",
    ringkasan: str = "",
    klaim: str = "",
    fakta: str = "",
) -> list[dict[str, str]]:
    context_parts = []
    if ringkasan.strip():
        context_parts.append(f"Ringkasan: {ringkasan.strip()}")
    if klaim.strip():
        context_parts.append(f"Klaim: {klaim.strip()}")
    if fakta.strip():
        context_parts.append(f"Fakta: {fakta.strip()}")
    if title.strip():
        context_parts.append(f"Judul: {title.strip()}")
    if context.strip():
        context_parts.append(f"Konteks: {context.strip()}")

    content = [{"type": "text", "text": DFK_INSTRUCTION}]
    if context_parts:
        content.append({"type": "text", "text": "\n".join(context_parts)})
    return content


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=600,
    scaledown_window=60,
    volumes={CACHE_DIR: hf_cache},
)
class QwenServer:
    @modal.enter()
    def load_model(self):
        import json
        import os

        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        token = os.environ.get("HF_TOKEN")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        base_model_id = MODEL_ID
        adapter_model_id = None

        try:
            adapter_config_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="adapter_config.json",
                token=token,
                cache_dir=CACHE_DIR,
            )
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_id = adapter_config.get("base_model_name_or_path") or base_model_id
            adapter_model_id = MODEL_ID
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(
            base_model_id,
            token=token,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )

        if adapter_model_id:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_model_id,
                token=token,
                cache_dir=CACHE_DIR,
            )

        self.model.eval()

    @modal.method()
    def generate(
        self,
        prompt: str | None = None,
        image_url: str | None = None,
        image_base64: str | None = None,
        title: str = "",
        context: str = "",
        ringkasan: str = "",
        klaim: str = "",
        fakta: str = "",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        import torch

        if image_url and image_base64:
            raise ValueError("Provide either image_url or image_base64, not both.")

        if prompt:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        else:
            content = build_dfk_content(
                title=title,
                context=context,
                ringkasan=ringkasan,
                klaim=klaim,
                fakta=fakta,
            )

        if image_url:
            content.append({"type": "image", "url": image_url})
        elif image_base64:
            image = _decode_image(image_base64)
            content.append({"type": "image", "image": image})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        new_token_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
        output = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"text": output.strip()}


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
def infer(payload: dict[str, Any]) -> dict[str, Any]:
    return QwenServer().generate.remote(
        prompt=payload.get("prompt"),
        image_url=payload.get("image_url"),
        image_base64=payload.get("image_base64"),
        title=str(payload.get("title") or ""),
        context=str(payload.get("context") or payload.get("text") or ""),
        ringkasan=str(payload.get("ringkasan") or payload.get("summary") or ""),
        klaim=str(payload.get("klaim") or payload.get("claim") or ""),
        fakta=str(payload.get("fakta") or payload.get("fact") or ""),
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.9)),
    )


@app.local_entrypoint()
def main(
    prompt: str | None = None,
    image_url: str | None = None,
    title: str = "",
    context: str = "",
    ringkasan: str = "",
    klaim: str = "",
    fakta: str = "",
    max_new_tokens: int = 256,
):
    result = QwenServer().generate.remote(
        prompt=prompt,
        image_url=image_url,
        title=title,
        context=context,
        ringkasan=ringkasan,
        klaim=klaim,
        fakta=fakta,
        max_new_tokens=max_new_tokens,
    )
    print(result["text"])
