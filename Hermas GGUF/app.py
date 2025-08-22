# app.py
import os, random
from typing import Dict

import modal
from modal import Image, enter, method

# ===== Konfigurasi model (Hermes 2 Pro GGUF) =====
MODEL_DIR = "/model"
BASE_MODEL = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
GGUF_FILENAME = "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf"

# ===== Wajib: variabel global 'app' =====
app = modal.App("lcpp-hermes2pro")

GPU_CONFIG = "T4"

PROMPT_TEMPLATE = (
    """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
)

def _download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    os.makedirs(MODEL_DIR, exist_ok=True)
    target = os.path.join(MODEL_DIR, GGUF_FILENAME)
    if not os.path.isfile(target):
        snapshot_download(
            BASE_MODEL,
            local_dir=MODEL_DIR,
            allow_patterns=[GGUF_FILENAME],  # unduh 1 file gguf
        )
    print("Model path:", target)
    move_cache()

# ===== Image: FastAPI + llama-cpp-python (prebuilt cu121) =====
lcpp_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "fastapi[standard]",           # <— WAJIB untuk @fastapi_endpoint
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "transformers==4.38.1",
        "sse_starlette",
    )
    .apt_install(
        "build-essential",
        "git",
        "ocl-icd-opencl-dev",
        "opencl-headers",
        "clinfo",
        "libclblast-dev",
        "libopenblas-dev",
    )
    .run_commands(
        "mkdir -p /etc/OpenCL/vendors",
        'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd',
        "python -m pip install --upgrade pip",
        # prebuilt wheel untuk CUDA 12.1 → tidak compile
        'pip install "llama-cpp-python" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121',
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CUDA_DOCKER_ARCH": "ALL"})
    .run_function(_download_model, timeout=60 * 20, force_build=False)
)

@app.cls(
    gpu=GPU_CONFIG,
    image=lcpp_image,
    timeout=80,
    container_idle_timeout=80,
    allow_concurrent_inputs=10,
)
class Model:
    @enter()
    def start_engine(self):
        from llama_cpp import Llama
        model_path = os.path.join(MODEL_DIR, GGUF_FILENAME)
        self.engine = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=16384,
            offload_kqv=True,
        )
        self.template = PROMPT_TEMPLATE

    @method()
    async def completion_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        seed: int = None,
        gen_config: Dict = {},
    ):
        self.engine.set_seed(seed if seed is not None else random.randint(0, 999999))
        tpl = gen_config.get("prompt_template", self.template)
        res = self.engine(
            prompt=tpl.format(system_prompt=system_prompt, prompt=prompt),
            **{k: v for k, v in gen_config.items() if k != "prompt_template"},
        )
        return res["choices"][0]["text"]

@app.function(image=lcpp_image)
@modal.fastapi_endpoint(method="POST")
def handle_req(param: Dict):
    model = Model()
    return {"res": model.completion_stream.remote(**param)}
