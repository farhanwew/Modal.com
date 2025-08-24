import os, random, asyncio, ctypes, base64, io, gzip, json
from dataclasses import dataclass
from typing import Dict, Any, Optional

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
        "fastapi[standard]",          # <— WAJIB untuk @fastapi_endpoint
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "transformers==4.38.1",
        "sse_starlette",
        "numpy>=1.26",
        "requests",
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


# ====== Control Vector (CVEC) helpers ======
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # akan error saat dipakai jika numpy gagal terpasang


@dataclass
class ControlVector:
    directions: Dict[int, "np.ndarray"]  # layer -> vektor (float32, shape=[n_embd])

    def scaled(self, s: float) -> "ControlVector":
        return ControlVector({int(k): (np.asarray(v, dtype=np.float32) * float(s)) for k, v in self.directions.items()})


def _as_control_vector(obj: Any) -> ControlVector:
    """Terima dict dengan key 'directions' (keys bisa str/int) atau ControlVector langsung."""
    if isinstance(obj, ControlVector):
        return obj
    if isinstance(obj, dict) and "directions" in obj:
        d = {int(k): np.asarray(v, dtype=np.float32) for k, v in obj["directions"].items()}
        return ControlVector(d)
    raise ValueError("Format control vector tidak dikenali; berikan dict {'directions': {layer:int->array}} atau file .npy/.npz yang memuatnya.")


def _load_cvec_from_path(path: str) -> ControlVector:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Control vector tidak ditemukan: {path}")
    data = np.load(path, allow_pickle=True)

    # .npy object-array 0-D → .item()
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    # .npz → ambil entri bernama 'directions' jika ada, selain itu coba 'arr_0'
    if hasattr(data, "files"):
        if "directions" in data.files:
            data = {"directions": data["directions"].item() if isinstance(data["directions"], np.ndarray) else data["directions"]}
        elif "arr_0" in data.files:
            maybe = data["arr_0"]
            data = maybe.item() if isinstance(maybe, np.ndarray) and maybe.shape == () else maybe
        else:
            raise ValueError("File .npz tidak memuat 'directions'.")

    return _as_control_vector(data)


# ====== Base64 loaders for Control Vector ======

def _load_cvec_from_b64(b64_str: str) -> ControlVector:
    """Muat .npy/.npz dari string base64 (mendukung data: URL dan gzip)."""
    try:
        s = (b64_str or "").strip()
        if not s:
            raise ValueError("npy_b64 kosong")
        payload = s.split(",")[-1]
        raw = base64.b64decode(payload, validate=True)
        # jika digzip
        if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
            raw = gzip.decompress(raw)
        bio = io.BytesIO(raw)
        data = np.load(bio, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if hasattr(data, "files"):
            if "directions" in data.files:
                maybe = data["directions"]
                data = {"directions": maybe.item() if isinstance(maybe, np.ndarray) and maybe.shape == () else maybe}
            elif "arr_0" in data.files:
                maybe = data["arr_0"]
                data = maybe.item() if isinstance(maybe, np.ndarray) and maybe.shape == () else maybe
            else:
                raise ValueError("File .npz tidak memuat 'directions'.")
        return _as_control_vector(data)
    except Exception as e:
        raise RuntimeError(f"Decode base64 control vector gagal: {e}") from None


def _load_cvec_from_directions_b64(b64_str: str) -> ControlVector:
    """Muat directions dari JSON yang di-base64-kan: {layer:int -> list/array}."""
    try:
        payload = (b64_str or "").strip().split(",")[-1]
        raw = base64.b64decode(payload, validate=True)
        text = raw.decode("utf-8")
        obj = json.loads(text)
        if "directions" in obj:
            return _as_control_vector(obj)
        return _as_control_vector({"directions": obj})
    except Exception as e:
        raise RuntimeError(f"Decode base64 JSON directions gagal: {e}") from None


def _download_file(url: str, dst: str, timeout: int = 120) -> str:
    """Robust downloader dengan error yang bisa diserialisasi oleh Modal."""
    try:
        import requests
        url = str(url).strip()
        if not url:
            raise RuntimeError("URL kosong untuk control vector")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True, headers={"User-Agent": "modal-cvec/1.0"}) as r:
            try:
                r.raise_for_status()
            except requests.HTTPError:
                # Hindari mengirim objek file di exception (masalah pickling)
                raise RuntimeError(f"Gagal download control vector. HTTP {r.status_code} dari {url}") from None
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 20):  # 1 MB per chunk
                    if chunk:
                        f.write(chunk)
        return dst
    except Exception as e:
        # Pastikan exception bisa diserialisasi (string saja)
        raise RuntimeError(f"Download control vector gagal: {e}") from None


def _flatten_cvec(cvec: ControlVector, start_layer: Optional[int] = None, end_layer: Optional[int] = None):
    keys = sorted(int(k) for k in cvec.directions.keys())
    if not keys:
        raise ValueError("ControlVector kosong.")
    auto_start, auto_end = min(keys), max(keys)
    start = auto_start if start_layer is None else int(start_layer)
    end = auto_end if end_layer is None else int(end_layer)
    if end < start:
        raise ValueError("end_layer harus >= start_layer")

    n_embd = int(np.asarray(cvec.directions[keys[0]]).shape[-1])

    vecs = []
    for l in range(start, end + 1):
        v = cvec.directions.get(l)
        if v is None:
            v = np.zeros((n_embd,), dtype=np.float32)  # isi gap layer dengan nol
        else:
            v = np.asarray(v, dtype=np.float32)
            if int(v.shape[-1]) != n_embd:
                raise ValueError("n_embd tidak konsisten antar layer pada control vector")
        vecs.append(v)

    flat = np.concatenate(vecs).astype(np.float32)
    return flat, n_embd, start, end


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
        from llama_cpp import llama_cpp as llama_lib  # fungsi C low-level

        model_path = os.path.join(MODEL_DIR, GGUF_FILENAME)
        self.engine = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=16384,
            offload_kqv=True,
        )
        self.template = PROMPT_TEMPLATE
        self._lock = asyncio.Lock()  # hindari balapan saat apply/clear CVEC
        self._llama_lib = llama_lib

        # Cek ketersediaan simbol CVEC sedini mungkin
        if not hasattr(self._llama_lib, "llama_apply_adapter_cvec"):
            # Jangan crash saat start; akan error saat dipakai
            print("[WARN] llama_apply_adapter_cvec tidak tersedia di binding saat ini.")

    # === Internal: apply / clear control vector ===
    def _apply_or_clear_cvec(self, cvec: Optional[ControlVector], start_layer: Optional[int] = None, end_layer: Optional[int] = None):
        if not hasattr(self._llama_lib, "llama_apply_adapter_cvec"):
            raise RuntimeError(
                "Binding llama-cpp-python tidak mendukung 'llama_apply_adapter_cvec'. Pastikan pakai wheel prebuilt terbaru untuk CUDA/cuBLAS."
            )

        lctx = self.engine.ctx
        if cvec is None:
            rc = self._llama_lib.llama_apply_adapter_cvec(lctx, None, 0, 0, 0, 0)
            if rc != 0:
                raise RuntimeError(f"Gagal clear control vector, rc={rc}")
            return

        flat, n_embd, start, end = _flatten_cvec(cvec, start_layer, end_layer)
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._llama_lib.llama_apply_adapter_cvec(lctx, ptr, flat.size, int(n_embd), int(start), int(end))
        if rc != 0:
            raise RuntimeError(f"Gagal apply control vector, rc={rc}")
        
        
    @method()
    async def completion_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        seed: int = None,
        gen_config: Dict[str, Any] = {},
        # ——— NEW: parameter opsional untuk Control Vector ———
        control: Optional[Dict[str, Any]] = None,
    ):
        """Generate teks dan lanjutkan meski gagal apply control vector."""
        gen_config = dict(gen_config or {})

        # Seed
        self.engine.set_seed(seed if seed is not None else random.randint(0, 999999))
        tpl = gen_config.pop("prompt_template", self.template)

        # Siapkan Control Vector (jika diminta)
        cvec: Optional[ControlVector] = None
        start_layer = end_layer = None
        if control:
            try:
                # Aliases untuk kompatibilitas skema eksternal
                if "control_vector_npy_b64" in control and "npy_b64" not in control:
                    control["npy_b64"] = control.pop("control_vector_npy_b64")
                if "control_strength" in control and "strength" not in control:
                    control["strength"] = control.pop("control_strength")
                if "control_vector_b64_json" in control and "directions_b64" not in control:
                    control["directions_b64"] = control.pop("control_vector_b64_json")

                strength = float(control.get("strength", 1.0))
                start_layer = control.get("start_layer")
                end_layer = control.get("end_layer")

                if "directions" in control:
                    cvec = _as_control_vector({"directions": control["directions"]}).scaled(strength)
                elif control.get("npy_path"):
                    cvec = _load_cvec_from_path(control["npy_path"]).scaled(strength)
                elif control.get("npy_url"):
                    local = os.path.join(MODEL_DIR, "cvec.npy")
                    _download_file(control["npy_url"], local)
                    cvec = _load_cvec_from_path(local).scaled(strength)
                elif control.get("npy_b64"):
                    cvec = _load_cvec_from_b64(control["npy_b64"]).scaled(strength)
                elif control.get("directions_b64"):
                    cvec = _load_cvec_from_directions_b64(control["directions_b64"]).scaled(strength)
                else:
                    raise ValueError("'control' diberikan tapi tidak ada 'directions' / 'npy_path' / 'npy_url'.")
            except Exception as e:
                print(f"[WARN] Gagal memproses control vector: {e}. Lanjut tanpa kontrol.")
                cvec = None

        async with self._lock:
            # Coba apply control vector jika ada
            if cvec is not None:
                try:
                    self._apply_or_clear_cvec(cvec, start_layer, end_layer)
                    if hasattr(self.engine, "reset"):
                        self.engine.reset()
                except Exception as e:
                    print(f"[WARN] Gagal apply control vector: {e}. Melanjutkan tanpa kontrol.")
                    cvec = None

            # Generate teks
            res = self.engine(
                prompt=tpl.format(system_prompt=system_prompt, prompt=prompt),
                **gen_config,
            )
            text = res["choices"][0]["text"]

            # Clear control vector jika sempat diterapkan
            if cvec is not None:
                try:
                    self._apply_or_clear_cvec(None)
                    if hasattr(self.engine, "reset"):
                        self.engine.reset()
                except Exception as e:
                    print(f"[WARN] Gagal clear control vector: {e}.")

        return text



@app.function(image=lcpp_image)
@modal.fastapi_endpoint(method="POST")
def handle_req(param: Dict[str, Any]):
    model = Model()

    # pastikan dict & siapkan gen_config
    param = dict(param or {})
    gen_config = dict(param.get("gen_config") or {})

    # pindahkan prompt_template -> gen_config, lalu hapus dari top-level
    if "prompt_template" in param and param["prompt_template"] is not None:
        gen_config["prompt_template"] = param["prompt_template"]
        del param["prompt_template"]

    param["gen_config"] = gen_config

    # hanya kirim argumen yang dikenali completion_stream
    allowed = {"prompt", "system_prompt", "seed", "gen_config", "control"}
    filtered = {k: v for k, v in param.items() if k in allowed}

    # panggil dan langsung return string hasilnya
    result = model.completion_stream.remote(**filtered)
    return {"res": result}


