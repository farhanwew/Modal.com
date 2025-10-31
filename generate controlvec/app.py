# cvec_generator.py
import os, io, base64, re
from typing import Dict, Any, List, Optional

import modal
from modal import Image

# ========= App & Config =========
APP_NAME = "cvec-generator"
MODEL_ID = "NousResearch/Hermes-2-Pro-Mistral-7B"      # public model
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

# Layer range negatif = hitung dari belakang (transformer blocks)
LAYER_RANGE_DEFAULT = list(range(-5, -18, -1))

TRUE_FACTS_URL = "https://raw.githubusercontent.com/vgel/repeng/refs/heads/main/notebooks/data/true_facts.json"

# ========= Persistent Volume =========
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "hf-cache-vol")
MODEL_DIR = "/models/hf"                               # mount point di container
HF_REPO_ID = os.environ.get("HF_REPO_ID", MODEL_ID)   # default = MODEL_ID
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

# ========= Base Image (deps) =========
base_image = (
    Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "curl")
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "torch",
        "accelerate",
        "transformers",
        "repeng",
        "requests",
        "sentencepiece",     # penting buat tokenizer SPM
        "tokenizers",        # opsional
        "safetensors",       # opsional
        "huggingface-hub",   # snapshot_download
    )
)

# ========= Globals (lazy-loaded) =========
model = None
tokenizer = None
suffixes: List[str] = []

system_tag = "<|im_start|>system\n"
user_tag   = "<|im_start|>user\n"
asst_tag   = "<|im_start|>assistant\n"
eos_tag    = "<|im_end|>\n"


def _load_suffixes(url: str) -> List[str]:
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _ensure_loaded():
    """
    Lazy-load model/tokenizer.
    Prefer lokal (Volume) -> fallback HF Hub.
    """
    global model, tokenizer, suffixes
    if model is not None and tokenizer is not None and len(suffixes) > 0:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prefer_local = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    try:
        if prefer_local:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
            # set pad_token ke eos kalau belum ada
            if getattr(tokenizer, "pad_token_id", None) is None:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is None:
                    tokenizer.pad_token_id = 0
                else:
                    tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, local_files_only=True, dtype=torch.float16
            )
        else:
            print(f"[WARN] MODEL_DIR {MODEL_DIR} kosong; download {MODEL_ID} dari Hub (non-persistent).")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if getattr(tokenizer, "pad_token_id", None) is None:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is None:
                    tokenizer.pad_token_id = 0
                else:
                    tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float16
            )

        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        suffixes[:] = _load_suffixes(TRUE_FACTS_URL)
    except Exception as e:
        import traceback
        print("ERROR in _ensure_loaded():", e)
        print(traceback.format_exc())
        raise


def generate_antonym(attribute: str, model, tokenizer) -> str:
    prompt = (
        f"{user_tag}What is the single-word opposite of '{attribute}'? "
        f"REPLY WITH JUST THE WORD.{eos_tag}{asst_tag}The opposite is '"
    )
    toks = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        toks.input_ids,
        attention_mask=toks.get("attention_mask", None),
        max_new_tokens=10,
        temperature=0.1,
        repetition_penalty=1.2,
    )
    full = tokenizer.decode(out[0][toks.input_ids.shape[-1]:], skip_special_tokens=True)
    return (
        full.strip()
        .split()[0]
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
    )


def generate_examples(persona: str, num_examples: int, model, tokenizer) -> List[str]:
    prompt = (
        f"{user_tag}I need short, one-sentence descriptions of a person's behavior if the person is {persona}. "
        f"Provide a numbered list of {num_examples} distinct examples. YOU MUST PROVIDE {num_examples} items. "
        f"For example: 1. a compassionate friend. 2. a sympathetic listener. "
        f"Do not write anything else, just the numbered list.{eos_tag}{asst_tag}"
    )
    toks = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        toks.input_ids,
        attention_mask=toks.get("attention_mask", None),
        max_new_tokens=150,
        temperature=0.6,
        do_sample=True,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    body = full.split(eos_tag)[-1].strip()
    examples = re.findall(r"^\s*\d+\.\s*(.*)", body, re.MULTILINE)
    return examples[:num_examples]


def make_dataset(
    template: str,
    positive_personas: List[str],
    negative_personas: List[str],
    suffix_list: List[str],
) -> List["DatasetEntry"]:
    from repeng import DatasetEntry
    dataset = []
    for suffix in suffix_list:
        for pos, neg in zip(positive_personas, negative_personas):
            pos_t = template.format(persona=pos)
            neg_t = template.format(persona=neg)
            dataset.append(
                DatasetEntry(
                    positive=f"{system_tag}{pos_t}{eos_tag}{user_tag}Hey{eos_tag}{asst_tag}{suffix}",
                    negative=f"{system_tag}{neg_t}{eos_tag}{user_tag}Hey{eos_tag}{asst_tag}{suffix}",
                )
            )
    return dataset


def generate_control_vector(
    attribute: str,
    model,
    tokenizer,
    num_examples: int = 3,
    layer_range: Optional[List[int]] = None,
    forced_negative: Optional[str] = None,
) -> "np.ndarray":
    """
    Compat shim untuk berbagai versi repeng:
    - Coba jalur baru: ControlVector.train(..., method="pca_center")
    - Jika TypeError (API lama), baca representations lalu reduksi manual.
    """
    import numpy as np
    from repeng import ControlVector, ControlModel

    base = model

    pos_persona = attribute
    neg_persona = forced_negative or generate_antonym(pos_persona, base, tokenizer)

    pos_examples = generate_examples(f"a very {pos_persona} person", num_examples, base, tokenizer)
    pos_examples.append(f"a very {pos_persona} person")

    neg_examples = generate_examples(f"a very {neg_persona} person", num_examples, base, tokenizer)
    neg_examples.append(f"a very {neg_persona} person")

    dataset = make_dataset(
        "Pretend you're a person who acts like this: '{persona}'. Now, make a statement about the world.",
        pos_examples, neg_examples, suffixes
    )

    cm = ControlModel(base, layer_range or LAYER_RANGE_DEFAULT)
    cm.reset()
    try:
        # Jalur API baru
        cv = ControlVector.train(cm, tokenizer, dataset, method="pca_center")
        vec_like = getattr(cv, "vector", cv)
        arr = (
            vec_like.detach().float().cpu().numpy().astype(np.float32)
            if hasattr(vec_like, "detach")
            else np.asarray(vec_like, dtype=np.float32)
        )
    except TypeError:
        # Jalur API lama: tanpa argumen 'method'
        try:
            from repeng.extract import read_representations
        except Exception:
            # fallback jika modul dipindah
            from repeng import extract as _extract
            read_representations = _extract.read_representations

        dirs = read_representations(cm, tokenizer, dataset)  # dict[layer] -> (n, dim) / (dim,)
        # Reduksi manual cepat: rata-rata per-layer (kalau 2D), lalu rata-rata antar-layer
        layer_means = []
        for v in dirs.values():
            if hasattr(v, "detach"):
                v = v.detach().float().cpu().numpy()
            else:
                v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2:
                layer_means.append(v.mean(axis=0))
            else:
                layer_means.append(v.astype(np.float32))
        arr = np.mean(np.stack(layer_means, axis=0), axis=0).astype(np.float32)
    finally:
        cm.reset()

    return arr


# ========= Prewarm (download model → Volume) =========
@app.function(image=base_image, volumes={"/models": vol})
def prewarm_models():
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    return {"ok": True, "model_dir": MODEL_DIR}


# ========= FastAPI Endpoint (GPU) =========
@app.function(
    image=base_image,
    gpu=GPU_TYPE,                 # <-- gunakan GPU (mis. "L4")
    volumes={"/models": vol},     # <-- mount volume agar model persist
    timeout=60 * 30,
)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import numpy as np

    api = FastAPI(title=APP_NAME)

    @api.get("/health")
    def health():
        local = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
        return {
            "ok": True,
            "model": MODEL_ID,
            "gpu": GPU_TYPE,
            "has_local_model": local,
        }

    @api.post("/generate")
    def generate(payload: Dict[str, Any]):
        """
        Generate control vector dari attribute(s).

        Request JSON:
        {
          "attribute": "happy",
          // ATAU
          "attributes": ["happy", "confident"],
          "negative": "sad",                  # opsional (paksa antonym)
          "num_examples": 3,                  # opsional
          "layer_range": [-5,-6,...,-17],     # opsional
          "merge_mode": "mean"                # "mean" | "concat" (jika multi attributes)
        }

        Response JSON:
        {
          "attribute": "happy" | ["happy","confident"],
          "filename": "happy.npy",
          "data_b64": "<BASE64_NPY>",
          "shape": [4096] atau [N,4096] jika concat,
          "negative_used": "sad"
        }
        """
        try:
            _ensure_loaded()

            single_attr = payload.get("attribute")
            multi_attrs = payload.get("attributes")

            if not single_attr and not multi_attrs:
                raise HTTPException(400, detail="Provide either 'attribute' (str) or 'attributes' (list)")

            attributes = [single_attr] if single_attr else list(multi_attrs)
            if not isinstance(attributes, list):
                attributes = [attributes]

            negative = payload.get("negative")
            num_examples = int(payload.get("num_examples", 3))
            layer_range = payload.get("layer_range")
            merge_mode = payload.get("merge_mode", "mean")

            vectors = []
            negatives_used = []

            for attr in attributes:
                vec = generate_control_vector(
                    attr,
                    model,
                    tokenizer,
                    num_examples=num_examples,
                    layer_range=layer_range,
                    forced_negative=negative if len(attributes) == 1 else None,
                )
                vectors.append(vec)
                neg_used = negative or generate_antonym(attr, model, tokenizer)
                negatives_used.append(neg_used)

            if len(vectors) > 1:
                if merge_mode == "concat":
                    final_vector = np.concatenate(vectors, axis=0)
                else:
                    final_vector = np.mean(vectors, axis=0)
            else:
                final_vector = vectors[0]

            buf = io.BytesIO()
            np.save(buf, final_vector)
            npy_data = buf.getvalue()
            b64 = base64.b64encode(npy_data).decode("ascii")

            if len(attributes) == 1:
                filename = f"{attributes[0]}.npy"
                attr_info = attributes[0]
            else:
                filename = f"{'_'.join(attributes[:3])}_merged.npy"
                attr_info = attributes

            return JSONResponse(
                {
                    "attribute": attr_info,
                    "negative_used": negatives_used[0] if len(negatives_used) == 1 else negatives_used,
                    "filename": filename,
                    "data_b64": b64,
                    "shape": list(final_vector.shape),
                    "merge_mode": merge_mode if len(vectors) > 1 else None,
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            raise HTTPException(500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    return api
