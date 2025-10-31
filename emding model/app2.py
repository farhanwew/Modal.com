import modal
from typing import List

# === Build image + embed model saat build ===
def _download_model():
    """Dipanggil saat build image untuk predownload model embedding"""
    from sentence_transformers import SentenceTransformer
    import os

    model_dir = "/root/model/all-mpnet-base-v2"
    os.makedirs(model_dir, exist_ok=True)
    print("Downloading SentenceTransformer model to:", model_dir)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.save(model_dir)
    print("✅ Model saved into image layer")

# Build image
image = (
    modal.Image.debian_slim()
    .pip_install(["torch", "sentence-transformers", "numpy", "fastapi[standard]"])
    .run_function(_download_model)
)

app = modal.App("embedding-task-embed")

# === Model class ===
@app.cls(
    image=image,
    gpu="A10G",           # ✅ new syntax
    timeout=90,
)
class EmbeddingModel:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        print("Loading model from image cache...")
        self.model = SentenceTransformer("/root/model/all-mpnet-base-v2")
        print("✅ Model loaded and ready!")

    @modal.method()
    def embed(self, texts: List[str]):
        """Generate embeddings untuk list of texts"""
        import numpy as np
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True)
        return np.array(embeddings).tolist()


# === API endpoint ===
@app.function(image=image)
@modal.fastapi_endpoint(method="POST", label="embedding-api-2")  # ✅ updated decorator
def api_embed(item: dict):
    """Endpoint REST untuk generate embeddings"""
    texts = item.get("texts", [])
    if not texts or not isinstance(texts, list):
        return {"error": "Missing or invalid 'texts' field"}
    model = EmbeddingModel()
    embeddings = model.embed.remote(texts)
    return {"embeddings": embeddings}


# === Local test ===
@app.local_entrypoint()
def main():
    model = EmbeddingModel()
    texts = ["The sky is blue.", "The ocean is deep."]
    print("Embedding dim:", len(model.embed(texts)[0]))
