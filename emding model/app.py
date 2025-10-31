import modal
from typing import List

# === Config image ===
image = (
    modal.Image.debian_slim()
    .pip_install(["torch", "sentence-transformers", "numpy", "fastapi[standard]"])
    .add_local_dir(".", remote_path="/root")
)

app = modal.App("embedding-task")

# === Cache untuk model ===
model_volume = modal.Volume.from_name("embedding-cache", create_if_missing=True)

# === Model class ===
@app.cls(
    image=image,
    gpu=modal.gpu.A10G(count=1),
    volumes={"/model": model_volume},
    timeout=90,
)
class EmbeddingModel:
    @modal.enter()
    def load_model(self):
        """Load embedding model saat startup container"""
        import os
        from sentence_transformers import SentenceTransformer

        model_path = "/model/all-mpnet-base-v2"
        if not os.path.exists(model_path):
            print("Downloading model...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            model.save(model_path)
        else:
            model = SentenceTransformer(model_path)
        self.model = model
        print("Model loaded successfully!")

    @modal.method()
    def embed(self, texts: List[str]):
        """Generate embeddings untuk list of texts"""
        import numpy as np
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True)
        return np.array(embeddings).tolist()


# === Web endpoint ===
@app.function(image=image)
@modal.web_endpoint(method="POST", label="embedding-api")
def api_embed(item: dict):
    """Endpoint REST: kirim teks list"""
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
