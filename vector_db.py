import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from fastembed import TextEmbedding

COLLECTION_NAME = "persona_vectors"
# Use a local path safely in one central place
client = QdrantClient(path="qdrant_data")

# Loaded once here and imported by both main.py and ingestion.py, instead of
# each loading its own separate copy of the same model (which doubled
# startup time and RAM usage for no benefit).
#
# Uses fastembed (ONNX Runtime) instead of sentence-transformers (PyTorch) --
# same model, same 384-dim output, but without pulling in PyTorch, which was
# blowing past Render's 512MB memory limit on its own. fastembed is built by
# Qdrant themselves specifically for low-memory/serverless environments.
embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def encode_single(text: str) -> list:
    """Embeds one piece of text, returning a flat list of floats -- matches
    the shape callers previously got from SentenceTransformer.encode(text).tolist()."""
    return list(embedding_model.embed([text]))[0].tolist()

def encode_many(texts: list) -> list:
    """Embeds a list of texts, returning a list of flat float lists -- matches
    the shape callers previously got from SentenceTransformer.encode(texts).tolist()."""
    return [vec.tolist() for vec in embedding_model.embed(texts)]

def init_vector_db():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_vector_db()