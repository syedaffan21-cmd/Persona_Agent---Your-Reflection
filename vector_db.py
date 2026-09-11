import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "persona_vectors"
# Use a local path safely in one central place
client = QdrantClient(path="qdrant_data")

# Loaded once here and imported by both main.py and ingestion.py, instead of
# each loading its own separate copy of the same model (which doubled
# startup time and RAM usage for no benefit).
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def init_vector_db():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_vector_db()