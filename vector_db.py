import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "persona_vectors"
# Use a local path safely in one central place
client = QdrantClient(path="qdrant_data")

def init_vector_db():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_vector_db()