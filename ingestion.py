import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from vector_db import client, COLLECTION_NAME, init_vector_db

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_path: str) -> str:
    text = ""
    if not file_path:
        return text
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def process_and_store_document(file_path: str = None, persona: str = "My Personal Twin", direct_text: str = None):
    raw_text = ""
    if direct_text:
        raw_text = direct_text
    elif file_path:
        raw_text = extract_text_from_file(file_path)
        
    if not raw_text:
        return 0
    
    init_vector_db()
    
    chunks = []
    words = raw_text.split()
    for i in range(0, len(words), 300):
        chunks.append(" ".join(words[i:i + 300]))

    embeddings = model.encode(chunks).tolist()
    
    source_name = os.path.basename(file_path) if file_path else f"{persona}_profile"
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hash(f"{source_name}_{idx}_{persona}") & 0x7FFFFFFF
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                # Store the persona tag in the payload!
                payload={"text": chunk, "source": source_name, "persona": persona}
            )
        )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(chunks)