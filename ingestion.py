import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from vector_db import client, COLLECTION_NAME, init_vector_db

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_path: str) -> str:
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
    if re.search(r'\b\w \w \w\b', text):
        text = re.sub(r'(?<!^)(?<!\s)\s(?!\s)', '', text)
        
    return text

def process_and_store_document(file_path: str):
    raw_text = extract_text_from_file(file_path)
    if not raw_text:
        return 0
    
    init_vector_db()
    
    # Intelligently split text by sections if present, otherwise chunk normally
    chunks = []
    if "Project" in raw_text or "PROJECT" in raw_text:
        # Split resume into logical sections
        parts = re.split(r'(Projects|EXPERIENCE|Education|Skills)', raw_text, flags=re.IGNORECASE)
        current_header = "General"
        for part in parts:
            if part.lower() in ["projects", "experience", "education", "skills"]:
                current_header = part
            elif len(part.strip()) > 30:
                chunks.append(f"[{current_header}] {part.strip()}")
    
    # Fallback standard chunking if section splitting yielded nothing
    if not chunks:
        words = raw_text.split()
        for i in range(0, len(words), 300):
            chunks.append(" ".join(words[i:i + 300]))

    embeddings = model.encode(chunks).tolist()
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hash(f"{file_path}_{idx}") & 0x7FFFFFFF
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": chunk, "source": os.path.basename(file_path)}
            )
        )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(chunks)