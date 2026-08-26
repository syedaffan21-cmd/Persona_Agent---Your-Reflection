import os
import json
import shutil
import io
import base64
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from ingestion import process_and_store_document
from graph_db import graph_db
from vector_db import client as db_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pypdf
import requests
import PIL.Image

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

chat_model = SentenceTransformer('all-MiniLM-L6-v2')

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

GRAPH_USER_NODE = os.getenv("GRAPH_USER_NODE", "Affan Syed")

class PersonaTrainRequest(BaseModel):
    name: str
    social_urls: List[str] = []

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extracts text content reliably based on file extension (supports PDF and TXT)."""
    text_content = ""
    ext = filename.lower().split(".")[-1]
    
    try:
        if ext == "pdf":
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        
    return text_content.strip()

def extract_facts_with_llm(text_content: str) -> list[dict]:
    """
    Uses the LLM to dynamically pull (relation, target) facts out of ANY
    uploaded text, instead of relying on a fixed keyword list. Returns a
    list of {"relation": "...", "target": "..."} dicts describing edges
    from the central user node.
    """
    excerpt = text_content[:6000]

    extraction_prompt = (
        "Extract factual attributes, interests, skills, goals, relationships, "
        "and experiences about the person described in the text below. "
        "Return ONLY a JSON array (no markdown, no commentary) of objects shaped like:\n"
        '[{"relation": "LIKES", "target": "Football"}, '
        '{"relation": "WORKS_WITH", "target": "Python"}]\n'
        "Rules:\n"
        "- relation must be an UPPER_SNAKE_CASE verb phrase (e.g. LIKES, WORKS_WITH, "
        "STUDIED_AT, PLAYS, AIMS_FOR, DEVELOPED)\n"
        "- target must be a short noun phrase (a specific entity, skill, place, or goal)\n"
        "- Extract at most 15 of the most meaningful facts\n"
        "- If nothing extractable is found, return []\n\n"
        f"TEXT:\n{excerpt}"
    )

    try:
        response = deepseek_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": "You are a precise information-extraction engine. Output valid JSON only."},
                {"role": "user", "content": extraction_prompt}
            ],
            stream=False
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.endswith("json"):
                raw = raw[:-4]

        facts = json.loads(raw)
        if not isinstance(facts, list):
            return []

        cleaned = []
        for f in facts:
            relation = str(f.get("relation", "")).strip().upper().replace(" ", "_")
            target = str(f.get("target", "")).strip()
            if relation and target and relation.replace("_", "").isalnum():
                cleaned.append({"relation": relation, "target": target})
        return cleaned
    except Exception as e:
        print(f"LLM fact extraction failed: {e}")
        return []


def save_profile_text_to_neo4j(filename: str, text_content: str, persona: str) -> dict:
    """Dynamically extracts entities/relationships from uploaded text and
    writes them into Neo4j tagged specifically to the target persona."""
    if not graph_db.driver:
        raise RuntimeError(
            "Neo4j driver is not connected (check Aura instance is running "
            "and NEO4J_URI/NEO4J_PASSWORD are correct)"
        )

    clean_persona = persona.strip() if persona else "My Personal Twin"

    facts = extract_facts_with_llm(text_content)

    with graph_db.driver.session() as session:
        session.run(
            "MERGE (u:Entity {name: $uname}) SET u.type = 'User'",
            uname=clean_persona
        )

        for fact in facts:
            graph_db.add_fact(clean_persona, fact["relation"], fact["target"])

        session.run(
            """
            MATCH (u:Entity {name: $uname})
            MERGE (d:Document {name: $fname})
            MERGE (u)-[:UPLOADED]->(d)
            """,
            uname=clean_persona, fname=filename
        )

    print(f"Mapped {len(facts)} fact(s) from '{filename}' into Neo4j graph for persona '{clean_persona}'.")
    return {"facts_added": len(facts), "facts": facts}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), persona: str = Form("My Personal Twin")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
            
        text_data = extract_text_from_file(file_path, file.filename)
        
        if not text_data:
            raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty")
        
        chunk_count = process_and_store_document(file_path, persona=persona)
        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="Could not index text chunks into vector database")

        graph_result = {"facts_added": 0, "facts": []}
        graph_error = None
        try:
            graph_result = save_profile_text_to_neo4j(file.filename, text_data, persona=persona)
        except Exception as e:
            graph_error = str(e)
            print(f"Neo4j sync error: {graph_error}")

        return {
            "filename": file.filename,
            "status": "success",
            "chunks_stored": chunk_count,
            "graph_facts_added": graph_result["facts_added"],
            "graph_facts": graph_result["facts"],
            "graph_error": graph_error,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-persona")
async def train_persona(request: PersonaTrainRequest):
    try:
        scraped_summaries = []
        for url in request.social_urls:
            scraped_summaries.append(f"Profile/Link: {url}")
            
        links_str = ", ".join(request.social_urls) if request.social_urls else "None provided"
        extra_context = " ".join(scraped_summaries)
        training_text = f"Persona Profile Name: {request.name}. Connected Professional/Social Links: {links_str}. {extra_context}"
        
        process_and_store_document(None, persona=request.name, direct_text=training_text)
        
        try:
            save_profile_text_to_neo4j(f"{request.name}_profile_links", training_text, persona=request.name)
        except Exception as e:
            print(f"Graph sync error during persona training: {e}")
        
        return {
            "status": "success",
            "message": f"Successfully initialized and trained persona '{request.name}' with {len(request.social_urls)} links."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_persona(
    message: str = Form(...),
    persona: str = Form("My Personal Twin"),
    file: Optional[UploadFile] = File(None)
):
    try:
        file_attachment_context = ""
        image_base64_data = None
        
        if file and file.filename:
            file_bytes = await file.read()
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            # Check if uploaded file is an image and encode to base64 for OpenRouter/DeepSeek vision format
            if file.content_type and file.content_type.startswith("image/"):
                encoded_string = base64.b64encode(file_bytes).decode('utf-8')
                image_base64_data = f"data:{file.content_type};base64,{encoded_string}"
                file_attachment_context = f"\n\n[User attached an image named: {file.filename}]"
            else:
                extracted_file_text = extract_text_from_file(file_path, file.filename)
                if extracted_file_text:
                    file_attachment_context = f"\n\n[Attached File Content from {file.filename}]:\n{extracted_file_text}"
                else:
                    file_attachment_context = f"\n\n[User attached a file named: {file.filename}]"

        full_message_query = message + file_attachment_context
        query_vector = chat_model.encode(full_message_query).tolist()
        
        search_result = db_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=4,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="persona",
                        match=MatchValue(value=persona)
                    )
                ]
            )
        )
        vector_contexts = [hit.payload["text"] for hit in search_result.points if hit.payload]
        
        graph_facts = graph_db.get_related_facts(persona)
        
        context_corpus = "\n".join(vector_contexts) if vector_contexts else "No document records found."
        graph_corpus = "\n".join(graph_facts) if graph_facts else "No graph records found."
        
        system_prompt = (
            f"CRITICAL INSTRUCTION: Your name and identity are strictly '{persona}'. "
            f"You must embody this specific persona named '{persona}' at all times. "
            f"If the user asks for your name, you must state that you are '{persona}'.\n\n"
            "STRICT FORMATTING RULE FOR CODE: If the user asks for programming code or code snippets, you MUST output ONLY the raw code. "
            "Do NOT include markdown block ticks (like ```), do NOT include explanations, do NOT include instructions on how to run it, do NOT include any comments, and do NOT include console prompt strings (like System.out.print or input prompts).\n\n"
            f"Here is context retrieved from the user's uploaded personal documents and records:\n{context_corpus}\n\n"
            f"Here are related relationship or entity facts:\n{graph_corpus}\n\n"
            "Guidelines:\n"
            f"- Always respond as '{persona}'.\n"
            "- If code is requested, provide ONLY the clean code with zero comments or input prompts."
        )

        # Construct message content supporting standard text or OpenAI/OpenRouter vision format
        if image_base64_data:
            user_content_payload = [
                {"type": "text", "text": full_message_query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64_data
                    }
                }
            ]
        else:
            user_content_payload = full_message_query

        response = deepseek_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_payload}
            ],
            stream=False
        )

        bot_reply = response.choices[0].message.content

        return {
            "response": bot_reply,
            "retrieved_context_count": len(vector_contexts),
            "graph_facts_count": len(graph_facts)
        }
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "Persona Twin dynamic backend is running"}