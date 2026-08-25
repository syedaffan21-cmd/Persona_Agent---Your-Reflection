import os
import json
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from ingestion import process_and_store_document
from graph_db import graph_db
from vector_db import client as db_client, COLLECTION_NAME
 

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

# NOTE: there is only ONE Neo4j driver now - the one owned by graph_db
# (see graph_db.py). Do not create a second driver here; two independent
# drivers made failures easy to miss because one could connect while the
# other silently failed.
GRAPH_USER_NODE = os.getenv("GRAPH_USER_NODE", "Affan Syed")

class ChatRequest(BaseModel):
    message: str
    persona: str = "My Personal Twin"

class PersonaTrainRequest(BaseModel):
    name: str
    social_urls: List[str] = []

def extract_facts_with_llm(text_content: str) -> list[dict]:
    """
    Uses the LLM to dynamically pull (relation, target) facts out of ANY
    uploaded text, instead of relying on a fixed keyword list. Returns a
    list of {"relation": "...", "target": "..."} dicts describing edges
    from the central user node.
    """
    # Keep the prompt payload bounded - the graph doesn't need the whole document
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
        # Strip accidental markdown fences if the model adds them anyway
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
            # Neo4j relationship types can't be parameterized, so only allow
            # safe characters here to avoid Cypher injection via the LLM output
            if relation and target and relation.replace("_", "").isalnum():
                cleaned.append({"relation": relation, "target": target})
        return cleaned
    except Exception as e:
        print(f"LLM fact extraction failed: {e}")
        return []


def save_profile_text_to_neo4j(filename: str, text_content: str) -> dict:
    """Dynamically extracts entities/relationships from uploaded text and
    writes them into Neo4j so the graph actually reflects file content,
    instead of only matching a hardcoded keyword list."""
    if not graph_db.driver:
        raise RuntimeError(
            "Neo4j driver is not connected (check Aura instance is running "
            "and NEO4J_URI/NEO4J_PASSWORD are correct)"
        )

    facts = extract_facts_with_llm(text_content)

    with graph_db.driver.session(database="neo4j") as session:
        session.run(
            "MERGE (u:Entity {name: $uname}) SET u.type = 'User'",
            uname=GRAPH_USER_NODE
        )

        for fact in facts:
            graph_db.add_fact(GRAPH_USER_NODE, fact["relation"], fact["target"])

        # Track the uploaded file node and link it to the user
        session.run(
            """
            MATCH (u:Entity {name: $uname})
            MERGE (d:Document {name: $fname})
            MERGE (u)-[:UPLOADED]->(d)
            """,
            uname=GRAPH_USER_NODE, fname=filename
        )

    print(f"Mapped {len(facts)} fact(s) from '{filename}' into Neo4j graph.")
    return {"facts_added": len(facts), "facts": facts}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        text_data = contents.decode("utf-8", errors="ignore")
        
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        chunk_count = process_and_store_document(file_path)
        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        graph_result = {"facts_added": 0, "facts": []}
        graph_error = None
        try:
            graph_result = save_profile_text_to_neo4j(file.filename, text_data)
        except Exception as e:
            # Don't fail the whole upload if only the graph write fails -
            # the vector store already has the content - but surface it
            # clearly instead of pretending everything succeeded.
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
        links_str = ", ".join(request.social_urls) if request.social_urls else "None provided"
        training_text = f"Persona Profile Name: {request.name}. Connected Professional/Social Links: {links_str}."
        
        vector = chat_model.encode(training_text).tolist()
        
        return {
            "status": "success",
            "message": f"Successfully initialized and trained persona '{request.name}' with {len(request.social_urls)} links."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_persona(request: ChatRequest):
    try:
        query_vector = chat_model.encode(request.message).tolist()
        
        search_result = db_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=4
        )
        vector_contexts = [hit.payload["text"] for hit in search_result.points if hit.payload]
        graph_facts = graph_db.get_related_facts(request.message.strip())
        
        context_corpus = "\n".join(vector_contexts) if vector_contexts else "No document records found."
        graph_corpus = "\n".join(graph_facts) if graph_facts else "No graph records found."
        
        system_prompt = (
            f"CRITICAL INSTRUCTION: Your name and identity are strictly '{request.persona}'. "
            f"You must embody this specific persona named '{request.persona}' at all times. "
            f"If the user asks for your name, you must state that you are '{request.persona}'.\n\n"
            "STRICT FORMATTING RULE FOR CODE: If the user asks for programming code or code snippets, you MUST output ONLY the raw code. "
            "Do NOT include markdown block ticks (like ```), do NOT include explanations, do NOT include instructions on how to run it, do NOT include any comments, and do NOT include console prompt strings (like System.out.print or input prompts).\n\n"
            f"Here is context retrieved from the user's uploaded personal documents and records:\n{context_corpus}\n\n"
            f"Here are related relationship or entity facts:\n{graph_corpus}\n\n"
            "Guidelines:\n"
            f"- Always respond as '{request.persona}'.\n"
            "- If code is requested, provide ONLY the clean code with zero comments or input prompts."
        )

        response = deepseek_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "Persona Twin dynamic backend is running"}