# Persona Twin

Persona Twin is a personal "digital twin" chatbot. You feed it your own documents (resume, notes, chat exports, personal profile info), and it uses retrieval-augmented generation (RAG) plus a knowledge graph to answer questions and chat *as you* — with your facts, preferences, and background baked in.

It combines:
- **Vector search** (Qdrant) over chunks of your uploaded documents, for semantic recall of what you've written.
- **Knowledge graph** (Neo4j) storing structured facts about you (likes, skills, goals, relationships) for quick factual lookups.
- **LLM generation** (via OpenRouter/DeepSeek) to produce persona-consistent responses grounded in the retrieved context.
- A simple **web UI** (`index.html`) for chatting with your persona.

## Features

- 📄 **Document ingestion** — upload PDFs, `.txt`, or `.md` files; text is extracted, chunked, embedded, and stored in Qdrant.
- 🧠 **Knowledge graph facts** — key facts about you are stored as entities/relationships in Neo4j and pulled into every chat response.
- 💬 **Persona-based chat** — define a persona name and chat with an LLM that responds in that persona's voice, grounded in your documents and graph facts.
- 🔎 **Context-aware answers** — each chat request retrieves relevant document chunks (vector search) and related facts (graph query) before generating a reply.

## Architecture

```
Browser (index.html)
        │
        ▼
   FastAPI (main.py)
        │
   ┌────┴────────────────┐
   ▼                     ▼
Qdrant (vector_db.py)   Neo4j (graph_db.py)
   │                     │
ingestion.py         seed_graph.py
(chunking, embedding)  (seed facts)
```

- **`main.py`** — FastAPI app exposing `/ingest`, `/train-persona`, and `/chat` endpoints.
- **`ingestion.py`** — extracts text from uploaded files (PDF/txt/md), chunks it, generates embeddings with `sentence-transformers`, and stores them in Qdrant.
- **`vector_db.py`** — Qdrant client setup and collection initialization (local, file-based storage in `qdrant_data/`).
- **`graph_db.py`** — Neo4j driver wrapper for reading/writing entity-relationship facts.
- **`seed_graph.py`** — one-off script to seed some initial facts into the graph.
- **`index.html`** — front-end chat interface (Tailwind + vanilla JS).

## Prerequisites

- Python 3.10+
- A [Neo4j](https://neo4j.com/) instance (AuraDB free tier works fine)
- An [OpenRouter](https://openrouter.ai/) API key (used here as `DEEPSEEK_API_KEY`)

## Setup

1. **Clone the repo and enter the project folder:**
   ```bash
   git clone <your-repo-url>
   cd persona-twin
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables.** Create a `.env` file in the project root:
   ```env
   DEEPSEEK_API_KEY=your_openrouter_api_key
   NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   ```

4. **(Optional) Seed some initial graph facts:**
   ```bash
   python seed_graph.py
   ```

5. **Run the API server:**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

6. **Open the frontend:** open `index.html` directly in your browser, or serve it with any static file server.

## API Endpoints

| Method | Endpoint         | Description                                                   |
|--------|------------------|-----------------------------------------------------------------|
| GET    | `/`              | Health check                                                   |
| POST   | `/ingest`        | Upload a document (PDF/txt/md) to embed into the vector store  |
| POST   | `/train-persona` | Register a persona name and optional social/profile links      |
| POST   | `/chat`          | Send a message and get a persona-grounded response              |

### Example: chat request
```json
POST /chat
{
  "message": "What are your career goals?",
  "persona": "My Personal Twin"
}
```

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** Qdrant (local, file-based)
- **Graph store:** Neo4j
- **LLM:** DeepSeek model via OpenRouter API
- **Frontend:** HTML, Tailwind CSS, vanilla JS

## Project Structure

```
persona-twin/
├── main.py            # FastAPI app & endpoints
├── ingestion.py        # Document parsing, chunking, embedding
├── vector_db.py         # Qdrant client & collection setup
├── graph_db.py           # Neo4j driver & graph queries
├── seed_graph.py          # Script to seed initial graph facts
├── index.html              # Chat UI
├── requirements.txt          # Python dependencies
├── data/                       # Uploaded source documents (gitignored)
└── qdrant_data/                 # Local vector DB storage (gitignored)
```

## Notes / Limitations

- This is a personal/experimental project — the ingestion rules and graph fact extraction (e.g. keyword matching for interests) are simple and tailored to the original author's data; you'll likely want to adapt them for your own use case.
- Qdrant runs in local, file-based mode by default (no separate server needed).
- CORS is currently open to all origins (`allow_origins=["*"]`) — tighten this before deploying publicly.

## License

Add a license of your choice (e.g. MIT) here.
