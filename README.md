# Persona Twin

Build an AI "digital twin" of yourself (or anyone else) by feeding it your documents, chats, and social profiles. It learns your traits, facts, and speaking style, then talks back as you — in your own language and tone.

## How it works

1. **Train** a persona by uploading documents (PDFs, text files, chat exports) or pointing it at a social profile URL.
2. The backend extracts **facts** (stored in a Neo4j knowledge graph) and **personality traits** (stored per-persona), and keeps a **voice sample** of how that person actually writes/talks — prioritizing real chat messages and casual notes over formal documents, since those better reflect genuine speaking style.
3. **Chat** with the persona. It answers using its own facts/traits/voice, retrieves relevant context via vector search (Qdrant), and can pull in live external data (GitHub, Twitter/X, Instagram) mid-conversation when you share a link.

## Features

- Multi-persona support — create and switch between multiple digital twins
- Document ingestion: PDF, TXT, MD, images
- Web scraping for context: GitHub profiles, Twitter/X (via saved session cookies), Instagram (profile + individual posts, including photos and captions)
- Automatic trait extraction and deduplication (won't invent traits the source text doesn't support)
- Persona traits are manually editable (add/remove) from the sidebar
- Language-adaptive replies — mirrors the actual language/style the persona's creator uses (Hinglish, English, or otherwise), rather than a fixed hardcoded style
- Voice input/output (speech-to-text and text-to-speech in the browser)
- Chat history, per-chat pinning, editing, and renaming (stored locally in the browser)

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI (Python) |
| Knowledge graph | Neo4j Aura (facts, relationships) |
| Vector search | Qdrant (embedded/local mode) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Chat model | OpenRouter (`openrouter/free`), via the OpenAI Python SDK |
| Scraping | `httpx` + BeautifulSoup (GitHub, Instagram), Playwright/Chromium (Instagram fallback), `twifork` (Twitter/X, cookie-based) |
| Frontend | Single-file HTML/CSS/JS, no build step |

## Project structure

```
main.py            FastAPI app: all API endpoints, chat logic, scraping, trait extraction
graph_db.py         Neo4j connection and fact/trait graph operations
ingestion.py         Document parsing and vector embedding/storage
vector_db.py        Qdrant client setup and shared embedding model
index.html          Frontend (chat UI, persona/trait management)
requirements.txt    Python dependencies
Dockerfile           Container build (includes Playwright's Chromium)
```

## Local setup

**Requirements:** Python 3.11+, a Neo4j Aura instance (free tier works), an OpenRouter API key.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
playwright install chromium     # needed once, for Instagram scraping fallback
```

Create a `.env` file in the project root:

```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password
GRAPH_USER_NODE=Your Name
DEEPSEEK_API_KEY=sk-or-v1-your-openrouter-key
```

Run it:

```bash
uvicorn main:app --reload
```

Then open `index.html` directly in a browser. `API_BASE` near the top of its `<script>` block defaults to `http://localhost:8000` for local dev.

### Running with Docker instead

```bash
docker build -t persona-twin-backend .
docker run -p 8000:8000 --env-file .env persona-twin-backend
```

## Deploying

This app needs a real persistent server for the backend (embedded vector DB, headless browser scraping, streaming chat responses) — it can't run on serverless platforms like Vercel. The frontend, being a single static HTML file, deploys anywhere trivially.

- **Backend → [Render](https://render.com)**: connect this repo, let it build from the included `Dockerfile`, and add your `.env` values under the Environment tab. For Twitter/X scraping to keep working, add `twitter_cookies.json` as a **Secret File** in Render rather than committing it.
- **Frontend → [Vercel](https://vercel.com)**: import this repo, set the framework preset to "Other" with no build command, and deploy. Before deploying, update `API_BASE` in `index.html` to your live Render URL.

## Known limitations

- **OpenRouter free tier**: `openrouter/free` is capped at 50 requests/day (1000/day with $10+ credit added to your OpenRouter account). This is an account-wide limit — creating new accounts does not reset it. For real usage, add credits or switch to a paid model.
- **No persistent storage on free hosting tiers**: persona traits, voice samples, and the vector database currently live in local files/folders. Without a paid persistent disk, this resets on every redeploy or restart. Fine for testing; for a durable production setup, this data should move to a hosted database.
- **LinkedIn scraping is not supported.** LinkedIn aggressively blocks unauthenticated access with no public API equivalent; reliably scraping it would require using real login credentials programmatically, which risks the account and isn't something this project attempts.
- **Instagram/Twitter scraping relies on undocumented or session-based access**, not official APIs, so it can break or get rate-limited without notice.
- **No authentication on the API.** Anything reachable at the backend's URL can chat, train, or delete personas. Fine for personal/local use; add an access-key check before exposing this publicly to strangers.
