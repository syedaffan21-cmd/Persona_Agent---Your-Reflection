# Persona Twin

Persona Twin is a chat app where you create AI "personas" — each one can be
trained on your own writing, documents, or web links, and can even reply back
using a cloned voice of a real person (with their consent) instead of plain text.

Think of it as: **ChatGPT, but you can make several differently-trained
characters, and give each one its own voice.**

---

## ✨ What it can do

| Feature | Description |
|---|---|
| 🧑‍🤝‍🧑 Multiple personas | Create as many named personas as you want, each with its own chat history |
| 📚 Train a persona | Upload documents or paste web/social links so a persona "learns" a writing style |
| 🗣️ Voice cloning | Give a persona its own spoken voice (via Fish Audio) — one persona's voice never bleeds into another's |
| 🇮🇳 Hindi-friendly speech | Hinglish text (e.g. "kaise ho") is auto-converted to Hindi script before speaking, so it sounds natural instead of being read like English |
| ⚡ Faster replays | Already-spoken audio is cached in your browser, so clicking "Read Aloud" twice doesn't call the API twice |
| 🧠 Memory | Personas remember facts and traits using a knowledge graph + vector search |

---

## 🧱 How it's built

```
┌─────────────┐        ┌────────────────────┐        ┌────────────────┐
│  index.html │ ─────▶ │  main.py (FastAPI)  │ ─────▶ │  Fish Audio    │ (voice)
│  (Vercel)   │        │  (Render)            │ ─────▶ │  OpenRouter    │ (chat)
└─────────────┘        │                      │ ─────▶ │  Neo4j Aura    │ (memory)
                        └────────────────────┘ ─────▶ │  Qdrant (local)│ (search)
                                                         └────────────────┘
```

- **Frontend** — a single `index.html` file, no build step needed.
- **Backend** — `main.py`, a FastAPI server that handles chat, training, and voice.
- **Neo4j** — stores persona facts/traits as a graph.
- **Qdrant** — stores document text as searchable embeddings (runs locally, no account needed).

---

## 🚀 Getting started (run it on your own computer)

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install --with-deps chromium
```

### 2. Set up your secret keys

```bash
cp .env.example .env
```

Open `.env` and fill in each value. `.env.example` has a comment above every
line explaining where to get it. You'll need:

- An **OpenRouter** API key (for chat replies) → `DEEPSEEK_API_KEY`
- A **Fish Audio** API key (for voice cloning) → `FISH_AUDIO_API_KEY`
- A **Neo4j Aura** free database URI + password → `NEO4J_URI`, `NEO4J_PASSWORD`

> ⚠️ Never commit your real `.env` file. It's already in `.gitignore`.

### 3. Start the backend

```bash
uvicorn main:app --reload
```

It will run at `http://127.0.0.1:8000`.

### 4. Open the frontend

Open `index.html` in your browser (or right-click → "Open with Live Server"
in VS Code). Make sure the `API_BASE` variable near the top of the `<script>`
tag in `index.html` points to your backend's address.

---

## 🐳 Running with Docker instead

```bash
docker build -t persona-twin-backend .
docker run -p 8000:8000 --env-file .env -v "${PWD}/data:/app/data" persona-twin-backend
```

The `-v` part matters: it saves your data (like trained persona voices)
*outside* the container, so you don't lose it every time you restart.

---

 

---

## 🔑 Environment variables

| Variable | What it's for |
|---|---|
| `DEEPSEEK_API_KEY` | OpenRouter key — powers chat replies and Hindi text conversion |
| `FISH_AUDIO_API_KEY` | Powers voice cloning and text-to-speech |
| `FISH_AUDIO_TTS_MODEL` | Which Fish Audio voice quality tier to use (free or paid) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Connects to your Neo4j graph database |
| `GRAPH_USER_NODE` | The name of your default persona |
| `TWITTER_COOKIES_PATH` | Lets the app read a Twitter/X profile when training a persona from it |

Full details and sign-up links are in `.env.example`.

---

## 🔒 Please read before pushing to GitHub

- `twitter_cookies.json` and `raw_cookies.json` contain a **real, active login
  session** if you've filled them in — treat them exactly like a password.
  They're already excluded by `.gitignore`, but double-check before your
  first `git push` that they weren't accidentally staged.
- Only clone someone's voice if it's **your own voice**, or someone else's
  voice **with their clear permission**. This app doesn't block cloning any
  name you type in — it relies on you using it responsibly.

---

## 🧯 Common issues

| Problem | Likely cause |
|---|---|
| Voice playback fails with a `402` error | Your Fish Audio account has no API credit — top up at `fish.audio/app/developers` |
| Persona voice is gone after restarting | Local data folder wasn't saved — use the `-v` flag shown above, or a Render Persistent Disk |
| "Failed to connect to Neo4j" in logs | Double check `NEO4J_URI` / `NEO4J_PASSWORD` in your `.env` |
| Hinglish sounds like English when spoken | Should auto-convert to Hindi script — if it doesn't, check the backend logs for a transliteration error |
