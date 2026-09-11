import os
import json
import shutil
import io
import base64
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from ingestion import process_and_store_document
from graph_db import graph_db
from vector_db import client as db_client, COLLECTION_NAME, encode_single
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pypdf
import PIL.Image
import re
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup
from twikit import Client
from playwright.async_api import async_playwright

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

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

def safe_upload_path(filename: str) -> str:
    """Strips any directory components from a client-supplied filename before
    joining it to UPLOAD_DIR, so a name like '../../main.py' can't be used to
    write outside the intended folder (path traversal)."""
    base_name = os.path.basename((filename or "upload").replace("\\", "/"))
    base_name = base_name.lstrip(".") or "upload"
    return os.path.join(UPLOAD_DIR, base_name)

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

GRAPH_USER_NODE = os.getenv("GRAPH_USER_NODE", "Affan Syed")

class PersonaTrainRequest(BaseModel):
    name: str
    social_urls: List[str] = []

class TraitAddRequest(BaseModel):
    trait: str

TRAITS_FILE = os.path.join(UPLOAD_DIR, "persona_traits.json")

def init_traits_store():
    if not os.path.exists(TRAITS_FILE):
        default_traits = {
            "Affan": [
                "Calm, direct, and candid speaking style",
                "Street-smart and conversational with smooth explanations in Hinglish",
                "Motorcycle enthusiast who loves weekend breakfast rides",
                "Passionate about football, fitness, and gym workouts",
                "Fond of nature, mountains, and watching sunsets",
                "Values authenticity, discipline, and strongly dislikes dishonesty",
                "Enthusiastic about cyber security and artificial intelligence"
            ],
            "My Personal Twin": [
                "Helpful, thoughtful, and authentic conversational style",
                "Curious learner with strong analytical problem solving",
                "Direct and articulate communicator"
            ]
        }
        try:
            with open(TRAITS_FILE, "w", encoding="utf-8") as f:
                json.dump(default_traits, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error initializing persona traits file: {e}")

init_traits_store()

def load_persona_traits_from_disk() -> Dict[str, List[str]]:
    if os.path.exists(TRAITS_FILE):
        try:
            with open(TRAITS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading persona traits: {e}")
    return {}

def save_persona_traits_to_disk(data: Dict[str, List[str]]):
    try:
        with open(TRAITS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving persona traits: {e}")

VOICE_SAMPLES_FILE = os.path.join(UPLOAD_DIR, "persona_voice_samples.json")

def _voice_sample_priority(filename: str) -> int:
    """Ranks a source document by how well it likely reflects how a person actually
    talks. Chat exports and casual notes are far more representative of real speaking
    style/language than formal documents like resumes or structured profile sheets,
    which are usually written in stiff, formal English regardless of how someone
    actually chats day-to-day. Live messages the creator types while chatting with
    their own persona rank highest of all -- there's no more direct signal of their
    real voice than that."""
    name = filename.lower()
    if name == "live_chat_input":
        return 4
    if "chat" in name or "whatsapp" in name or "convo" in name or "conversation" in name:
        return 3
    if "likes" in name or "notes" in name or "diary" in name:
        return 2
    if name.endswith(".txt") or name.endswith(".md"):
        return 1
    return 0  # pdf, docx, resumes, structured profile sheets, images, etc.

def load_voice_samples() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(VOICE_SAMPLES_FILE):
        try:
            with open(VOICE_SAMPLES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading persona voice samples: {e}")
    return {}

def save_voice_samples(data: Dict[str, Dict[str, Any]]):
    try:
        with open(VOICE_SAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving persona voice samples: {e}")

def update_persona_voice_sample(persona: str, filename: str, text_content: str, accumulate: bool = False):
    """Keeps a short excerpt of whichever ingested source best reflects this
    persona's authentic writing/speaking style, so the chat prompt can mirror the
    creator's real language instead of guessing or defaulting to English.
    When accumulate=True (used for live chat messages), new text is appended to the
    existing same-priority sample rather than replacing it outright, so it builds up
    a richer picture across multiple messages instead of only ever reflecting the
    single most recent one."""
    persona_clean = persona.strip()
    text_clean = (text_content or "").strip()
    if not persona_clean or len(text_clean) < 20:
        return
    priority = _voice_sample_priority(filename)
    store = load_voice_samples()
    existing = store.get(persona_clean)

    if existing and existing.get("priority", -1) > priority:
        return  # a higher-quality source is already stored, don't overwrite it

    if accumulate and existing and existing.get("priority") == priority:
        combined = (existing.get("sample", "") + "\n" + text_clean).strip()
        sample = combined[-700:]
    else:
        # Skip likely boilerplate (e.g. WhatsApp export headers) at the very start of the file
        offset = 200 if len(text_clean) > 400 else 0
        sample = text_clean[offset:offset + 700].strip()

    store[persona_clean] = {"sample": sample, "priority": priority, "source": filename}
    save_voice_samples(store)

def get_persona_voice_sample(persona: str) -> Optional[str]:
    store = load_voice_samples()
    entry = store.get(persona.strip())
    return entry.get("sample") if entry else None

def purge_persona_voice_sample(persona: str):
    store = load_voice_samples()
    if persona.strip() in store:
        del store[persona.strip()]
        save_voice_samples(store)

def get_all_traits_for_persona(persona: str) -> List[str]:
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    local_traits = disk_data.get(persona_clean, [])
    graph_traits = graph_db.get_persona_traits(persona_clean) if graph_db.driver else []
    
    seen = set()
    combined = []
    for t in local_traits + graph_traits:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            combined.append(t_clean)
    return combined

_TRAIT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "with",
    "who", "is", "are", "style", "their", "they", "it", "for", "about", "very",
}

def _trait_tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _TRAIT_STOPWORDS}

def _is_near_duplicate_trait(candidate: str, existing_list: List[str], threshold: float = 0.5) -> bool:
    cand_tokens = _trait_tokens(candidate)
    if not cand_tokens:
        return False
    for e in existing_list:
        e_tokens = _trait_tokens(e)
        if not e_tokens:
            continue
        overlap = len(cand_tokens & e_tokens) / min(len(cand_tokens), len(e_tokens))
        if overlap >= threshold:
            return True
    return False

def store_traits_for_persona(persona: str, new_traits: List[str]) -> List[str]:
    persona_clean = persona.strip()
    if not persona_clean:
        return []
    disk_data = load_persona_traits_from_disk()
    existing = disk_data.get(persona_clean, [])
    for t in new_traits:
        t_clean = t.strip()
        if not t_clean:
            continue
        if _is_near_duplicate_trait(t_clean, existing):
            continue
        existing.append(t_clean)
        if graph_db.driver:
            graph_db.add_trait(persona_clean, t_clean)
    disk_data[persona_clean] = existing
    save_persona_traits_to_disk(disk_data)
    return existing

def remove_trait_from_persona(persona: str, trait_to_remove: str) -> List[str]:
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    existing = disk_data.get(persona_clean, [])
    updated = [t for t in existing if t.lower() != trait_to_remove.strip().lower()]
    disk_data[persona_clean] = updated
    save_persona_traits_to_disk(disk_data)
    if graph_db.driver:
        graph_db.delete_trait(persona_clean, trait_to_remove)
    return updated

def purge_persona_traits(persona: str):
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    if persona_clean in disk_data:
        del disk_data[persona_clean]
        save_persona_traits_to_disk(disk_data)
    if graph_db.driver:
        graph_db.delete_persona_traits(persona_clean)

class TwitterScraper:
    def __init__(self, cookies_file: Optional[str] = None):
        # Checks, in order: an explicit path (if passed), a TWITTER_COOKIES_PATH
        # env var (for hosts with unusual mount locations), Render's
        # "Secret Files" convention (/etc/secrets/<name>), and finally the
        # plain local file (for running outside Docker). First one that
        # actually exists on disk wins.
        candidates = [
            cookies_file,
            os.getenv("TWITTER_COOKIES_PATH"),
            "/etc/secrets/twitter_cookies.json",
            "twitter_cookies.json",
        ]
        self.cookies_file = next((p for p in candidates if p and os.path.exists(p)), None)

    def _make_client(self) -> Client:
        try:
            return Client("en-US", impersonate="chrome124")
        except Exception:
            return Client("en-US")

    async def _authenticated_client(self) -> Client:
        if not self.cookies_file or not os.path.exists(self.cookies_file):
            raise RuntimeError(f"Cookie file '{self.cookies_file}' not found.")
        client = self._make_client()
        client.load_cookies(self.cookies_file)
        if not await client.is_logged_in():
            raise RuntimeError("Twitter cookies are expired or invalid.")
        return client

    async def get_user_profile(self, username: str, tweet_count: int = 20) -> Dict[str, Any]:
        try:
            client = await self._authenticated_client()
            clean_username = username.strip().replace("@", "").split("?")[0]
            if not clean_username:
                return {"status": "error", "message": "Empty username."}

            user = await client.get_user_by_screen_name(clean_username)
            if not user:
                return {"status": "error", "message": f"User @{clean_username} not found."}

            tweets = await client.get_user_tweets(user.id, "Tweets", count=min(max(tweet_count, 1), 40))

            scraped_content = [
                f"Profile Name: {user.name} (@{user.screen_name})",
                f"Bio: {user.description or 'No bio'}",
                "Recent Tweets:"
            ]
            for tweet in tweets:
                text = getattr(tweet, "full_text", None) or tweet.text
                created = getattr(tweet, "created_at", "")
                scraped_content.append(f"- [{created}] {text}")

            return {"status": "success", "data": "\n".join(scraped_content)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search_tweets(self, query: str, limit: int = 10) -> List[str]:
        return []

twitter_client = TwitterScraper()

def extract_text_from_file(file_path: str, filename: str) -> str:
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

async def scrape_github_profile(url: str) -> str:
    try:
        clean_url = url.rstrip("/")
        parts = clean_url.split("/")
        username = parts[-1] if "github.com" in parts else parts[-2] if len(parts) > 1 else ""
        if not username:
            return "Invalid GitHub URL"
            
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
            profile_resp = await client.get(f"https://api.github.com/users/{username}", headers=headers)
            bio = "No bio found"
            fullname = username
            public_repos_count = 0
            if profile_resp.status_code == 200:
                p_data = profile_resp.json()
                bio = p_data.get("bio") or "No bio found"
                fullname = p_data.get("name") or username
                public_repos_count = p_data.get("public_repos", 0)

            repos_resp = await client.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated", headers=headers)
            projects = []
            repo_names = []
            if repos_resp.status_code == 200:
                repos_data = repos_resp.json()
                for repo in repos_data:
                    r_name = repo.get("name", "Unknown")
                    repo_names.append(r_name)
                    r_desc = repo.get("description") or "No desc"
                    r_lang = repo.get("language") or "Mixed"
                    projects.append(f"- {r_name} ({r_lang}): {r_desc}")
            
            projects_text = "\n".join(projects) if projects else "No public repositories found."
            return f"GitHub User: {fullname} (@{username})\nTotal Public Repositories: {public_repos_count}\nBio: {bio}\n\nRepositories:\n{projects_text}"
    except Exception as e:
        return f"Could not retrieve GitHub profile details: {e}"

async def scrape_linkedin_profile(url: str) -> str:
    return "LinkedIn profile public scraping requires authentication."

async def scrape_twitter_profile(url: str) -> str:
    handle = url.strip().split("/")[-1].replace("@", "").split("?")[0]
    profile_data = await twitter_client.get_user_profile(handle)
    if profile_data.get("status") == "success":
        return f"Twitter/X Profile Data for @{handle}:\n{profile_data.get('data')}"
    return f"Could not retrieve tweets for @{handle}."

async def scrape_instagram_post(url: str) -> str:
    """Scrapes a single Instagram post/reel. Instagram still populates Open Graph
    meta tags (image + caption) for individual public post pages even when logged
    out, so this works without needing the profile-level JSON endpoint."""
    clean_url = url.split("?")[0].rstrip("/")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(clean_url, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")

            og_image = soup.find("meta", property="og:image")
            og_desc = soup.find("meta", property="og:description")
            og_title = soup.find("meta", property="og:title")

            image_url = og_image.get("content") if og_image else "No image found"
            caption = og_desc.get("content") if og_desc else "No caption found"
            title = og_title.get("content") if og_title else ""

            if resp.status_code != 200 or (not og_image and not og_desc):
                return f"Could not access Instagram post ({clean_url}) — it may be private, deleted, or blocked by Instagram. Status: {resp.status_code}"

            return (
                f"Instagram Post ({clean_url}):\n"
                f"Title: {title}\n"
                f"Image: {image_url}\n"
                f"Caption: {caption}"
            )
    except Exception as e:
        return f"Could not scrape Instagram post {clean_url}: {e}"

async def scrape_instagram_profile(url: str, max_posts: int = 12) -> str:
    clean_url = url.rstrip("/")
    parts = [p for p in clean_url.split("/") if p]
    username = parts[-1] if parts else "unknown"
    if username.lower() in ["reel", "p", "explore", "stories"]:
        username = parts[-2] if len(parts) > 1 else "unknown"

    # --- Attempt 1: Instagram's public web-profile JSON endpoint ---
    # This is the same endpoint most open-source Instagram scrapers rely on: it
    # returns profile info plus recent post nodes (image URL + caption) for public
    # accounts without needing a logged-in session. Instagram may still rate-limit
    # or block it depending on IP/volume, so we always fall back below if it fails.
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "x-ig-app-id": "936619743392459",
            "Accept": "*/*",
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}",
                headers=headers
            )
            if resp.status_code == 200:
                data = resp.json()
                user = (data.get("data") or {}).get("user")
                if user:
                    bio = user.get("biography") or "No bio"
                    full_name = user.get("full_name") or username
                    follower_count = (user.get("edge_followed_by") or {}).get("count", "N/A")
                    is_private = user.get("is_private", False)

                    media_edges = ((user.get("edge_owner_to_timeline_media") or {}).get("edges")) or []
                    posts_summary = []
                    for edge in media_edges[:max_posts]:
                        node = edge.get("node", {})
                        caption_edges = ((node.get("edge_media_to_caption") or {}).get("edges")) or []
                        caption = caption_edges[0]["node"]["text"] if caption_edges else "(no caption)"
                        image_url = node.get("display_url", "")
                        shortcode = node.get("shortcode", "")
                        post_url = f"https://www.instagram.com/p/{shortcode}/" if shortcode else ""
                        like_count = (
                            (node.get("edge_liked_by") or {}).get("count")
                            or (node.get("edge_media_preview_like") or {}).get("count")
                            or "N/A"
                        )
                        media_type = "Video" if node.get("is_video") else "Photo"
                        posts_summary.append(
                            f"- {post_url}\n"
                            f"  Type: {media_type}\n"
                            f"  Image: {image_url}\n"
                            f"  Likes: {like_count}\n"
                            f"  Caption: {caption}"
                        )

                    if is_private and not posts_summary:
                        return (
                            f"Instagram Profile @{username} ({url}) is PRIVATE.\n"
                            f"Name: {full_name}\nBio: {bio}\nNo public posts are accessible."
                        )

                    posts_text = "\n\n".join(posts_summary) if posts_summary else "No public posts found."
                    return (
                        f"Instagram Public Profile Data for @{username} ({url}):\n"
                        f"Name: {full_name}\nFollowers: {follower_count}\nBio: {bio}\n\n"
                        f"Recent Posts ({len(posts_summary)} of {len(media_edges)} fetched):\n{posts_text}"
                    )
    except Exception as e:
        print(f"Instagram profile JSON endpoint failed for @{username}: {e}")

    # --- Attempt 2 (fallback): render the page and read Open Graph meta tags ---
    # Used when the JSON endpoint above is blocked/rate-limited by Instagram.
    # Less detailed (no per-post images/captions), but keeps the tool from
    # returning nothing at all.
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, timeout=30000)
            await page.wait_for_timeout(3000)

            html = await page.content()
            await browser.close()

            soup = BeautifulSoup(html, 'html.parser')
            og_title = soup.find("meta", property="og:title")
            og_desc = soup.find("meta", property="og:description")
            og_image = soup.find("meta", property="og:image")

            title_text = og_title.get("content") if og_title else f"Instagram Profile: {username}"
            desc_text = og_desc.get("content") if og_desc else "No public description available."
            image_line = f"\nProfile Image: {og_image.get('content')}" if og_image else ""

            return (
                f"Instagram Public Profile Data for @{username} ({url}) [limited data — JSON endpoint blocked]:\n"
                f"Name/Title: {title_text}\nBio/Details: {desc_text}{image_line}\n"
                f"Note: Per-post photos and captions were unavailable via this fallback method; "
                f"try again later or scrape individual post URLs (instagram.com/p/...) directly."
            )
    except Exception as e:
        return f"Target Instagram Profile: @{username} (URL: {url}). Scrape note: {e}"

async def scrape_url_content(url: str) -> str:
    if "github.com" in url.lower():
        return await scrape_github_profile(url)
    if "linkedin.com" in url.lower():
        return await scrape_linkedin_profile(url)
    if "twitter.com" in url.lower() or "x.com" in url.lower():
        return await scrape_twitter_profile(url)
    if "instagram.com" in url.lower():
        lowered = url.lower()
        if "/p/" in lowered or "/reel/" in lowered or "/tv/" in lowered:
            return await scrape_instagram_post(url)
        return await scrape_instagram_profile(url)

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                return f"Could not retrieve content from URL (Status code: {response.status_code})"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator=' ', strip=True)
            return text[:4000]
    except Exception as e:
        return f"Could not retrieve content from the URL: {e}"

async def scrape_photos_from_url(url: str) -> List[str]:
    return []

def _parse_json_loose(raw: str):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                return None
    return None

def extract_facts_and_traits_with_llm(text_content: str, existing_traits: Optional[List[str]] = None) -> dict:
    excerpt = text_content[:6000]
    existing_traits = existing_traits or []
    existing_traits_block = ""
    if existing_traits:
        existing_traits_block = (
            "\n\nTraits already recorded for this person from earlier documents:\n"
            + "\n".join(f"- {t}" for t in existing_traits) + "\n"
        )
    extraction_prompt = (
        "You are extracting information about a real person from a document about them. "
        "Analyze the TEXT below and extract two distinct categories:\n"
        "1. 'facts': Factual attributes, relationships, skills, education, projects, or experiences.\n"
        "2. 'traits': Personality traits, communication style, tone of voice, behavioral habits, core values.\n"
        f"{existing_traits_block}\n"
        "Return ONLY a JSON object (no markdown, no commentary) with this exact structure:\n"
        "{\n"
        '  "facts": [\n'
        '    {"relation": "LIKES", "target": "Football"},\n'
        '    {"relation": "WORKS_WITH", "target": "Python"}\n'
        "  ],\n"
        '  "traits": [\n'
        '    "Calm and direct communication style"\n'
        "  ]\n"
        "}\n\n"
        f"TEXT:\n{excerpt}"
    )
    try:
        response = deepseek_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": "You are a precise extraction engine. Output valid JSON only."},
                {"role": "user", "content": extraction_prompt}
            ],
            stream=False,
            timeout=15
        )
        raw = response.choices[0].message.content.strip()
        data = _parse_json_loose(raw)

        if isinstance(data, list):
            data = {"facts": data, "traits": []}
        elif not isinstance(data, dict):
            return {"facts": [], "traits": []}
            
        raw_facts = data.get("facts", [])
        cleaned_facts = []
        if isinstance(raw_facts, list):
            for f in raw_facts:
                if isinstance(f, dict):
                    relation = str(f.get("relation", "")).strip().upper().replace(" ", "_")
                    target = str(f.get("target", "")).strip()
                    if relation and target:
                        cleaned_facts.append({"relation": relation, "target": target})
                        
        raw_traits = data.get("traits", [])
        cleaned_traits = []
        for t in raw_traits if isinstance(raw_traits, list) else []:
            t_str = str(t).strip()
            if t_str and not _is_near_duplicate_trait(t_str, existing_traits) and not _is_near_duplicate_trait(t_str, cleaned_traits):
                cleaned_traits.append(t_str)

        return {"facts": cleaned_facts, "traits": cleaned_traits}
    except RateLimitError as e:
        print(f"LLM extraction skipped: OpenRouter free-tier daily limit hit ({e})")
        return {"facts": [], "traits": []}
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return {"facts": [], "traits": []}

def extract_facts_with_llm(text_content: str) -> list[dict]:
    return extract_facts_and_traits_with_llm(text_content).get("facts", [])

def save_profile_text_to_neo4j(filename: str, text_content: str, persona: str) -> dict:
    clean_persona = persona.strip() if persona else "My Personal Twin"
    existing_traits = get_all_traits_for_persona(clean_persona)
    extracted = extract_facts_and_traits_with_llm(text_content, existing_traits=existing_traits)
    facts = extracted.get("facts", [])
    traits = extracted.get("traits", [])
    
    store_traits_for_persona(clean_persona, traits)
    update_persona_voice_sample(clean_persona, filename, text_content)
    
    if graph_db.driver:
        try:
            with graph_db.driver.session() as session:
                session.run(
                    "MERGE (u:Entity {name: $uname}) SET u.type = 'User'",
                    uname=clean_persona
                )
                for fact in facts:
                    graph_db.add_fact(clean_persona, fact["relation"], fact["target"])
                for trait in traits:
                    graph_db.add_trait(clean_persona, trait)
                session.run(
                    """
                    MATCH (u:Entity {name: $uname})
                    MERGE (d:Document {name: $fname})
                    MERGE (u)-[:UPLOADED]->(d)
                    """,
                    uname=clean_persona, fname=filename
                )
        except Exception as e:
            print(f"Neo4j sync warning: {e}")
        
    return {
        "facts_added": len(facts),
        "facts": facts,
        "traits_added": len(traits),
        "traits": traits
    }

def consolidate_persona_traits_with_llm(persona: str) -> List[str]:
    persona_clean = persona.strip()
    current = get_all_traits_for_persona(persona_clean)
    return current

@app.post("/persona/{name}/traits/consolidate")
async def consolidate_persona_traits_endpoint(name: str):
    cleaned = consolidate_persona_traits_with_llm(name)
    return {"status": "success", "persona": name, "traits": cleaned}

@app.get("/persona/{name}/traits")
async def get_persona_traits_endpoint(name: str):
    traits = get_all_traits_for_persona(name)
    return {"persona": name, "traits": traits}

@app.post("/persona/{name}/traits")
async def add_persona_trait_endpoint(name: str, body: TraitAddRequest):
    clean_trait = body.trait.strip()
    if not clean_trait:
        raise HTTPException(status_code=400, detail="Trait cannot be empty")
    traits = store_traits_for_persona(name, [clean_trait])
    return {"status": "success", "persona": name, "traits": traits}

@app.delete("/persona/{name}/traits/{trait}")
async def delete_persona_trait_endpoint(name: str, trait: str):
    traits = remove_trait_from_persona(name, trait)
    return {"status": "success", "persona": name, "traits": traits}

@app.delete("/persona/{name}")
async def delete_persona_data(name: str):
    try:
        clean_name = name.strip()
        try:
            db_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="persona", match=MatchValue(value=clean_name))]
                )
            )
        except Exception as q_err:
            print(f"Qdrant deletion warning: {q_err}")
        if graph_db.driver:
            with graph_db.driver.session() as session:
                session.run("MATCH (u:Entity {name: $uname}) DETACH DELETE u", uname=clean_name)
        purge_persona_traits(clean_name)
        purge_persona_voice_sample(clean_name)
        return {"status": "success", "message": f"Successfully purged persona '{clean_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), persona: str = Form("My Personal Twin")):
    file_path = safe_upload_path(file.filename)
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.")
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        text_data = extract_text_from_file(file_path, file.filename)
        if not text_data:
            raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty")
        chunk_count = process_and_store_document(file_path, persona=persona)
        graph_result = save_profile_text_to_neo4j(file.filename, text_data, persona=persona)
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_stored": chunk_count,
            "graph_facts_added": graph_result.get("facts_added", 0),
            "graph_facts": graph_result.get("facts", []),
            "traits_added": graph_result.get("traits_added", 0),
            "traits": graph_result.get("traits", []),
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
            cleaned_url = url.strip()
            if not cleaned_url.startswith("http://") and not cleaned_url.startswith("https://"):
                cleaned_url = "https://" + cleaned_url
            try:
                scraped_text = await scrape_url_content(cleaned_url)
                scraped_summaries.append(f"Source URL ({cleaned_url}): {scraped_text[:3500]}")
            except Exception:
                scraped_summaries.append(f"Profile/Link: {cleaned_url}")
                
        extra_context = "\n\n".join(scraped_summaries) if scraped_summaries else f"Persona entity: {request.name}"
        training_text = f"Persona Profile Name: {request.name}.\n\nContent:\n{extra_context}"
        
        process_and_store_document(None, persona=request.name, direct_text=training_text)
        graph_result = save_profile_text_to_neo4j(f"{request.name}_profile", training_text, persona=request.name)
            
        return {
            "status": "success",
            "message": f"Successfully processed and trained persona '{request.name}'.",
            "facts_added": graph_result.get("facts_added", 0),
            "traits_added": graph_result.get("traits_added", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_persona(
    message: str = Form(...),
    persona: str = Form("My Personal Twin"),
    history: str = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        actual_message = message
        external_scraped_context = ""
        url_match = re.search(r'\[Scrape URL:\s*(https?://[^\s\]]+)\]', message)
        if url_match:
            target_url = url_match.group(1)
            scraped_text = await scrape_url_content(target_url)
            external_scraped_context = (
                f"\n\n=== EXTERNAL REFERENCE DATA — NOT YOU ===\n"
                f"Everything below until the closing marker is about a DIFFERENT real person the user "
                f"asked you to look up. It is NOT you, has nothing to do with your own identity, traits, "
                f"or facts, and must never be blended with them.\n\n"
                f"{scraped_text}\n\n"
                f"=== END EXTERNAL REFERENCE DATA ===\n"
                f"Reminder: when discussing the person above, refer to them by their own name/handle in "
                f"third person. Never say \"I\" or \"my\" about anything in that block, and never combine "
                f"their stats/achievements/identity with your own in the same sentence."
            )

        file_attachment_context = ""
        image_base64_data = None
        if file and file.filename:
            file_bytes = await file.read()
            if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.")
            file_path = safe_upload_path(file.filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
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

        full_message_query = actual_message + file_attachment_context + external_scraped_context
        if len(actual_message.strip()) >= 20:
            update_persona_voice_sample(persona, "live_chat_input", actual_message, accumulate=True)
        query_vector = encode_single(full_message_query)
        
        search_result = db_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
            query_filter=Filter(
                must=[FieldCondition(key="persona", match=MatchValue(value=persona))]
            )
        )
        vector_contexts = [hit.payload["text"] for hit in search_result.points if hit.payload]
        graph_facts = graph_db.get_related_facts(persona)
        persona_traits = get_all_traits_for_persona(persona)
        
        context_corpus = "\n".join(vector_contexts) if vector_contexts else "No document records found."
        graph_corpus = "\n".join(graph_facts) if graph_facts else "No graph records found."
        
        traits_section = ""
        if persona_traits:
            traits_list_str = "\n".join([f"- {t}" for t in persona_traits])
            traits_section = f"PERSONALITY TRAITS:\n{traits_list_str}\n\n"
        
        voice_sample = get_persona_voice_sample(persona)
        if voice_sample:
            style_instruction = (
                f"CRITICAL LANGUAGE & STYLE REQUIREMENT:\n"
                f"- Below is an authentic excerpt of how '{persona}' actually writes/talks in real life, taken "
                f"directly from their own messages or documents. Study its language (English, Hinglish, or any "
                f"other language or mix), tone, slang, and sentence rhythm, and reply using that SAME language "
                f"and style. Do not switch to plain formal English unless that is what this excerpt actually shows.\n"
                f"--- AUTHENTIC SAMPLE FROM '{persona}' START ---\n{voice_sample}\n--- SAMPLE END ---\n"
                f"- If the user's latest message below is written in a different language than the sample, "
                f"prioritize matching the sample's language, since that is this persona's genuine voice.\n"
            )
        else:
            style_instruction = (
                f"Talk entirely in character as '{persona}' based on their records.\n"
                f"- No authentic writing sample has been collected for '{persona}' yet, so mirror the language "
                f"the user is currently writing in below (e.g. if they write in Hinglish, reply in Hinglish; "
                f"if in English, reply in English; if in another language, reply in that language).\n"
            )

        system_prompt = (
            f"You are strictly '{persona}'. Embody this persona completely.\n\n"
            f"{traits_section}"
            f"{style_instruction}\n"
            "STRICT FORMATTING PROHIBITIONS:\n"
            "- NEVER USE ASTERISKS (*) OR HTML TAGS LIKE <b>, </b>.\n"
            "- Use UPPERCASE letters for emphasis instead.\n"
            "- Write in continuous, flowing conversational paragraphs. No bullet points.\n\n"
            "IF THE USER'S MESSAGE INCLUDES A BLOCK MARKED 'EXTERNAL REFERENCE DATA':\n"
            "- That block describes a completely different real person, unrelated to you. Answer the "
            "user's question about that person factually and in third person, in your own natural voice.\n"
            "- Do NOT adopt their stats, achievements, follower counts, or identity as your own, and do "
            "NOT weave their facts together with your own traits/facts in the same sentence, even if the "
            "user's message invites a comparison. If a comparison is genuinely asked for, keep the two "
            "clearly separated (e.g. \"they have X, while I...\") rather than merging them into one voice.\n\n"
            f"Context records for '{persona}':\n{context_corpus}\n\n"
            f"Relationship facts:\n{graph_corpus}"
        )

        messages_payload = [{"role": "system", "content": system_prompt}]
        if history:
            try:
                parsed_history = json.loads(history)
                for turn in parsed_history:
                    role = turn.get("role")
                    content = turn.get("content")
                    if role in ["user", "assistant"] and content:
                        messages_payload.append({"role": role, "content": content})
            except Exception:
                pass

        if image_base64_data:
            user_content_payload = [
                {"type": "text", "text": full_message_query},
                {"type": "image_url", "image_url": {"url": image_base64_data}}
            ]
        else:
            user_content_payload = full_message_query

        messages_payload.append({"role": "user", "content": user_content_payload})

        try:
            response = deepseek_client.chat.completions.create(
                model="openrouter/free",
                messages=messages_payload,
                stream=True,
                timeout=90
            )
        except RateLimitError as e:
            print(f"OpenRouter rate limit hit: {e}")
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI model has hit OpenRouter's free-tier daily limit (50 requests/day "
                    "with no credits added, 1000/day once you add $10+ credits). This resets "
                    "daily -- it isn't a bug in your persona. To fix it for deployment, add "
                    "credits to your OpenRouter account or point the model at a paid/higher-limit "
                    "model instead of 'openrouter/free'."
                )
            )
        except (APIConnectionError, APITimeoutError) as e:
            print(f"OpenRouter connection error: {e}")
            raise HTTPException(
                status_code=502,
                detail="Couldn't reach the AI model provider right now. Please try again in a moment."
            )
        except APIStatusError as e:
            print(f"OpenRouter API error ({e.status_code}): {e}")
            raise HTTPException(
                status_code=e.status_code,
                detail=f"The AI model provider returned an error: {getattr(e, 'message', str(e))}"
            )

        def generate_stream():
            try:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        clean_chunk = (content_chunk
                                       .replace('*', '')
                                       .replace('<b>', '')
                                       .replace('</b>', '')
                                       .replace('<strong>', '')
                                       .replace('</strong>', ''))
                        yield clean_chunk
            except Exception as stream_err:
                print(f"Streaming error: {stream_err}")

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong generating a response. Please try again.")

@app.get("/")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, loop="proactor")