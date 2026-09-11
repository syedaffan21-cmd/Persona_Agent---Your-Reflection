FROM python:3.11-slim

WORKDIR /app

# Makes Python print logs immediately instead of buffering them -- without
# this, real startup output can be invisible in Render's logs until the
# process exits, making a slow startup look like total silence.
ENV PYTHONUNBUFFERED=1

# System deps needed to install/run Playwright's Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Downloads Chromium + its OS-level dependencies (needed for the Instagram/
# Twitter/LinkedIn scrapers, which drive a real headless browser)
RUN playwright install --with-deps chromium

# Downloads and caches the embedding model INTO the image now, at build time,
# instead of leaving it to download on every fresh container start. Build time
# has no strict deadline; Render's runtime port-scan does -- this moves the
# slow part to where it's safe to be slow.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Render sets $PORT at runtime; default to 8000 for local `docker run` testing
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]