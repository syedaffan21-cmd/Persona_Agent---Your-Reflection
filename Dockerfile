FROM python:3.11-slim

WORKDIR /app

# System deps needed to install/run Playwright's Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Downloads Chromium + its OS-level dependencies (needed for the Instagram/
# Twitter/LinkedIn scrapers, which drive a real headless browser)
RUN playwright install --with-deps chromium

COPY . .

# Render sets $PORT at runtime; default to 8000 for local `docker run` testing
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
