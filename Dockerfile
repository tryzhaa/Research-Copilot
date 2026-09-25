# Hosted demo image (Hugging Face Spaces builds this on their servers; no local Docker needed).
FROM python:3.12-slim

# Spaces runs containers as uid 1000; the app writes caches under data/ at runtime.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# Bake the slow one-time downloads into the image so a cold start doesn't redo them:
# the Papers with Code index (~5 min) and the bge-small embedding model (~130 MB).
RUN python -m copilot.pwc && \
    python -c "from copilot.embeddings import embed_texts; embed_texts(['warm up'])"

ENV DEMO_MODE=1
EXPOSE 7860
# Spaces expects 7860; other hosts (Render, Koyeb) pass $PORT.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
