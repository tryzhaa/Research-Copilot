# Public demo image: read-only, rate-limited (DEMO_MODE=1), everything baked in.
#   docker build -t research-copilot .
#   docker run --rm -p 7860:7860 -e GROQ_API_KEY=... research-copilot   →  http://localhost:7860
# Needs ~1 GB of memory while searching.
FROM python:3.12-slim

# Run as an unprivileged user (uid 1000, which Hugging Face Spaces also expects);
# the app writes caches under data/ at runtime.
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app  # WORKDIR creates it root-owned; the app creates data/ and library.json here

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user

# Bake the slow one-time downloads into the image so a cold start doesn't redo them:
# the Papers with Code index (~5 min) and the bge-small embedding model (~130 MB).
# Only the modules they need are copied first, so ordinary code changes reuse this layer.
COPY --chown=user copilot/__init__.py copilot/errors.py copilot/pwc.py copilot/embeddings.py copilot/
RUN python -m copilot.pwc && \
    python -c "from copilot.embeddings import embed_texts; embed_texts(['warm up'])"

COPY --chown=user . .

ENV DEMO_MODE=1
EXPOSE 7860
# 7860 by default; hosts that assign a port (Render, Cloud Run) pass $PORT.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
