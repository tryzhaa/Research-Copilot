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
# Only pwc.py (standard library only) is copied first, and the model is fetched with fastembed
# directly, so code changes elsewhere reuse this layer instead of redownloading for ~10 min.
# The model name and cache dir must match copilot/embeddings.py (tests/test_embeddings.py checks).
COPY --chown=user copilot/__init__.py copilot/pwc.py copilot/
RUN python -m copilot.pwc && \
    python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='data/models').embed(['warm up']))"

COPY --chown=user . .

ENV DEMO_MODE=1
EXPOSE 7860
# 7860 by default; hosts that assign a port (Render, Cloud Run) pass $PORT.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
