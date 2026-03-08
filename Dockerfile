# ══════════════════════════════════════════════════════════════
# Dockerfile — Newsgroups Semantic Search
#
# Two-stage build:
#   Stage 1 (builder): install dependencies into a venv
#   Stage 2 (runtime): copy venv + code only — no build tools
#
# Data files (qdrant_db/, embeddings.npy, etc.) are NOT baked
# into the image. They are mounted at runtime via -v flags.
# This keeps the image small (~1GB) and builds fast.
#
# BUILD:
#   docker build -t newsgroups-search .
#
# RUN (mount your local data files):
#   docker run -p 8000:8000 \
#     -v %CD%\qdrant_db:/app/qdrant_db \
#     -v %CD%\embeddings.npy:/app/embeddings.npy \
#     -v %CD%\corpus_meta.json:/app/corpus_meta.json \
#     -v %CD%\cluster_results.npz:/app/cluster_results.npz \
#     -v %CD%\cluster_meta.json:/app/cluster_meta.json \
#     newsgroups-search
# ══════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# gcc is required to compile some Python packages from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first to leverage Docker layer caching.
# pip install only re-runs when requirements.txt changes,
# not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code only — data is mounted at runtime
COPY src/ ./src/
COPY main.py .

EXPOSE 8000

# Health check using Python's built-in urllib (curl not available
# in slim images)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Single worker — embedding model and Qdrant client are loaded
# once into shared memory. Multiple workers would each load their
# own copy, multiplying RAM usage unnecessarily.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]