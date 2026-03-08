"""
Part 4: FastAPI Service
========================
Exposes the semantic search system as a REST API with three endpoints:

    POST   /query        - semantic search with cache
    GET    /cache/stats  - cache performance metrics
    DELETE /cache        - flush cache and reset stats

All expensive objects (embedding model, Qdrant client, cluster data,
cache) are loaded once at startup via FastAPI's lifespan context
manager and stored in app_state for reuse across requests.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.embeddings import (
    load_embedding_model, load_vector_store,
    COLLECTION_NAME, EMBEDDING_MODEL
)
from src.clustering import (
    build_clusters, get_query_cluster_memberships,
    N_CLUSTERS, cluster_exists,
)
from src.cache import SemanticCache

logger    = logging.getLogger(__name__)
app_state = {}


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models and data once at server startup.

    Loading the embedding model on every request would add ~2 seconds
    of latency per query. Eager initialisation here keeps request
    handling fast.
    """
    logger.info("Starting up...")
    start = time.time()

    app_state["embedding_model"] = load_embedding_model()

    try:
        app_state["qdrant"] = load_vector_store()
    except RuntimeError as e:
        logger.error(str(e))
        raise

    if not cluster_exists():
        raise RuntimeError("Cluster data not found. Run: python -m src.clustering")

    embeddings = np.load("./embeddings.npy")
    with open("./corpus_meta.json") as f:
        json.load(f)  # validate file exists and is valid JSON

    membership_matrix, centroids, fpc, pca_model = build_clusters(embeddings, [])
    app_state["centroids"]  = centroids
    app_state["pca_model"]  = pca_model
    app_state["cache"]      = SemanticCache(similarity_threshold=0.85)

    logger.info(f"✅ Startup complete in {time.time() - start:.1f}s")
    yield
    logger.info("Shutting down...")
    app_state.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Newsgroups Semantic Search",
    description="Semantic search over 20 Newsgroups with fuzzy clustering and semantic cache.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:     str
    top_k:     int            = 5
    threshold: Optional[float] = None  # overrides default cache threshold


class QueryResponse(BaseModel):
    query:               str
    cache_hit:           bool
    matched_query:       Optional[str]
    similarity_score:    Optional[float]
    result:              str
    dominant_cluster:    int
    cluster_memberships: list[float]
    response_time_ms:    float


class CacheStatsResponse(BaseModel):
    total_entries:        int
    hit_count:            int
    miss_count:           int
    hit_rate:             float
    similarity_threshold: float
    n_clusters_used:      int


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a normalised 384-dim vector."""
    return app_state["embedding_model"].encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0]


def retrieve_documents(query_embedding: np.ndarray, top_k: int) -> str:
    """
    Search Qdrant for the top_k most similar documents.

    Returns a formatted string of results with category and similarity
    score for each document. In a production RAG system these documents
    would be passed to an LLM to generate a synthesised answer.
    """
    client: QdrantClient = app_state["qdrant"]
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )

    parts = []
    for i, hit in enumerate(hits):
        category = hit.payload.get("category", "unknown")
        score    = round(hit.score, 4)
        text     = hit.payload.get("text", "")[:500]
        parts.append(f"[Result {i+1}] Category: {category} (similarity: {score})\n{text}")

    return "\n\n---\n\n".join(parts)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Semantic search endpoint.

    Flow:
      1. Embed the query.
      2. Compute fuzzy cluster memberships.
      3. Check the semantic cache — return immediately on hit.
      4. On miss: retrieve from Qdrant, store in cache, return result.
    """
    start = time.time()
    cache: SemanticCache = app_state["cache"]

    query_embedding  = embed_query(request.query)
    query_membership = get_query_cluster_memberships(
        query_embedding, app_state["centroids"], app_state["pca_model"]
    )
    dominant_cluster = int(np.argmax(query_membership))

    original_threshold = cache.similarity_threshold
    if request.threshold is not None:
        cache.set_threshold(request.threshold)

    cache_result = cache.lookup(query_embedding, query_membership)

    if cache_result is not None:
        entry, score = cache_result
        if request.threshold is not None:
            cache.set_threshold(original_threshold)
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(score, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
            cluster_memberships=[round(float(m), 4) for m in query_membership],
            response_time_ms=round((time.time() - start) * 1000, 2),
        )

    result = retrieve_documents(query_embedding, request.top_k)
    cache.store(
        query=request.query,
        query_embedding=query_embedding,
        result=result,
        dominant_cluster=dominant_cluster,
        membership_vector=query_membership,
    )

    if request.threshold is not None:
        cache.set_threshold(original_threshold)

    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_memberships=[round(float(m), 4) for m in query_membership],
        response_time_ms=round((time.time() - start) * 1000, 2),
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Return current cache performance metrics."""
    return CacheStatsResponse(**app_state["cache"].get_stats())


@app.delete("/cache")
async def flush_cache():
    """Flush all cache entries and reset statistics."""
    app_state["cache"].flush()
    return {"message": "Cache flushed", "status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    client: QdrantClient = app_state.get("qdrant")
    cache:  SemanticCache = app_state.get("cache")
    info = client.get_collection(COLLECTION_NAME) if client else None
    return {
        "status":           "healthy",
        "vector_store_docs": info.points_count if info else 0,
        "cache_entries":    cache._total_entries() if cache else 0,
        "embedding_model":  EMBEDDING_MODEL,
        "n_clusters":       N_CLUSTERS,
    }