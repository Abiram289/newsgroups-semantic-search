# Newsgroups Semantic Search

Semantic search system over the 20 Newsgroups dataset (~19,740 documents) with fuzzy clustering and a from-scratch semantic cache. Built as a take-home assignment for an AI/ML Engineer role.

---

## What it does

Traditional keyword search fails when the user's words don't match the document's words. This system embeds both queries and documents into a shared vector space — so "nasa rocket launch" and "space shuttle mission" are close together even with no shared words.

On top of that, a semantic cache recognises when two differently-phrased queries mean the same thing and returns the cached result without touching the vector database.

**Full pipeline:**

1. Parse and clean 19,740 raw USENET newsgroup posts
2. Embed each document using `all-MiniLM-L6-v2` → 384-dimensional vectors
3. Store vectors in Qdrant for fast approximate nearest-neighbour search
4. Cluster documents using Fuzzy C-Means (k=12) to assign each document a probability distribution over 12 clusters
5. At query time: embed → get cluster memberships → check semantic cache → if miss, search Qdrant → cache result → return

---

## Architecture

```
Query string
    │
    ▼
Embedding model (all-MiniLM-L6-v2)
    │  384-dim unit vector
    ▼
FCM membership (PCA 384→100, then cosine distances to 12 centroids)
    │  [0.09, 0.13, 0.07, ...] — sums to 1.0
    ▼
Semantic cache lookup
    │  search only clusters where membership > 0.1
    │  cosine similarity vs cached query embeddings
    ├── HIT (similarity ≥ 0.85) → return cached result immediately
    │
    └── MISS → Qdrant ANN search (top-k by cosine similarity)
                    │
                    ▼
               store in cache (indexed by cluster memberships)
                    │
                    ▼
               return results
```

---

## Project structure

```
newsgroups_semantic_search/
├── src/
│   ├── __init__.py
│   ├── embeddings.py     # raw file parsing, embedding, Qdrant ingestion
│   ├── clustering.py     # PCA + Fuzzy C-Means (numpy, from scratch)
│   ├── cache.py          # semantic cache (dict + cosine similarity)
│   └── api.py            # FastAPI: /query, /cache/stats, /cache, /health
├── main.py               # uvicorn entry point
├── ui.html               # browser test UI (no framework, plain JS)
├── Dockerfile            # two-stage build, data mounted at runtime
├── .dockerignore
└── requirements.txt
```

---

## Design decisions

### Embedding model: all-MiniLM-L6-v2

- 384-dimensional output — good quality/speed balance
- Runs fully locally, no API key or internet access required after first download (~90MB)
- STS benchmark score: 68.07
- ~14,000 sentences/sec on CPU
- `normalize_embeddings=True` — all output vectors are unit length, so cosine similarity reduces to a dot product

### Vector database: Qdrant

Qdrant's Python client is pure Python — no C++ compilation required. The alternative (ChromaDB) requires compiling `chroma-hnswlib` from C++, which fails on Windows without Microsoft Visual C++ Build Tools. Both use HNSW for approximate nearest-neighbour search with cosine distance.

### Why fuzzy clustering over K-Means

K-Means assigns exactly one cluster label per document. But a post about gun control legislation genuinely belongs to both a politics cluster and a firearms cluster — hard assignment loses that information.

Fuzzy C-Means assigns every document a probability distribution over all k clusters. A document with memberships `[0.05, 0.60, 0.35, ...]` is saying "60% politics, 35% firearms." Rows always sum to 1.0.

This distribution is also used by the cache: a query is looked up in every cluster where membership exceeds 0.1, so cross-topic queries find relevant cached results correctly.

### FCM implemented from scratch

`scikit-fuzzy` imports the `imp` module which was removed in Python 3.12. Rather than patching a third-party library, Fuzzy C-Means is implemented directly in ~80 lines of numpy.

Key implementation details:

- **Cosine distance** (not Euclidean) — embeddings are unit vectors on a hypersphere; cosine distance = 1 - dot(a,b) is the natural metric
- **Centroid initialisation from random data points** — initialising from a random membership matrix causes all centroids to collapse to the data mean (all equidistant → uniform memberships → FPC = 1/k, useless). Picking actual data points spreads centroids across the real distribution from iteration 1
- **PCA 384→100 dims before clustering** — in 384 dimensions all points become roughly equidistant (curse of dimensionality). PCA projects onto axes of maximum variance, retaining ~68% of information while making distances meaningful
- **Convergence** at max membership change < 0.005, typically ~50 iterations
- **FPC = 0.123** — above the minimum of 0.083 (= 1/12), confirming real cluster structure was found

### Semantic cache

```python
# Data structure
store: dict[cluster_id, list[CacheEntry]]

# Lookup
relevant_clusters = clusters where query_membership > 0.1
best_match = max cosine_similarity(query_embedding, entry.query_embedding)
             over all entries in relevant_clusters
if best_match >= threshold: return cached result   # HIT
else: return None                                  # MISS

# Storage
store entry in ALL clusters where membership > 0.1
```

The cluster partitioning keeps lookups sub-linear as the cache grows — instead of comparing against every cached entry (O(n)), we only compare against entries in relevant clusters (O(n/k)).

The similarity threshold (default 0.85) is the single tunable parameter:

- Too high (0.99): only near-identical phrasings hit — low hit rate
- Too low (0.70): loosely related queries hit — wrong results returned
- 0.85: recognises paraphrases while avoiding false matches

### FastAPI

- Lifespan context manager loads the embedding model, Qdrant client, and cluster data once at startup — not on every request
- Pydantic models for request/response validation
- `threshold` is an optional per-request override so the UI's slider works without restarting the server

---

## Setup

### Prerequisites

- Python 3.11+
- [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) — extract to `data/20_newsgroups/` so the structure is `data/20_newsgroups/alt.atheism/49960`, etc.

### Install

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### One-time setup (run in order)

```bash
# Parses raw files, embeds 19,740 documents, stores in Qdrant (~5 min)
python -m src.embeddings

# PCA dimensionality reduction + Fuzzy C-Means clustering (~10 min)
python -m src.clustering
```

These generate: `qdrant_db/`, `embeddings.npy`, `corpus_meta.json`, `cluster_results.npz`, `cluster_meta.json`

### Start the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test

**Recommended:** open `ui.html` in your browser. Features:

- Query input with top_k and similarity threshold controls
- 8 preset queries designed to demonstrate cache hits
- Live cache stats (hit count, miss count, hit rate)
- Fuzzy cluster membership bar chart per query
- Query history with hit/miss badges

**Alternative:** Swagger UI at `http://localhost:8000/docs`

---

## Docker

### Option A — Pull from Docker Hub

```bash
docker pull 7401323497/newsgroups-semantic-search:latest
```

Run (mount your local data files — not baked into the image):

```bash
# Linux/macOS
docker run -p 8000:8000 \
  -v $(pwd)/qdrant_db:/app/qdrant_db \
  -v $(pwd)/embeddings.npy:/app/embeddings.npy \
  -v $(pwd)/corpus_meta.json:/app/corpus_meta.json \
  -v $(pwd)/cluster_results.npz:/app/cluster_results.npz \
  -v $(pwd)/cluster_meta.json:/app/cluster_meta.json \
  7401323497/newsgroups-semantic-search:latest
```

```powershell
# Windows PowerShell
docker run -p 8000:8000 `
  -v ${PWD}/qdrant_db:/app/qdrant_db `
  -v ${PWD}/embeddings.npy:/app/embeddings.npy `
  -v ${PWD}/corpus_meta.json:/app/corpus_meta.json `
  -v ${PWD}/cluster_results.npz:/app/cluster_results.npz `
  -v ${PWD}/cluster_meta.json:/app/cluster_meta.json `
  7401323497/newsgroups-semantic-search:latest
```

### Option B — Build from source

```bash
docker build -t newsgroups-search .
# then run with the same -v flags above, replacing the image name
```

**Why data is not baked into the image:** the vector store and embeddings total ~136MB. Including them would make every image rebuild re-upload 136MB unnecessarily. Volume mounting keeps the image small and separates application code from data — the correct pattern for production deployments where data lives in a managed database service.

---

## API reference

| Method | Endpoint       | Description                             |
| ------ | -------------- | --------------------------------------- |
| POST   | `/query`       | Semantic search with semantic cache     |
| GET    | `/cache/stats` | Cache performance metrics               |
| DELETE | `/cache`       | Flush all cache entries and reset stats |
| GET    | `/health`      | Server health and collection info       |

### POST /query

Request:

```json
{
  "query": "nasa and space exploration",
  "top_k": 5,
  "threshold": 0.85
}
```

Response (cache miss):

```json
{
  "query": "nasa and space exploration",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[Result 1] Category: sci.space (similarity: 0.6689)\n...",
  "dominant_cluster": 6,
  "cluster_memberships": [
    0.091, 0.095, 0.076, 0.074, 0.081, 0.076, 0.102, 0.079, 0.095, 0.076, 0.061,
    0.092
  ],
  "response_time_ms": 142.3
}
```

Response (cache hit on paraphrased query):

```json
{
  "query": "tell me about the space program",
  "cache_hit": true,
  "matched_query": "nasa and space exploration",
  "similarity_score": 0.9134,
  "result": "[Result 1] Category: sci.space (similarity: 0.6689)\n...",
  "dominant_cluster": 6,
  "cluster_memberships": [
    0.088, 0.091, 0.079, 0.071, 0.084, 0.079, 0.098, 0.082, 0.091, 0.079, 0.065,
    0.091
  ],
  "response_time_ms": 8.6
}
```

Cache hits are ~15x faster than misses (8ms vs 142ms) because they skip the Qdrant search entirely.

### GET /cache/stats

```json
{
  "total_entries": 4,
  "hit_count": 3,
  "miss_count": 4,
  "hit_rate": 0.4286,
  "similarity_threshold": 0.85,
  "n_clusters_used": 7
}
```
