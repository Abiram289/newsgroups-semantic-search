# Newsgroups Semantic Search

Semantic search system over the 20 Newsgroups dataset with fuzzy clustering and a semantic cache layer. Built as a take-home assignment for an AI/ML Engineer role.

## What it does

- Embeds 19,740 newsgroup documents using `all-MiniLM-L6-v2`
- Stores vectors in Qdrant for fast approximate nearest-neighbour search
- Clusters documents using Fuzzy C-Means (implemented from scratch in numpy)
- Caches query results semantically — paraphrased queries return cached results without hitting the vector database
- Exposes everything via a FastAPI REST service

## Architecture

```
Query
  → Embed (all-MiniLM-L6-v2, 384 dims)
  → Fuzzy cluster memberships (FCM, k=12)
  → Semantic cache lookup (cosine similarity ≥ 0.85)
      → HIT:  return cached result
      → MISS: search Qdrant → store in cache → return result
```

## Project structure

```
newsgroups_semantic_search/
├── src/
│   ├── embeddings.py   # data loading, embedding, Qdrant storage
│   ├── clustering.py   # PCA + Fuzzy C-Means from scratch
│   ├── cache.py        # semantic cache implementation
│   └── api.py          # FastAPI endpoints
├── main.py             # app entry point
├── ui.html             # browser-based test UI
├── Dockerfile
└── requirements.txt
```

## Design decisions

**Embedding model: all-MiniLM-L6-v2**
384-dimensional output, runs fully locally, strong STS benchmark score (68.07), ~14k sentences/sec on CPU.

**Vector database: Qdrant**
Pure Python client — no C++ compilation required. Same HNSW approximate nearest-neighbour search as ChromaDB, works on all platforms without build tools.

**Fuzzy C-Means over K-Means**
Hard clustering assigns one label per document. A post about gun legislation genuinely belongs to both politics and firearms clusters. FCM gives each document a probability distribution over all clusters (rows sum to 1.0), which also enables smarter cache partitioning.

**FCM implemented from scratch**
scikit-fuzzy imports the `imp` module removed in Python 3.12. Rather than patching the library, FCM is implemented directly in ~80 lines of numpy using cosine distance on L2-normalised vectors.

**Semantic cache**
Entries are partitioned by cluster ID. Lookup only searches clusters where the query has membership > 0.1, keeping comparisons sub-linear as the cache grows. Similarity threshold (default 0.85) is tunable per-request.

## Setup

### Prerequisites

- Python 3.11+
- [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) extracted to `data/20_newsgroups/`

### Install dependencies

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Run setup scripts (one time)

```bash
# Step 1: embed documents and build vector store (~5 min)
python -m src.embeddings

# Step 2: run fuzzy clustering (~10 min)
python -m src.clustering
```

### Start the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test with the UI (recommended)

Open `ui.html` in your browser. The UI lets you run queries, adjust the similarity threshold, see fuzzy cluster membership bars per query, and watch cache hit/miss stats update in real time.

### Test with curl (alternative)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "nasa and space exploration", "top_k": 5}'
```

## Docker

### Option A — Pull from Docker Hub (fastest)

```bash
docker pull 7401323497/newsgroups-semantic-search:latest
docker run -p 8000:8000 \
  -v ${PWD}/qdrant_db:/app/qdrant_db \
  -v ${PWD}/embeddings.npy:/app/embeddings.npy \
  -v ${PWD}/corpus_meta.json:/app/corpus_meta.json \
  -v ${PWD}/cluster_results.npz:/app/cluster_results.npz \
  -v ${PWD}/cluster_meta.json:/app/cluster_meta.json \
  7401323497/newsgroups-semantic-search:latest
```

### Option B — Build from source

```bash
docker build -t newsgroups-search .
```

Run with data mounted:

```powershell
docker run -p 8000:8000 `
  -v ${PWD}/qdrant_db:/app/qdrant_db `
  -v ${PWD}/embeddings.npy:/app/embeddings.npy `
  -v ${PWD}/corpus_meta.json:/app/corpus_meta.json `
  -v ${PWD}/cluster_results.npz:/app/cluster_results.npz `
  -v ${PWD}/cluster_meta.json:/app/cluster_meta.json `
  newsgroups-search
```

## API endpoints

| Method | Endpoint       | Description                       |
| ------ | -------------- | --------------------------------- |
| POST   | `/query`       | Semantic search with cache        |
| GET    | `/cache/stats` | Hit rate, entry count, miss count |
| DELETE | `/cache`       | Flush cache and reset stats       |
| GET    | `/health`      | Server health check               |

### Example response

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

On a follow-up query with similar meaning:

```json
{
  "query": "tell me about the space program",
  "cache_hit": true,
  "matched_query": "nasa and space exploration",
  "similarity_score": 0.9134,
  "response_time_ms": 8.6
}
```
