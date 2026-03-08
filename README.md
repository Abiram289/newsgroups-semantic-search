# Newsgroups Semantic Search

A lightweight semantic search system over the 20 Newsgroups dataset with fuzzy clustering and a from-scratch semantic cache.

## Architecture

```
Raw Text (20k docs)
      │
      ▼
┌─────────────────┐
│   Embeddings    │  all-MiniLM-L6-v2 → 384-dim vectors
│   (Part 1)      │  stored in ChromaDB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fuzzy Clustering│  PCA (384→50 dims) + Fuzzy C-Means
│   (Part 2)      │  k=12 clusters, m=2 fuzziness
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Cache  │  Cluster-partitioned, cosine similarity
│   (Part 3)      │  threshold=0.85, no external libraries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI API   │  POST /query, GET /cache/stats, DELETE /cache
│   (Part 4)      │
└─────────────────┘
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build vector store (downloads model, embeds 20k docs — ~5-10 min)
python -m src.embeddings

# 4. Run fuzzy clustering (~3-5 min)
python -m src.clustering

# 5. Start the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

```bash
# Search
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is nasa"}'

# Second query — semantic cache hit
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "tell me about nasa"}'

# Cache stats
curl http://localhost:8000/cache/stats

# Flush cache
curl -X DELETE http://localhost:8000/cache
```

## Key Design Decisions

### Embedding Model: `all-MiniLM-L6-v2`
- 384-dim output, runs fully locally
- Strong semantic similarity performance
- 6x faster than full-size models at this task

### Vector DB: ChromaDB
- In-process, no separate server
- Persists to disk — no re-embedding on restart
- Cosine similarity search built-in

### Fuzzy Clustering: Fuzzy C-Means (k=12, m=2)
- k=12 chosen via FPC elbow analysis (see notebooks/)
- m=2 is standard fuzziness — avoids near-hard clusters (m<1.5) and meaningless uniform distributions (m>3)
- PCA to 50 dims first — fixes curse of dimensionality

### Semantic Cache Threshold: 0.85
- Lower than 0.85: unrelated queries incorrectly hit cache
- Higher than 0.90: paraphrases miss cache, defeating its purpose
- Cluster-partitioned lookup: O(n/k) instead of O(n) comparisons

## Docker

```bash
docker build -t newsgroups-search .
docker run -p 8000:8000 newsgroups-search
```

## Interactive API Docs
Visit `http://localhost:8000/docs` after starting the server.
