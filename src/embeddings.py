"""
Part 1: Embedding & Vector Database Setup
==========================================
Loads raw files from the UCI 20 Newsgroups dataset, parses and
cleans each document, embeds with all-MiniLM-L6-v2, and stores
vectors in Qdrant for fast similarity search.

Raw file structure (USENET RFC 1036 format):
    From: user@domain.com        <- header block
    Newsgroups: alt.atheism      <- header block (reveals category)
    Subject: Re: Some topic      <- header block
                                 <- blank line = header/body boundary
    > quoted reply line          <- quoted text from previous poster
    Actual post content here.    <- body (what we keep)
    --                           <- signature marker
    John Smith                   <- signature (discard)
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR        = Path("./data/20_newsgroups")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "newsgroups"
BATCH_SIZE      = 64


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_newsgroup_file(filepath: Path) -> Optional[str]:
    """
    Extract body text from a raw USENET newsgroup file.

    Parsing steps:
      1. Decode bytes as latin-1 (1997 files use mixed encodings;
         latin-1 maps all 256 byte values so it never fails)
      2. Split on the first blank line to isolate the body from headers.
         Headers are discarded because the Newsgroups: field contains
         the category name — keeping it would make clustering trivial.
      3. Strip lines starting with '>' (quoted text from previous posters).
      4. Stop at '--' signature marker (everything after is name/tagline).
      5. Final text cleaning pass.
    """
    try:
        raw = filepath.read_bytes().decode("latin-1", errors="replace")
    except Exception as e:
        logger.debug(f"Could not read {filepath}: {e}")
        return None

    lines = raw.splitlines()

    # Find first blank line — header/body boundary
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break

    body_lines = lines[body_start:]

    # Remove quoted lines
    body_lines = [l for l in body_lines if not l.strip().startswith(">")]

    # Remove signature block
    clean_lines = []
    for line in body_lines:
        if line.strip() == "--":
            break
        clean_lines.append(line)

    return _clean_text(" ".join(clean_lines))


def _clean_text(text: str) -> Optional[str]:
    """
    Final cleaning pass on extracted body text.

    Removes email addresses, URLs, non-ASCII encoding artifacts,
    separator lines, and excess whitespace. Discards documents
    shorter than 50 characters — too short to produce a meaningful
    embedding for clustering.
    """
    if not text or not isinstance(text, str):
        return None

    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'^[\-=\*]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 50 else None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw_data(data_dir: Path = DATA_DIR):
    """
    Walk the 20_newsgroups/ folder and load all documents.

    Category folders are sorted alphabetically, giving a deterministic
    integer label mapping (alt.atheism=0, comp.graphics=1, ...,
    talk.religion.misc=19) consistent with sklearn's ordering.

    Returns:
        texts:        list of cleaned document strings
        labels:       list of integer category IDs (0–19)
        target_names: sorted list of category folder names
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir.absolute()}\n"
            f"Expected structure: data/20_newsgroups/alt.atheism/49960"
        )

    category_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    target_names  = [d.name for d in category_dirs]
    logger.info(f"Found {len(target_names)} categories")

    texts, labels = [], []
    skipped = 0

    for label_id, category_dir in enumerate(category_dirs):
        files = [f for f in category_dir.iterdir() if f.is_file()]
        for filepath in tqdm(files, desc=f"[{label_id:2d}] {category_dir.name}", leave=False):
            parsed = parse_newsgroup_file(filepath)
            if parsed is not None:
                texts.append(parsed)
                labels.append(label_id)
            else:
                skipped += 1

    logger.info(f"Loaded {len(texts)} documents, discarded {skipped}")
    return texts, labels, target_names


# ── Embedding ────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Load the all-MiniLM-L6-v2 sentence transformer model.

    Model choice rationale:
      - 384-dimensional output: good quality/speed balance
      - Runs fully locally: no API key or internet access required
      - ~90MB download, cached after first run
      - STS benchmark score: 68.07 (strong for its size class)
      - Processes ~14,000 sentences/sec on CPU
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Model ready — output dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_documents(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed all documents into a (n_docs, 384) numpy array.

    normalize_embeddings=True applies L2 normalisation so each vector
    has unit length. This allows cosine similarity to be computed as
    a dot product, and ensures scores fall in [0, 1].
    """
    logger.info(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"Embedding matrix: {embeddings.shape}")
    return embeddings


# ── Vector store ─────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Connect to the local persistent Qdrant instance.

    Qdrant was chosen over ChromaDB because its Python client
    requires no C++ compilation, making it installable on Windows
    without Microsoft Visual C++ Build Tools.
    """
    return QdrantClient(path=QDRANT_PATH)


def store_in_qdrant(texts, embeddings, labels, target_names, client):
    """
    Store document vectors in Qdrant with cosine distance metric.

    Each point contains:
      - vector:   384-dim embedding (used for similarity search)
      - payload:  original text, integer label, category name
                  (returned alongside search results)
    """
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    for start in tqdm(range(0, len(texts), 256), desc="Storing in Qdrant"):
        end = min(start + 256, len(texts))
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={
                        "text":     texts[i],
                        "label":    int(labels[i]),
                        "category": target_names[labels[i]],
                    },
                )
                for i in range(start, end)
            ],
        )

    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Stored {info.points_count} documents in Qdrant at {QDRANT_PATH}/")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_vector_store():
    """
    Full pipeline: parse raw files → embed → store in Qdrant.

    Run once with: python -m src.embeddings
    Results are persisted to disk so the API loads them at startup
    without re-embedding.
    """
    texts, labels, target_names = load_raw_data()
    model      = load_embedding_model()
    embeddings = embed_documents(texts, model)
    client     = get_qdrant_client()
    store_in_qdrant(texts, embeddings, labels, target_names, client)

    np.save("./embeddings.npy", embeddings)
    with open("./corpus_meta.json", "w") as f:
        json.dump({"labels": labels, "target_names": target_names, "n_docs": len(texts)}, f)

    logger.info("Done. Next: python -m src.clustering")
    return client, embeddings, texts, labels


def load_vector_store() -> QdrantClient:
    """Load the existing Qdrant collection. Requires build_vector_store() to have run."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Loaded Qdrant collection: {info.points_count} documents")
        return client
    except Exception:
        raise RuntimeError("Collection not found. Run: python -m src.embeddings")


if __name__ == "__main__":
    build_vector_store()