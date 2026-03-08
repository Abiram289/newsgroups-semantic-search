"""
Part 3: Semantic Cache
======================
A from-scratch semantic cache that recognises paraphrased queries.

Traditional caches use exact key matching — "what is nasa" and
"tell me about nasa" are different keys, both miss. This cache
compares query embeddings using cosine similarity, so queries with
the same meaning return the cached result regardless of phrasing.

Data structure:
    A dict mapping cluster_id → list[CacheEntry].

    When a new query arrives, its fuzzy cluster memberships are
    computed. The lookup only searches clusters where membership
    exceeds a relevance threshold (0.1), reducing comparisons from
    O(n) to O(n/k) as the cache grows.

Similarity threshold:
    The single tunable parameter. Controls the minimum cosine
    similarity required for a cache hit.

    High threshold (e.g. 0.95): only near-identical phrasings hit.
    Low threshold  (e.g. 0.70): loosely related queries hit, risking
                                 incorrect results being returned.
    Default 0.85 balances paraphrase recognition with accuracy.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One cached query–result pair."""
    query:            str
    query_embedding:  np.ndarray   # stored to avoid re-embedding on lookup
    result:           str
    dominant_cluster: int
    membership_vector: np.ndarray
    timestamp:        float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-partitioned semantic cache using cosine similarity.

    Entries are stored in the lists for every cluster where the
    query has membership > 0.1. Lookup searches only those clusters,
    keeping comparisons sub-linear as the cache grows.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._store: dict[int, list[CacheEntry]] = defaultdict(list)
        self._hit_count  = 0
        self._miss_count = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding:  np.ndarray,
        query_membership: np.ndarray,
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Search for a semantically similar cached query.

        Only searches clusters where query_membership > 0.1,
        reducing the number of comparisons proportionally.

        Returns (entry, similarity_score) on hit, None on miss.
        """
        relevant_clusters = np.where(query_membership > 0.1)[0]
        if len(relevant_clusters) == 0:
            relevant_clusters = [int(np.argmax(query_membership))]

        best_entry = None
        best_score = -1.0

        for cluster_id in relevant_clusters:
            for entry in self._store.get(int(cluster_id), []):
                score = self._cosine_similarity(query_embedding, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self._hit_count += 1
            logger.debug(f"Cache HIT: score={best_score:.4f} query='{best_entry.query[:50]}'")
            return best_entry, best_score

        self._miss_count += 1
        logger.debug(f"Cache MISS: best score={best_score:.4f}")
        return None

    def store(
        self,
        query:            str,
        query_embedding:  np.ndarray,
        result:           str,
        dominant_cluster: int,
        membership_vector: np.ndarray,
    ) -> CacheEntry:
        """
        Add a query–result pair to the cache.

        The entry is inserted into every cluster where membership > 0.1
        so cross-cluster lookups find it correctly.
        """
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant_cluster,
            membership_vector=membership_vector.copy(),
        )

        relevant_clusters = np.where(membership_vector > 0.1)[0]
        if len(relevant_clusters) == 0:
            relevant_clusters = [dominant_cluster]

        for cluster_id in relevant_clusters:
            self._store[int(cluster_id)].append(entry)

        return entry

    # ── Similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.

        Cosine similarity measures the angle between vectors, ignoring
        magnitude. This is appropriate for text embeddings where two
        documents can discuss the same topic at different lengths.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ── Cache management ──────────────────────────────────────────────────────

    def flush(self) -> None:
        """Clear all entries and reset statistics."""
        self._store       = defaultdict(list)
        self._hit_count   = 0
        self._miss_count  = 0
        logger.info("Cache flushed")

    def get_stats(self) -> dict:
        """Return current cache performance metrics."""
        total   = self._hit_count + self._miss_count
        entries = self._total_entries()
        return {
            "total_entries":      entries,
            "hit_count":          self._hit_count,
            "miss_count":         self._miss_count,
            "hit_rate":           round(self._hit_count / total, 4) if total > 0 else 0.0,
            "similarity_threshold": self.similarity_threshold,
            "n_clusters_used":    len(self._store),
        }

    def set_threshold(self, threshold: float) -> None:
        assert 0.0 < threshold <= 1.0
        self.similarity_threshold = threshold

    def _total_entries(self) -> int:
        """Count unique entries across all cluster lists."""
        seen = set()
        for entries in self._store.values():
            for entry in entries:
                seen.add(id(entry))
        return len(seen)

    def __repr__(self) -> str:
        s = self.get_stats()
        return (f"SemanticCache(entries={s['total_entries']}, "
                f"hits={s['hit_count']}, misses={s['miss_count']}, "
                f"threshold={self.similarity_threshold})")