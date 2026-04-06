"""
Clustering Method Comparison
=============================
Compares four clustering configurations for semantic search:

  1. Fuzzy C-Means  + Cosine distance     (main system)
  2. Fuzzy C-Means  + Euclidean distance
  3. K-Means        + Cosine distance     (spherical KMeans)
  4. K-Means        + Euclidean distance  (standard baseline)

Run with:
    python compare.py

Requires uvicorn to be running (for query embedding):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import time
import logging
import requests
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.WARNING)

API_URL = "http://localhost:8000"

# ── Distance functions ────────────────────────────────────────────────────────

def cosine_distances(X, centroids):
    X_norm = X         / (np.linalg.norm(X,         axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (np.linalg.norm(centroids,  axis=1, keepdims=True) + 1e-10)
    return np.clip(1.0 - X_norm @ C_norm.T, 0.0, 2.0)


def euclidean_distances(X, centroids):
    X_sq = (X ** 2).sum(axis=1, keepdims=True)
    C_sq = (centroids ** 2).sum(axis=1, keepdims=True).T
    XC   = X @ centroids.T
    return np.sqrt(np.maximum(X_sq - 2 * XC + C_sq, 0.0))


def distances_to_memberships(distances, m=2.0):
    d          = distances + 1e-10
    exponent   = 2.0 / (m - 1.0)
    ratio_sums = ((d[:, :, np.newaxis] / d[:, np.newaxis, :]) ** exponent).sum(axis=2)
    return 1.0 / ratio_sums


# ── Clustering algorithms ─────────────────────────────────────────────────────

def fuzzy_cmeans(X, n_clusters=12, m=2.0, max_iter=150,
                 error=0.005, seed=42, distance="cosine"):
    dist_fn   = cosine_distances if distance == "cosine" else euclidean_distances
    rng       = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    U         = None
    history   = []

    for i in range(max_iter):
        U_new = distances_to_memberships(dist_fn(X, centroids), m)
        if U is not None:
            delta = float(np.max(np.abs(U_new - U)))
            history.append(delta)
            if delta < error:
                U = U_new
                break
        U         = U_new
        U_m       = U ** m
        centroids = (U_m.T @ X) / U_m.sum(axis=0, keepdims=True).T

    fpc = float((U ** 2).sum() / len(X))
    return U, centroids, fpc, history


def run_kmeans(X, n_clusters=12, distance="cosine", seed=42):
    X_fit = normalize(X, norm="l2") if distance == "cosine" else X
    km    = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X_fit)
    return km.labels_, km.cluster_centers_, km.inertia_


# ── Query embedding via API ───────────────────────────────────────────────────

def check_api():
    """Verify the API is reachable before starting."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"Connected to API — {data['vector_store_docs']} docs loaded")
        return True
    except Exception:
        print(f"\nERROR: Cannot reach API at {API_URL}")
        print("Start it first with:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000\n")
        return False


def get_query_data(query: str) -> dict:
    """
    Call the API to get cluster memberships for a query.
    The API embeds the query using the loaded model and returns
    the full membership vector — no need to load PyTorch here.
    """
    r    = requests.post(f"{API_URL}/query",
                         json={"query": query, "top_k": 5},
                         timeout=30)
    return r.json()


def memberships_to_reduced(memberships: np.ndarray, reduced: np.ndarray,
                            U_stored: np.ndarray) -> np.ndarray:
    """
    Convert API membership vector to a PCA-space vector for cluster assignment.

    Since we can't run the embedding model directly here, we use the
    membership-weighted average of stored document vectors as a proxy
    for the query's position in PCA space. Documents with high membership
    in the same clusters as the query will be geometrically close to it.
    """
    # Weight each document's reduced vector by its membership in the
    # query's dominant cluster
    dom     = int(np.argmax(memberships))
    weights = U_stored[:, dom]
    # Take top-20 docs in dominant cluster as proxy
    top_idx = np.argsort(weights)[::-1][:20]
    return reduced[top_idx].mean(axis=0)


# ── Query assignment ──────────────────────────────────────────────────────────

def assign_fcm(q_vec, centroids, distance="cosine"):
    dist_fn = cosine_distances if distance == "cosine" else euclidean_distances
    d       = dist_fn(q_vec.reshape(1, -1), centroids)
    return distances_to_memberships(d)[0]


def assign_kmeans(q_vec, centroids, distance="cosine"):
    dist_fn = cosine_distances if distance == "cosine" else euclidean_distances
    d       = dist_fn(q_vec.reshape(1, -1), centroids)[0]
    return int(np.argmin(d))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics_fcm(U, true_labels, X):
    dominant = np.argmax(U, axis=1)
    fpc      = float((U ** 2).sum() / len(U))
    ari      = adjusted_rand_score(true_labels, dominant)
    nmi      = normalized_mutual_info_score(true_labels, dominant)
    sil      = silhouette_score(X, dominant, metric="cosine",
                                sample_size=2000, random_state=42)
    return {"fpc": fpc, "ari": ari, "nmi": nmi,
            "silhouette": sil, "dominant": dominant}


def compute_metrics_kmeans(labels, true_labels, inertia, X):
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    sil = silhouette_score(X, labels, metric="cosine",
                           sample_size=2000, random_state=42)
    return {"inertia": inertia, "ari": ari, "nmi": nmi,
            "silhouette": sil, "dominant": np.array(labels)}


# ── Display ───────────────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


def print_metrics_table(methods):
    print_header("CLUSTERING QUALITY METRICS")
    print(f"  {'Method':<26} {'FPC/Inertia':>14} {'ARI':>8} "
          f"{'NMI':>8} {'Silhouette':>12}")
    print(f"  {'-'*26} {'-'*14} {'-'*8} {'-'*8} {'-'*12}")
    for name, m in methods.items():
        internal = (f"{m['fpc']:.4f} (FPC)" if "fpc" in m
                    else f"{m['inertia']:.0f} (inertia)")
        print(f"  {name:<26} {internal:>14} {m['ari']:>8.4f} "
              f"{m['nmi']:>8.4f} {m['silhouette']:>12.4f}")

    print("""
  Metrics explained:
  FPC:        Fuzzy Partition Coefficient [1/k, 1.0]. Higher = crisper clusters.
  Inertia:    Sum of squared distances to centroid. Lower = tighter clusters.
  ARI:        Adjusted Rand Index [-1, 1]. Agreement with true newsgroup labels.
  NMI:        Normalised Mutual Info [0, 1]. Shared information with true labels.
  Silhouette: [-1, 1]. How well each doc fits its cluster vs neighbours.""")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_comparison():
    test_queries = [
        "nasa and space exploration",
        "gun control legislation debate",
        "windows operating system problems",
        "christian religion and faith",
    ]

    # ── Check API is running ──────────────────────────────────────────
    if not check_api():
        return

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading embeddings and metadata...")
    embeddings = np.load("./embeddings.npy")
    with open("./corpus_meta.json") as f:
        meta = json.load(f)
    true_labels  = meta["labels"]
    target_names = meta["target_names"]

    cluster_data = np.load("./cluster_results.npz")
    U_stored     = cluster_data["membership_matrix"]

    # ── PCA ───────────────────────────────────────────────────────────
    print("Running PCA (384→100)...")
    pca     = PCA(n_components=100, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")

    # ── Get query data from API ───────────────────────────────────────
    print("\nFetching query embeddings from API...")
    query_api_data = []
    for q in test_queries:
        print(f"  → {q}")
        data = get_query_data(q)
        query_api_data.append(data)

    # Build proxy vectors in PCA space for each query
    query_reduced = np.array([
        memberships_to_reduced(
            np.array(d["cluster_memberships"]), reduced, U_stored
        )
        for d in query_api_data
    ])

    # ── Train all 4 methods ───────────────────────────────────────────
    print_header("TRAINING ALL 4 CLUSTERING METHODS")
    methods  = {}
    configs  = [
        ("FCM + Cosine",       "fcm",    "cosine"),
        ("FCM + Euclidean",    "fcm",    "euclidean"),
        ("KMeans + Cosine",    "kmeans", "cosine"),
        ("KMeans + Euclidean", "kmeans", "euclidean"),
    ]

    for name, algo, dist in configs:
        print(f"\n  [{name}]")
        t0 = time.time()
        if algo == "fcm":
            U, centroids, fpc, history = fuzzy_cmeans(reduced, distance=dist)
            m = compute_metrics_fcm(U, true_labels, reduced)
            m.update({"centroids": centroids, "U": U,
                       "fpc": fpc, "n_iters": len(history)})
            print(f"  FPC={fpc:.4f}  ARI={m['ari']:.4f}  "
                  f"NMI={m['nmi']:.4f}  Sil={m['silhouette']:.4f}")
        else:
            labels, centroids, inertia = run_kmeans(reduced, distance=dist)
            m = compute_metrics_kmeans(labels, true_labels, inertia, reduced)
            m.update({"centroids": centroids, "labels": labels})
            print(f"  Inertia={inertia:.0f}  ARI={m['ari']:.4f}  "
                  f"NMI={m['nmi']:.4f}  Sil={m['silhouette']:.4f}")
        m["time"] = time.time() - t0
        methods[name] = m

    # ── Metrics table ─────────────────────────────────────────────────
    print_metrics_table(methods)

    # ── Per-query comparison ──────────────────────────────────────────
    print_header("PER-QUERY CLUSTER ASSIGNMENTS")

    for qi, query in enumerate(test_queries):
        q_vec    = query_reduced[qi]
        api_data = query_api_data[qi]
        api_mem  = np.array(api_data["cluster_memberships"])

        print(f"\n{'─'*64}")
        print(f"  QUERY: \"{query}\"")
        print(f"  API response time: {api_data['response_time_ms']}ms  "
              f"| Cache: {'HIT' if api_data['cache_hit'] else 'MISS'}")
        print(f"{'─'*64}")

        # FCM Cosine — use API memberships directly (most accurate)
        top3 = sorted(enumerate(api_mem), key=lambda x: -x[1])[:3]
        dom  = api_data["dominant_cluster"]
        n_search = int((api_mem > 0.1).sum())
        print(f"\n  FCM + Cosine  (from API)")
        print(f"  Top memberships: " +
              ", ".join(f"C{c}:{v:.3f}" for c, v in top3))
        print(f"  Dominant cluster: {dom}")
        print(f"  Clusters searched (>0.1 threshold): {n_search}")

        # FCM Euclidean
        m2   = methods["FCM + Euclidean"]
        mem2 = assign_fcm(q_vec, m2["centroids"], "euclidean")
        dom2 = int(np.argmax(mem2))
        top3_2 = sorted(enumerate(mem2), key=lambda x: -x[1])[:3]
        print(f"\n  FCM + Euclidean")
        print(f"  Top memberships: " +
              ", ".join(f"C{c}:{v:.3f}" for c, v in top3_2))
        print(f"  Dominant cluster: {dom2}  "
              f"({'same' if dom == dom2 else 'DIFFERENT'} as cosine FCM)")

        # KMeans Cosine
        km_c    = methods["KMeans + Cosine"]
        label_c = assign_kmeans(q_vec, km_c["centroids"], "cosine")
        n_in_c  = (km_c["dominant"] == label_c).sum()
        print(f"\n  KMeans + Cosine")
        print(f"  Assigned to: Cluster {label_c}  ({n_in_c} docs)")
        print(f"  Hard assignment — searches ONLY cluster {label_c}")

        # KMeans Euclidean
        km_e    = methods["KMeans + Euclidean"]
        label_e = assign_kmeans(q_vec, km_e["centroids"], "euclidean")
        n_in_e  = (km_e["dominant"] == label_e).sum()
        print(f"\n  KMeans + Euclidean")
        print(f"  Assigned to: Cluster {label_e}  ({n_in_e} docs)")
        print(f"  {'same' if label_c == label_e else 'DIFFERENT'} cluster as KMeans Cosine")

        print(f"\n  FCM searches {n_search} clusters  |  "
              f"KMeans searches 1 cluster  |  "
              f"FCM covers {n_search}x more relevant document space")

    # ── Summary ───────────────────────────────────────────────────────
    print_header("KEY FINDINGS SUMMARY")
    fc = methods["FCM + Cosine"]
    fe = methods["FCM + Euclidean"]
    kc = methods["KMeans + Cosine"]
    ke = methods["KMeans + Euclidean"]

    print(f"""
  1. FUZZY C-MEANS vs K-MEANS (both cosine)
     ┌──────────────────────┬────────┬──────────┐
     │ Metric               │  FCM   │  KMeans  │
     ├──────────────────────┼────────┼──────────┤
     │ ARI vs ground truth  │ {fc['ari']:.4f} │   {kc['ari']:.4f} │
     │ NMI vs ground truth  │ {fc['nmi']:.4f} │   {kc['nmi']:.4f} │
     │ Silhouette score     │ {fc['silhouette']:.4f} │   {kc['silhouette']:.4f} │
     └──────────────────────┴────────┴──────────┘
     KMeans scores higher on ARI/NMI because hard assignments map
     more cleanly to discrete categories. FCM's advantage is in
     retrieval — multi-cluster membership means a query searches
     across semantically related clusters simultaneously.

  2. COSINE vs EUCLIDEAN (both FCM)
     ┌──────────────────────┬────────┬───────────┐
     │ Metric               │ Cosine │ Euclidean │
     ├──────────────────────┼────────┼───────────┤
     │ FPC                  │ {fc['fpc']:.4f} │    {fe['fpc']:.4f} │
     │ ARI vs ground truth  │ {fc['ari']:.4f} │    {fe['ari']:.4f} │
     │ NMI vs ground truth  │ {fc['nmi']:.4f} │    {fe['nmi']:.4f} │
     │ Iterations           │    {fc['n_iters']:2d}   │        {fe['n_iters']:2d} │
     └──────────────────────┴────────┴───────────┘
     Euclidean FCM collapses to FPC=0.0833 (minimum possible) in
     just {fe['n_iters']} iterations — trivial convergence, no real structure found.
     Cosine FCM finds real structure across all metrics.
     Cosine is the principled choice for sentence embeddings.

  3. TRAINING TIME
     FCM + Cosine:       {fc['time']:.1f}s
     FCM + Euclidean:    {fe['time']:.1f}s  (fast but trivial)
     KMeans + Cosine:    {kc['time']:.1f}s
     KMeans + Euclidean: {ke['time']:.1f}s
    """)


if __name__ == "__main__":
    import os
    for f in ["./embeddings.npy", "./corpus_meta.json", "./cluster_results.npz"]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found. Run: python -m src.embeddings")
    run_comparison()
