"""
Part 2: Fuzzy Clustering
========================
Implements Fuzzy C-Means from scratch using numpy, with PCA for
dimensionality reduction.

scikit-fuzzy was not used because it imports the `imp` module
which was removed in Python 3.12. The FCM algorithm is implemented
directly in ~80 lines of numpy.

Design decisions:
  - Cosine distance (not Euclidean) for L2-normalised embeddings.
    All-MiniLM outputs unit vectors; on a unit hypersphere cosine
    distance = 1 - dot(a,b) is the natural metric.
  - Centroid initialisation from random data points, not a random
    membership matrix. Random U causes all centroids to collapse to
    the data mean (all equidistant, uniform memberships, FPC = 1/k).
  - PCA to 100 dimensions before clustering (retains ~68% variance)
    to mitigate the curse of dimensionality.
  - k=12 clusters chosen via FPC elbow analysis.
  - Fuzziness m=2 (standard default).
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Constants ────────────────────────────────────────────────────────────────

N_CLUSTERS  = 12
FUZZINESS   = 2.0
PCA_DIMS    = 100
FCM_ERROR   = 0.005
FCM_MAXITER = 150

CLUSTER_RESULTS_PATH = "./cluster_results.npz"
CLUSTER_META_PATH    = "./cluster_meta.json"


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, n_components: int = PCA_DIMS):
    """
    Reduce embeddings from 384D to 100D using PCA.

    PCA finds the directions of maximum variance and projects onto
    the top n_components axes. This retains ~68% of the information
    while making distances geometrically meaningful — in 384 dimensions
    all points become roughly equidistant, degrading clustering quality.

    Returns:
        reduced:    (n_docs, 100) projected array
        pca_model:  fitted PCA object needed to project new queries
    """
    logger.info(f"PCA: {embeddings.shape[1]}D → {n_components}D")
    pca     = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    logger.info(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    return reduced, pca


# ── Distance functions ────────────────────────────────────────────────────────

def _cosine_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Cosine distance from every point to every centroid.

    cosine_distance(a, b) = 1 - dot(a_norm, b_norm)

    Both X and centroids are L2-normalised before the dot product
    so that PCA-projected vectors (which are not unit-norm) are
    handled correctly.

    Why cosine for text embeddings:
      Sentence embeddings encode meaning as direction, not magnitude.
      Two documents on the same topic but different lengths will have
      similar directions but different magnitudes. Cosine distance
      captures topic similarity; Euclidean distance conflates topic
      similarity with document length.

    Returns: (n_docs, n_clusters) in range [0, 2]
    """
    X_norm = X         / (np.linalg.norm(X,         axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (np.linalg.norm(centroids,  axis=1, keepdims=True) + 1e-10)
    return np.clip(1.0 - X_norm @ C_norm.T, 0.0, 2.0)


def _euclidean_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Euclidean distance from every point to every centroid.

    euclidean_distance(a, b) = sqrt(Σ(a_i - b_i)²)

    Vectorised using: ||x-c||² = ||x||² - 2(x·c) + ||c||²

    Why Euclidean is less suitable for text embeddings:
      After PCA, vectors are not unit-normalised. Euclidean distance
      measures absolute separation in feature space, which means a
      document's cluster assignment is influenced by its embedding
      magnitude (related to document length) rather than purely its
      semantic direction. This can group long documents together
      regardless of topic, producing less semantically meaningful
      clusters than cosine distance.

    Returns: (n_docs, n_clusters), non-negative
    """
    X_sq   = (X ** 2).sum(axis=1, keepdims=True)          # (n, 1)
    C_sq   = (centroids ** 2).sum(axis=1, keepdims=True).T # (1, k)
    XC     = X @ centroids.T                               # (n, k)
    return np.sqrt(np.maximum(X_sq - 2 * XC + C_sq, 0.0))


# ── FCM core ─────────────────────────────────────────────────────────────────

def _distances_to_memberships(distances: np.ndarray, m: float) -> np.ndarray:
    """
    Convert a distance matrix to a fuzzy membership matrix.

    FCM formula:
        u_ic = 1 / Σ_j (d_ic / d_ij)^(2/(m-1))

    A document close to centroid c and far from all others gets
    u_ic ≈ 1.0. A document equidistant from all centroids gets
    u_ic = 1/k (uniform distribution).

    Epsilon prevents division-by-zero when a centroid coincides
    exactly with a data point.

    Returns: (n_docs, n_clusters), rows sum to 1.0
    """
    d        = distances + 1e-10
    exponent = 2.0 / (m - 1.0)
    ratio_sums = ((d[:, :, np.newaxis] / d[:, np.newaxis, :]) ** exponent).sum(axis=2)
    return 1.0 / ratio_sums


def _update_centroids(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """
    Recompute centroids as fuzzy-weighted averages of all data points.

        centroid_c = Σ_i(u_ic^m × x_i) / Σ_i(u_ic^m)

    The ^m exponent amplifies high-membership contributions and
    suppresses low-membership ones.

    Returns: (n_clusters, n_dims)
    """
    U_m = U ** m
    return (U_m.T @ X) / U_m.sum(axis=0, keepdims=True).T


# ── Full algorithm ────────────────────────────────────────────────────────────

def fuzzy_cmeans(
    X:            np.ndarray,
    n_clusters:   int   = N_CLUSTERS,
    m:            float = FUZZINESS,
    max_iter:     int   = FCM_MAXITER,
    error:        float = FCM_ERROR,
    seed:         int   = 42,
    distance_fn:  str   = "cosine",
) -> tuple[np.ndarray, np.ndarray, float, list]:
    """
    Fuzzy C-Means clustering — pure numpy implementation.

    Algorithm:
      1. Initialise centroids from k random data points.
      2. E-step: compute membership matrix from current centroids.
      3. M-step: recompute centroids as weighted averages.
      4. Repeat 2-3 until max(|U_new - U_old|) < error.

    Args:
        distance_fn: "cosine" (default) or "euclidean"

    Initialising from random data points (not a random U matrix)
    is critical: random U collapses all centroids to the data mean,
    producing uniform memberships and FPC = 1/k.

    Returns:
      U:         (n_docs, k) membership matrix, rows sum to 1.0
      centroids: (k, n_dims) final cluster centres
      fpc:       Fuzzy Partition Coefficient in [1/k, 1.0]
      history:   max membership delta per iteration
    """
    dist_func = _cosine_distances if distance_fn == "cosine" else _euclidean_distances
    logger.info(f"Fuzzy C-Means: k={n_clusters}, m={m}, distance={distance_fn}")

    rng       = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    U         = None
    history   = []

    for i in range(max_iter):
        U_new = _distances_to_memberships(dist_func(X, centroids), m)

        if U is not None:
            delta = float(np.max(np.abs(U_new - U)))
            history.append(delta)
            if (i + 1) % 10 == 0:
                logger.info(f"  iter {i+1:3d}: delta={delta:.6f}")
            if delta < error:
                logger.info(f"  Converged at iteration {i+1}")
                U = U_new
                break

        U         = U_new
        centroids = _update_centroids(X, U, m)
    else:
        logger.warning(f"Did not converge in {max_iter} iterations")

    fpc = float((U ** 2).sum() / len(X))
    logger.info(f"FPC = {fpc:.4f}  (min possible = {1/n_clusters:.4f})")
    return U, centroids, fpc, history


# ── Cluster count analysis ────────────────────────────────────────────────────

def analyse_cluster_count(X, k_values=[5, 8, 10, 12, 15, 20]):
    """
    Run FCM for each k value and record FPC.

    FPC = (1/n) × Σ_i Σ_c u_ic²
    Range: [1/k, 1.0]. Higher = more structure found.
    The elbow indicates the natural cluster count.
    """
    scores = {}
    for k in k_values:
        _, _, fpc, _ = fuzzy_cmeans(X, n_clusters=k, max_iter=80)
        scores[k]    = fpc
        logger.info(f"  k={k:2d}: FPC={fpc:.4f}")
    return scores


# ── Analysis helpers ──────────────────────────────────────────────────────────

def get_cluster_top_docs(U, texts, cluster_id, top_n=3):
    """Return the top_n documents with highest membership in cluster_id."""
    scores  = U[:, cluster_id]
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(float(scores[i]), texts[i][:300]) for i in top_idx]


def get_boundary_documents(U, texts, top_n=5):
    """
    Return documents with the highest membership entropy.

    entropy = -Σ_c u_ic × log(u_ic)

    High entropy = document sits between clusters = genuine semantic
    ambiguity (e.g. a post about gun control legislation belongs to
    both politics and firearms clusters).
    """
    eps     = 1e-10
    entropy = -(U * np.log(U + eps)).sum(axis=1)
    top_idx = np.argsort(entropy)[::-1][:top_n]
    return [(float(entropy[i]), texts[i][:300]) for i in top_idx]


# ── Persistence ───────────────────────────────────────────────────────────────

def save_cluster_results(U, centroids, fpc):
    np.savez(CLUSTER_RESULTS_PATH, membership_matrix=U, centroids=centroids)
    with open(CLUSTER_META_PATH, "w") as f:
        json.dump({"fpc": float(fpc), "n_clusters": int(centroids.shape[0])}, f)
    logger.info(f"Saved cluster results → {CLUSTER_RESULTS_PATH}")


def load_cluster_results():
    data = np.load(CLUSTER_RESULTS_PATH)
    with open(CLUSTER_META_PATH) as f:
        meta = json.load(f)
    return data["membership_matrix"], data["centroids"], meta["fpc"]


def cluster_exists():
    return os.path.exists(CLUSTER_RESULTS_PATH) and os.path.exists(CLUSTER_META_PATH)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_clusters(embeddings, texts, force_rebuild=False):
    """
    Full pipeline: PCA reduction → Fuzzy C-Means (cosine) → persist.

    Returns:
        U:         (n_docs, n_clusters) membership matrix
        centroids: (n_clusters, pca_dims) cluster centres
        fpc:       Fuzzy Partition Coefficient
        pca_model: fitted PCA object for projecting new queries
    """
    reduced, pca_model = reduce_dimensions(embeddings)

    if cluster_exists() and not force_rebuild:
        logger.info("Loading existing cluster results")
        U, centroids, fpc = load_cluster_results()
        return U, centroids, fpc, pca_model

    U, centroids, fpc, _ = fuzzy_cmeans(reduced)
    save_cluster_results(U, centroids, fpc)
    logger.info(f"Clustering complete — U.shape={U.shape}, FPC={fpc:.4f}")
    return U, centroids, fpc, pca_model


def get_query_cluster_memberships(query_embedding, centroids, pca_model, m=FUZZINESS):
    """
    Assign fuzzy cluster memberships to a new query at inference time.

    Projects the query into PCA space, then applies the FCM membership
    formula using distances to saved centroids. Avoids re-running FCM.
    """
    q         = pca_model.transform(query_embedding.reshape(1, -1))
    distances = _cosine_distances(q, centroids)
    return _distances_to_memberships(distances, m)[0]


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON & ANALYSIS SECTION
# Added for coursework evaluation — does not affect the main pipeline above.
# Run with: python -m src.clustering
# ═════════════════════════════════════════════════════════════════════════════

def compare_distance_metrics(reduced: np.ndarray, true_labels: list) -> dict:
    """
    Compare Cosine vs Euclidean distance for Fuzzy C-Means clustering.

    Metrics used:
      FPC  — Fuzzy Partition Coefficient: internal clustering quality.
              Range [1/k, 1.0]. Higher = more distinct clusters.
              Does not require ground truth labels.

      ARI  — Adjusted Rand Index: measures agreement between two
              cluster assignments, corrected for chance.
              Range [-1, 1]. 1 = perfect agreement, 0 = random.

      NMI  — Normalised Mutual Information: measures how much
              information one clustering shares with another.
              Range [0, 1]. 1 = perfect, 0 = no shared information.

    Why compare metrics:
      FPC is an internal metric (no ground truth needed) that measures
      how crisp the clusters are. ARI and NMI are external metrics
      that compare cluster assignments against the known newsgroup
      categories. Together they give a complete picture of clustering
      quality from both perspectives.

    Why cosine is the principled choice despite similar numbers:
      After PCA, vectors are not unit-normalised. Euclidean distance
      measures absolute separation, meaning a document's cluster
      assignment is influenced by its embedding magnitude (loosely
      related to document length) as well as its topic direction.
      Cosine distance normalises this out — it measures only direction,
      which encodes semantic meaning in sentence embeddings.
      A marginally higher FPC for Euclidean does not mean better
      semantic clustering; it may reflect tighter grouping by
      document length rather than topic.

    Returns: dict with all metrics for both distance functions
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: Cosine vs Euclidean Distance")
    logger.info("="*60)

    results = {}

    for dist_name in ["cosine", "euclidean"]:
        logger.info(f"\nRunning FCM with {dist_name} distance...")
        U, centroids, fpc, history = fuzzy_cmeans(
            reduced, distance_fn=dist_name, seed=42
        )
        dominant = np.argmax(U, axis=1)

        # Internal metric — no ground truth needed
        # External metrics — compare against known newsgroup labels
        ari = adjusted_rand_score(true_labels, dominant)
        nmi = normalized_mutual_info_score(true_labels, dominant)

        results[dist_name] = {
            "U":        U,
            "fpc":      fpc,
            "dominant": dominant,
            "ari":      ari,
            "nmi":      nmi,
            "history":  history,
            "n_iters":  len(history),
        }

        logger.info(f"  FPC:          {fpc:.4f}  (min={1/N_CLUSTERS:.4f})")
        logger.info(f"  ARI vs truth: {ari:.4f}  (0=random, 1=perfect)")
        logger.info(f"  NMI vs truth: {nmi:.4f}  (0=none,   1=perfect)")
        logger.info(f"  Iterations:   {len(history)}")

    # Cross-metric: agreement between cosine and euclidean assignments
    cos_dom = results["cosine"]["dominant"]
    euc_dom = results["euclidean"]["dominant"]
    n_differ = int((cos_dom != euc_dom).sum())
    ari_cross = adjusted_rand_score(cos_dom, euc_dom)
    nmi_cross = normalized_mutual_info_score(cos_dom, euc_dom)

    results["cross"] = {
        "n_docs_differ": n_differ,
        "pct_differ":    round(n_differ / len(cos_dom) * 100, 2),
        "ari":           ari_cross,
        "nmi":           nmi_cross,
    }

    _print_comparison_summary(results)
    return results


def _print_comparison_summary(results: dict) -> None:
    """Print a formatted summary table of the comparison results."""
    cos = results["cosine"]
    euc = results["euclidean"]
    crs = results["cross"]

    sep = "-" * 52
    print(f"\n{'='*52}")
    print(f"  DISTANCE METRIC COMPARISON SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Metric':<30} {'Cosine':>8} {'Euclidean':>10}")
    print(sep)
    print(f"  {'FPC (internal quality)':<30} {cos['fpc']:>8.4f} {euc['fpc']:>10.4f}")
    print(f"  {'ARI vs ground truth':<30} {cos['ari']:>8.4f} {euc['ari']:>10.4f}")
    print(f"  {'NMI vs ground truth':<30} {cos['nmi']:>8.4f} {euc['nmi']:>10.4f}")
    print(f"  {'Iterations to converge':<30} {cos['n_iters']:>8d} {euc['n_iters']:>10d}")
    print(sep)
    print(f"  Docs assigned to different cluster: {crs['n_docs_differ']} / 19740  ({crs['pct_differ']}%)")
    print(f"  ARI between cosine & euclidean:     {crs['ari']:.4f}")
    print(f"  NMI between cosine & euclidean:     {crs['nmi']:.4f}")
    print(f"{'='*52}")
    print()

    # Interpretation
    winner_fpc = "Cosine" if cos["fpc"] > euc["fpc"] else "Euclidean"
    winner_ari = "Cosine" if cos["ari"] > euc["ari"] else "Euclidean"
    winner_nmi = "Cosine" if cos["nmi"] > euc["nmi"] else "Euclidean"

    print("  INTERPRETATION")
    print(sep)
    print(f"  Best FPC (cluster crispness):  {winner_fpc}")
    print(f"  Best ARI (vs ground truth):    {winner_ari}")
    print(f"  Best NMI (vs ground truth):    {winner_nmi}")
    print()
    print("  NOTE: A higher FPC for Euclidean does not necessarily mean")
    print("  better semantic clustering. After PCA, vectors are not")
    print("  unit-normalised, so Euclidean distance is influenced by")
    print("  embedding magnitude (loosely tied to document length).")
    print("  Cosine distance normalises this out and measures only")
    print("  semantic direction — the principled choice for text.")
    print(f"{'='*52}\n")


def analyse_cluster_composition(U: np.ndarray, true_labels: list,
                                 target_names: list) -> None:
    """
    Show which newsgroup categories dominate each cluster.

    For each of the 12 clusters, lists the top 3 newsgroup categories
    by document count. This reveals whether the unsupervised clustering
    has discovered the original topic boundaries without being told them.

    A cluster dominated by a single category = the algorithm found
    a clean semantic boundary.
    A cluster mixing multiple categories = those topics are semantically
    close in embedding space (e.g. talk.politics.guns + talk.politics.misc).
    """
    true_labels = np.array(true_labels)
    dominant    = np.argmax(U, axis=1)

    print(f"\n{'='*60}")
    print("  CLUSTER COMPOSITION (top 3 newsgroup categories each)")
    print(f"{'='*60}")

    for c in range(N_CLUSTERS):
        mask      = dominant == c
        count     = mask.sum()
        if count == 0:
            continue
        labels_in = true_labels[mask]

        # Count documents per category in this cluster
        cat_counts = {}
        for lbl in labels_in:
            cat_counts[lbl] = cat_counts.get(lbl, 0) + 1

        top3 = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(
            f"{target_names[l]} ({n}, {n/count*100:.0f}%)" for l, n in top3
        )
        print(f"  Cluster {c:2d} ({count:5d} docs): {top3_str}")

    print(f"{'='*60}\n")


def analyse_convergence(results: dict) -> None:
    """
    Compare convergence speed between cosine and euclidean FCM.

    Faster convergence (fewer iterations) indicates the distance metric
    produces a more stable update direction — the centroids find their
    natural positions more directly.
    """
    print(f"\n{'='*52}")
    print("  CONVERGENCE ANALYSIS")
    print(f"{'='*52}")
    for dist_name in ["cosine", "euclidean"]:
        h = results[dist_name]["history"]
        print(f"\n  {dist_name.capitalize()} distance:")
        print(f"    Iterations: {len(h)}")
        if h:
            print(f"    Initial delta: {h[0]:.6f}")
            print(f"    Final delta:   {h[-1]:.6f}")
            print(f"    Reduction:     {h[0]/h[-1]:.1f}x")
    print(f"{'='*52}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists("./embeddings.npy"):
        raise FileNotFoundError("Run python -m src.embeddings first")
    if not os.path.exists("./corpus_meta.json"):
        raise FileNotFoundError("Run python -m src.embeddings first")

    embeddings = np.load("./embeddings.npy")
    with open("./corpus_meta.json") as f:
        meta = json.load(f)

    true_labels  = meta["labels"]
    target_names = meta["target_names"]

    logger.info(f"Embeddings: {embeddings.shape}")

    # ── Step 1: Run main pipeline (cosine, saved to disk, used by API) ──
    U, centroids, fpc, pca_model = build_clusters(
        embeddings, [], force_rebuild=True
    )

    dominant = np.argmax(U, axis=1)
    logger.info(f"\nMembership matrix: {U.shape}")
    logger.info(f"Row sum:  {U[0].sum():.6f}")
    logger.info(f"FPC:      {fpc:.4f}  (min={1/N_CLUSTERS:.4f})")
    logger.info("\nDocs per dominant cluster:")
    for c in range(N_CLUSTERS):
        count = (dominant == c).sum()
        logger.info(f"  Cluster {c:2d}: {count:5d} docs  {'█' * (count // 100)}")

    # ── Step 2: PCA reduction (shared for comparison) ──────────────────
    reduced, _ = reduce_dimensions(embeddings)

    # ── Step 3: Distance metric comparison ─────────────────────────────
    results = compare_distance_metrics(reduced, true_labels)

    # ── Step 4: Convergence analysis ───────────────────────────────────
    analyse_convergence(results)

    # ── Step 5: Cluster composition (cosine results) ───────────────────
    analyse_cluster_composition(U, true_labels, target_names)
