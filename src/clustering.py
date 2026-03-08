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
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Constants ────────────────────────────────────────────────────────────────

N_CLUSTERS  = 12     # determined by FPC elbow analysis
FUZZINESS   = 2.0    # m parameter: 1=hard clustering, 2=standard fuzzy, >3=too uniform
PCA_DIMS    = 100    # captures ~68% of variance; reduces curse of dimensionality
FCM_ERROR   = 0.005  # convergence threshold on membership matrix change
FCM_MAXITER = 150

CLUSTER_RESULTS_PATH = "./cluster_results.npz"
CLUSTER_META_PATH    = "./cluster_meta.json"


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, n_components: int = PCA_DIMS):
    """
    Reduce embeddings from 384D to 100D using PCA.

    PCA finds the directions of maximum variance and projects onto
    the top n_components axes. This retains ~68% of the information
    while making Euclidean/cosine distances geometrically meaningful —
    in 384 dimensions all points become roughly equidistant, which
    degrades clustering quality.

    Returns:
        reduced:    (n_docs, 100) projected array
        pca_model:  fitted PCA object needed to project new queries
    """
    logger.info(f"PCA: {embeddings.shape[1]}D → {n_components}D")
    pca     = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    logger.info(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    return reduced, pca


# ── FCM core ─────────────────────────────────────────────────────────────────

def _cosine_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Cosine distance from every point to every centroid.

    cosine_distance(a, b) = 1 - dot(a_norm, b_norm)

    Both X and centroids are L2-normalised before the dot product
    so that PCA-projected vectors (which are not unit-norm) are
    handled correctly.

    Returns: (n_docs, n_clusters) in range [0, 2]
    """
    X_norm = X         / (np.linalg.norm(X,         axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (np.linalg.norm(centroids,  axis=1, keepdims=True) + 1e-10)
    return np.clip(1.0 - X_norm @ C_norm.T, 0.0, 2.0)


def _distances_to_memberships(distances: np.ndarray, m: float) -> np.ndarray:
    """
    Convert a distance matrix to a fuzzy membership matrix.

    FCM formula:
        u_ic = 1 / Σ_j (d_ic / d_ij)^(2/(m-1))

    A document close to centroid c and far from all others gets
    u_ic ≈ 1.0. A document equidistant from all centroids gets
    u_ic = 1/k (uniform distribution).

    Epsilon is added to distances to prevent division-by-zero when
    a centroid coincides exactly with a data point.

    Returns: (n_docs, n_clusters), rows sum to 1.0
    """
    d        = distances + 1e-10
    exponent = 2.0 / (m - 1.0)
    # ratio_sums[i, c] = Σ_j (d[i,c] / d[i,j])^exponent
    ratio_sums = ((d[:, :, np.newaxis] / d[:, np.newaxis, :]) ** exponent).sum(axis=2)
    return 1.0 / ratio_sums


def _update_centroids(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """
    Recompute centroids as fuzzy-weighted averages of all data points.

        centroid_c = Σ_i(u_ic^m × x_i) / Σ_i(u_ic^m)

    The ^m exponent amplifies the contribution of high-membership
    points and suppresses low-membership ones.

    Returns: (n_clusters, n_dims)
    """
    U_m = U ** m
    return (U_m.T @ X) / U_m.sum(axis=0, keepdims=True).T


# ── Full algorithm ────────────────────────────────────────────────────────────

def fuzzy_cmeans(
    X:        np.ndarray,
    n_clusters: int   = N_CLUSTERS,
    m:          float = FUZZINESS,
    max_iter:   int   = FCM_MAXITER,
    error:      float = FCM_ERROR,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, float, list]:
    """
    Fuzzy C-Means clustering — pure numpy implementation.

    Algorithm:
      1. Initialise centroids from k random data points.
      2. E-step: compute membership matrix from current centroids.
      3. M-step: recompute centroids as weighted averages.
      4. Repeat 2–3 until max(|U_new - U_old|) < error.

    Initialising from random data points (not a random U matrix)
    is critical: random U causes all centroids to collapse to the
    data mean, producing uniform memberships and FPC = 1/k.

    Returns:
      U:         (n_docs, k) membership matrix, rows sum to 1.0
      centroids: (k, n_dims) final cluster centres
      fpc:       Fuzzy Partition Coefficient ∈ [1/k, 1.0]
      history:   max membership delta per iteration
    """
    logger.info(f"Fuzzy C-Means: k={n_clusters}, m={m}, max_iter={max_iter}")

    rng       = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    U         = None
    history   = []

    for i in range(max_iter):
        U_new = _distances_to_memberships(_cosine_distances(X, centroids), m)

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
    Run FCM for each k value and record the Fuzzy Partition Coefficient.

    FPC = (1/n) × Σ_i Σ_c u_ic²

    Range: [1/k, 1.0]. Higher means more structure was found.
    The elbow — where FPC gain diminishes — indicates the natural
    number of clusters in the data.
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

    Maximum entropy indicates a document that sits between clusters —
    genuinely ambiguous, belonging meaningfully to multiple topics.
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
    Full pipeline: PCA reduction → Fuzzy C-Means → persist results.

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

    Projects the query into the same PCA space as the training data,
    then applies the FCM membership formula using distances to the
    saved cluster centroids. This avoids re-running full FCM.
    """
    q         = pca_model.transform(query_embedding.reshape(1, -1))
    distances = _cosine_distances(q, centroids)
    return _distances_to_memberships(distances, m)[0]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists("./embeddings.npy"):
        raise FileNotFoundError("Run python -m src.embeddings first")

    embeddings = np.load("./embeddings.npy")
    with open("./corpus_meta.json") as f:
        meta = json.load(f)

    logger.info(f"Embeddings: {embeddings.shape}")

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