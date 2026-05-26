import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Prevents GUI thread errors in web servers

PC_COLS = ["PC1", "PC2", "PC3"]


def load_local_pca_data(file_path: str) -> pd.DataFrame:
    """Loads the PCA output from the local backend directory and validates structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required input file: {file_path}")

    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in PC_COLS):
        raise ValueError(f"The input CSV must contain structural coordinates: {PC_COLS}")
    return df


# ---------------------------------------------------------------------------
# 1. Fit  (train + label in one shot — DBSCAN has no separate predict)
# ---------------------------------------------------------------------------

def fit_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    algorithm: str = "auto",
    scale_features: bool = True,
) -> pd.DataFrame:
    """
    Fits DBSCAN on the PC coordinate columns and returns a copy of the
    DataFrame with a `cluster_id` column appended. Does NOT mutate the input.

    cluster_id == -1 means the row is a noise point (not part of any cluster).

    Args:
        df:             DataFrame containing PC1, PC2, PC3 columns.
        eps:            Maximum distance between two samples to be neighbours.
        min_samples:    Minimum samples in a neighbourhood to form a core point.
        metric:         Distance metric ('euclidean', 'manhattan', etc.).
        algorithm:      Nearest-neighbour algorithm ('auto', 'ball_tree',
                        'kd_tree', 'brute').
        scale_features: StandardScale the PC coords before fitting.
                        Strongly recommended — DBSCAN is distance-sensitive.

    Returns:
        Copy of df with `cluster_id` column added.
    """
    pc_coords = df[PC_COLS].values.astype(float)
    if scale_features:
        pc_coords = StandardScaler().fit_transform(pc_coords)

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        n_jobs=-1,
    ).fit_predict(pc_coords)

    result_df = df.copy()
    result_df["cluster_id"] = labels
    return result_df


# ---------------------------------------------------------------------------
# 2. Extract metrics
# ---------------------------------------------------------------------------

def extract_cluster_metrics(
    df: pd.DataFrame,
    sample_size: int = 10000,
    random_state: int = 42,
) -> dict:
    """
    Computes DBSCAN cluster quality diagnostics from a fitted DataFrame.
    Noise points (cluster_id == -1) are excluded from the Silhouette score
    since the metric is undefined for unlabelled points.

    Requires `cluster_id` column to be present (call fit_dbscan first).

    Args:
        df:           DataFrame with PC1, PC2, PC3 and cluster_id columns.
        sample_size:  Max rows used for Silhouette score computation.
        random_state: Seed for subsample index selection.

    Returns:
        Dict with n_clusters_found, noise_count, noise_ratio,
        silhouette_score_sample, and cluster_distribution.
    """
    pc_coords = df[PC_COLS].values.astype(float)
    labels = df["cluster_id"].values

    unique_clusters = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(unique_clusters)

    noise_mask = labels == -1
    noise_count = int(noise_mask.sum())
    noise_ratio = round(float(noise_count / len(labels)), 4)

    sil_avg = None
    if n_clusters >= 2:
        coords_valid = pc_coords[~noise_mask]
        labels_valid = labels[~noise_mask]
        n = min(len(coords_valid), sample_size)
        idx = np.random.default_rng(random_state).choice(len(coords_valid), n, replace=False)
        sil_avg = float(silhouette_score(coords_valid[idx], labels_valid[idx]))

    cluster_distribution = {
        (f"Cluster_{cid}" if cid != -1 else "Noise"): int(count)
        for cid, count in df["cluster_id"].value_counts().items()
    }

    return {
        "n_clusters_found": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "silhouette_score_sample": sil_avg,
        "cluster_distribution": cluster_distribution,
    }


def generate_cluster_summary(
    edf: pd.DataFrame,
    cluster_column: str = "cluster_id",
    numeric_variance_threshold: float = 0.0,
    categorical_dominance_threshold: float = 0.6,
    max_unique_categories: int = 20,
) -> Dict[str, Any]:
    """
    Clean DBSCAN cluster summary optimised for frontend consumption.
    Noise points (cluster_id == -1) appear as a separate "Noise" entry.

    Outputs:
    - Only important numeric features (based on variance across clusters)
    - Only dominant categorical values (based on threshold)
    - Removes noisy/high-cardinality columns
    """
    df = pd.read_csv("output_source/02/cleaned_preprocessed.csv")
    df[cluster_column] = edf[cluster_column].values
    df["UNSPSC Code"] = df["UNSPSC Code"].astype("category")

    if cluster_column not in df.columns:
        raise ValueError(f"'{cluster_column}' not found in dataframe")

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != cluster_column]
    categorical_cols = [c for c in df.select_dtypes(exclude=np.number).columns if c != cluster_column]

    numeric_stats = df.groupby(cluster_column)[numeric_cols].mean()
    numeric_variance = numeric_stats.var().sort_values(ascending=False)
    important_numeric_cols = numeric_variance[numeric_variance > numeric_variance_threshold].index.tolist()

    result: Dict[str, Any] = {
        "cluster_column": cluster_column,
        "important_numeric_features": important_numeric_cols,
        "clusters": {},
    }

    for cluster_id in df[cluster_column].unique():
        cluster_df = df[df[cluster_column] == cluster_id]
        label = "Noise" if cluster_id == -1 else str(cluster_id)

        cluster_data: Dict[str, Any] = {
            "row_count": int(len(cluster_df)),
            "is_noise": bool(cluster_id == -1),
            "numeric_summary": {},
            "dominant_categories": {},
        }

        for col in important_numeric_cols:
            values = cluster_df[col].dropna()
            if values.empty:
                continue
            cluster_data["numeric_summary"][col] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        for col in categorical_cols:
            series = cluster_df[col].dropna()
            if series.empty or series.nunique() > max_unique_categories:
                continue
            vc = series.value_counts(normalize=True)
            top_ratio = float(vc.iloc[0])
            if top_ratio < categorical_dominance_threshold:
                continue
            cluster_data["dominant_categories"][col] = {
                "value": str(vc.index[0]),
                "percentage": round(top_ratio * 100, 2),
            }

        result["clusters"][label] = cluster_data

    return result


# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------

def save_predicted_csv(df: pd.DataFrame, output_path: str) -> str:
    """
    Saves the fitted DataFrame (including cluster_id) to a CSV file.
    Creates parent directories if they do not exist.

    Args:
        df:          DataFrame to persist (should include cluster_id).
        output_path: Full file path for the output CSV.

    Returns:
        The resolved output_path string.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path



def generate_3d_cluster_image(df: pd.DataFrame, output_dir: str = "static/plots") -> str:
    """Generates a 3D scatter plot and saves it as a PNG file.
    Noise points are rendered as grey x markers."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    clusters = sorted([c for c in df["cluster_id"].unique() if c != -1])
    colors = plt.cm.get_cmap("tab10", max(len(clusters), 1))

    for i, cluster in enumerate(clusters):
        sub = df[df["cluster_id"] == cluster]
        ax.scatter(sub["PC1"], sub["PC2"], sub["PC3"],
                   label=f"Cluster {cluster}", c=[colors(i)],
                   alpha=0.6, edgecolors="w", s=40)

    noise = df[df["cluster_id"] == -1]
    if not noise.empty:
        ax.scatter(noise["PC1"], noise["PC2"], noise["PC3"],
                   label="Noise", c=["#888888"],
                   alpha=0.3, edgecolors="none", s=15, marker="x")

    ax.set_title(f"DBSCAN Space Segmentation ({len(clusters)} clusters)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()

    filepath = os.path.join(output_dir, "dbscan_cluster.png")
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    return "dbscan_cluster.png"
    """Generates a 3D scatter plot and saves it as a PNG file.
    Noise points (cluster_id == -1) are rendered in grey separately."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    clusters = sorted([c for c in df["cluster_id"].unique() if c != -1])
    colors = plt.cm.get_cmap("tab10", max(len(clusters), 1))

    for i, cluster in enumerate(clusters):
        sub = df[df["cluster_id"] == cluster]
        ax.scatter(
            sub["PC1"], sub["PC2"], sub["PC3"],
            label=f"Cluster {cluster}",
            c=[colors(i)],
            alpha=0.6,
            edgecolors="w",
            s=40,
        )

    noise = df[df["cluster_id"] == -1]
    if not noise.empty:
        ax.scatter(
            noise["PC1"], noise["PC2"], noise["PC3"],
            label="Noise",
            c=["#888888"],
            alpha=0.3,
            edgecolors="none",
            s=15,
            marker="x",
        )

    ax.set_title(f"DBSCAN Space Segmentation ({len(clusters)} clusters)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()

    filepath = os.path.join(output_dir, "dbscan_cluster.png")
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    return "dbscan_cluster.png"