import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Prevents GUI thread errors in web servers
import matplotlib.pyplot as plt

def load_local_pca_data(file_path: str) -> pd.DataFrame:
    """Loads the PCA output from the local backend directory and validates structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required input file: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = ['PC1', 'PC2', 'PC3']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"The input CSV must contain structural coordinates: {required_cols}")
    return df



PC_COLS = ["PC1", "PC2", "PC3"]


# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------

def train_kmeans(
    df: pd.DataFrame,
    k: int,
    random_state: int = 42,
    init_strategy: str = "k-means++",
    n_init:int = 10,
    max_iterations: int = 300
) -> KMeans:
    """
    Fits a K-Means model on the PC coordinate columns of the given DataFrame.
    Returns the trained KMeans instance — does NOT mutate the DataFrame.

    Args:
        df:           DataFrame containing PC1, PC2, PC3 columns.
        k:            Number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        Fitted KMeans model.
    """
    pc_coords = df[PC_COLS].values
    kmeans = KMeans(
        n_clusters=k,
        init=init_strategy,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iterations
    )
    kmeans.fit(pc_coords)
    return kmeans


# ---------------------------------------------------------------------------
# 2. Predict
# ---------------------------------------------------------------------------

def predict_clusters(
    df: pd.DataFrame,
    kmeans: KMeans,
) -> pd.DataFrame:
    """
    Uses a pre-trained KMeans model to assign cluster labels to each row.
    Returns a new DataFrame with a `cluster_id` column appended —
    does NOT mutate the input DataFrame.

    Args:
        df:     DataFrame containing PC1, PC2, PC3 columns.
        kmeans: A fitted KMeans instance (from train_kmeans).

    Returns:
        Copy of df with an additional `cluster_id` column.
    """
    pc_coords = df[PC_COLS].values
    result_df = df.copy()
    result_df["cluster_id"] = kmeans.predict(pc_coords)
    return result_df


# ---------------------------------------------------------------------------
# 3. Extract metrics
# ---------------------------------------------------------------------------

def extract_cluster_metrics(
    df: pd.DataFrame,
    kmeans: KMeans,
    sample_size: int = 10000,
    random_state: int = 42,
) -> dict:
    """
    Computes cluster quality diagnostics from a predicted DataFrame.
    Subsamples rows for the Silhouette score to stay responsive on large data.

    Requires `cluster_id` column to be present (i.e. call predict_clusters first).

    Args:
        df:           DataFrame with PC1, PC2, PC3 and cluster_id columns.
        kmeans:       The fitted KMeans instance used for prediction.
        sample_size:  Max rows used for the Silhouette score calculation.
        random_state: Seed for the subsample index selection.

    Returns:
        Dict with inertia, silhouette_score_sample, and cluster_distribution.
    """
    pc_coords = df[PC_COLS].values
    labels = df["cluster_id"].values

    inertia = float(kmeans.inertia_)

    n = min(len(pc_coords), sample_size)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(pc_coords), n, replace=False)
    sil_avg = float(silhouette_score(pc_coords[idx], labels[idx]))

    cluster_counts = df["cluster_id"].value_counts().to_dict()
    cluster_distribution = {
        f"Cluster_{cid}": int(count) for cid, count in cluster_counts.items()
    }

    return {
        "inertia": inertia,
        "silhouette_score_sample": sil_avg,
        "cluster_distribution": cluster_distribution,
    }


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------

def save_predicted_csv(
    df: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Saves the predicted DataFrame (including cluster_id) to a CSV file.
    Creates parent directories if they do not exist.

    Args:
        df:          DataFrame to persist (should include cluster_id).
        output_path: Full file path for the output CSV.

    Returns:
        The resolved output_path string.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_3d_cluster_base64(df: pd.DataFrame) -> str:
    """Generates a 3D scatter plot and returns it directly as a Base64 string."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    clusters = sorted(df['cluster_id'].unique())
    colors = plt.cm.get_cmap('tab10', len(clusters))
    
    for i, cluster in enumerate(clusters):
        sub_set = df[df['cluster_id'] == cluster]
        ax.scatter(
            sub_set['PC1'], 
            sub_set['PC2'], 
            sub_set['PC3'], 
            label=f'Cluster {cluster}',
            c=[colors(i)],
            alpha=0.6,
            edgecolors='w',
            s=40
        )
        
    ax.set_title(f'K-Means Space Segmentation (K={len(clusters)})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.tight_layout()
    
    # Save image to an in-memory byte buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    # Encode binary stream data to text-safe Base64 string
    base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_string}"

def generate_3d_cluster_image(df: pd.DataFrame, output_dir: str = "static/plots") -> str:
    """Generates a 3D scatter plot and saves it as a PNG file."""

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    clusters = sorted(df['cluster_id'].unique())
    colors = plt.cm.get_cmap('tab10', len(clusters))

    for i, cluster in enumerate(clusters):
        sub_set = df[df['cluster_id'] == cluster]
        ax.scatter(
            sub_set['PC1'],
            sub_set['PC2'],
            sub_set['PC3'],
            label=f'Cluster {cluster}',
            c=[colors(i)],
            alpha=0.6,
            edgecolors='w',
            s=40
        )

    ax.set_title(f'K-Means Space Segmentation (K={len(clusters)})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()

    plt.tight_layout()

    filename = f"cluster.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    return filename