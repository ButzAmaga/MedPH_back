import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Prevents GUI thread errors in web servers


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


def fit_predict_clusters(
    df: pd.DataFrame,
    k: int,
    random_state: int = 42,
    init_strategy: str = "k-means++",
    n_init: int = 10,
    max_iterations: int = 300
) -> tuple[pd.DataFrame, KMeans]:
    """
    Fits a K-Means model on the PC coordinate columns and assigns cluster
    labels in one step. Returns both the annotated DataFrame and the trained
    model. Does NOT mutate the input DataFrame.

    Args:
        df:            DataFrame containing PC1, PC2, PC3 columns.
        k:             Number of clusters.
        random_state:  Seed for reproducibility.
        init_strategy: Initialization strategy (default: "k-means++").
        n_init:        Number of initializations to run.
        max_iterations: Maximum number of iterations per run.

    Returns:
        Tuple of (annotated DataFrame with `cluster_id` column, fitted KMeans).
    """
    pc_coords = df[PC_COLS].values
    kmeans = KMeans(
        n_clusters=k,
        init=init_strategy,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iterations
    )

    result_df = df.copy()
    result_df["cluster_id"] = kmeans.fit_predict(pc_coords)
    return result_df, kmeans

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

def generate_cluster_summary(
    edf: pd.DataFrame, # preprocessed pca with cluster
    cluster_column: str = "cluster_id",
    numeric_variance_threshold: float = 0.0,
    categorical_dominance_threshold: float = 0.6,
    max_unique_categories: int = 20
) -> Dict[str, Any]:
    """
    Clean cluster summary optimized for frontend consumption.

    Outputs:
    - Only important numeric features (based on variance across clusters)
    - Only dominant categorical values (based on threshold)
    - Removes noisy/high-cardinality columns
    """
    # get the clusters id from pca preprocessed data
    edf_cluster_id = edf[cluster_column]

    # load the cleaned original data
    df = pd.read_csv('output_source/02/cleaned_preprocessed.csv')

    # bind the edf_cluster_id to the original data
    df[cluster_column] = edf_cluster_id

    # make the UNSPSC Code a categorical
    df['UNSPSC Code'] = df['UNSPSC Code'].astype("category")

    if cluster_column not in df.columns:
        raise ValueError(f"'{cluster_column}' not found in dataframe")

    # =========================
    # Split column types
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != cluster_column]

    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != cluster_column]

    clusters = df[cluster_column].unique()

    # =========================
    # Compute numeric cluster stats
    # =========================
    numeric_stats = df.groupby(cluster_column)[numeric_cols].mean()

    # Determine "important" numeric features by variance across clusters
    numeric_variance = numeric_stats.var().sort_values(ascending=False)

    important_numeric_cols = numeric_variance[
        numeric_variance > numeric_variance_threshold
    ].index.tolist()

    # =========================
    # Build result
    # =========================
    result = {
        "cluster_column": cluster_column,
        "important_numeric_features": important_numeric_cols,
        "clusters": {}
    }

    for cluster_id in clusters:
        cluster_df = df[df[cluster_column] == cluster_id]

        cluster_data = {
            "row_count": int(len(cluster_df)),
            "numeric_summary": {},
            "dominant_categories": {}
        }

        # =========================
        # NUMERIC (filtered)
        # =========================
        for col in important_numeric_cols:

            values = cluster_df[col].dropna()

            if values.empty:
                continue

            cluster_data["numeric_summary"][col] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max())
            }

        # =========================
        # CATEGORICAL (filtered)
        # =========================
        for col in categorical_cols:

            series = cluster_df[col].dropna()

            if series.empty:
                continue

            # Skip high-cardinality columns
            if series.nunique() > max_unique_categories:
                continue

            value_counts = series.value_counts(normalize=True)

            top_value = value_counts.index[0]
            top_ratio = float(value_counts.iloc[0])

            # Only keep if dominant enough
            if top_ratio < categorical_dominance_threshold:
                continue

            cluster_data["dominant_categories"][col] = {
                "value": str(top_value),
                "percentage": round(top_ratio * 100, 2)
            }

        result["clusters"][str(cluster_id)] = cluster_data

    return result



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