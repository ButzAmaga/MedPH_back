import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
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

def fit_kmeans_and_extract_metrics(df: pd.DataFrame, k: int, random_state: int = 42) -> tuple:
    """
    Fits K-Means and returns the updated dataframe along with structural diagnostic metrics.
    """
    pc_coords = df[['PC1', 'PC2', 'PC3']].values
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init=10)
    
    df['cluster_id'] = kmeans.fit_predict(pc_coords)
    
    # Calculate performance and cohesion diagnostics
    inertia = float(kmeans.inertia_)
    
    # Subsample for Silhouette score computation to keep the API responsive on larger data
    sample_size = min(len(pc_coords), 10000)
    idx = np.random.choice(len(pc_coords), sample_size, replace=False)
    sil_avg = float(silhouette_score(pc_coords[idx], df['cluster_id'].iloc[idx]))

    # Capture structural details per cluster
    cluster_counts = df['cluster_id'].value_counts().to_dict()
    cluster_distribution = {f"Cluster_{cid}": int(count) for cid, count in cluster_counts.items()}

    metrics = {
        "inertia": inertia,
        "silhouette_score_sample": sil_avg,
        "cluster_distribution": cluster_distribution
    }
    
    return df, metrics

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