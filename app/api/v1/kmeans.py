from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# Ingest internal modular library tools
from services.lib_kmeans import (
    extract_cluster_metrics,
    generate_3d_cluster_image,
    load_local_pca_data,
    generate_3d_cluster_base64,
    predict_clusters,
    train_kmeans
)

router = APIRouter(
    prefix="/analytics",
    tags=["Internal Pipeline Segmentation Engine"]
)

# Enforce strict paths relative to workspace execution
SOURCE_PATH = "output_source/cluster_source/cluster_src.csv" # for source cluster
INPUT_PATH = "output_source/03/pca_processed.csv"
OUTPUT_DIR = Path("output_source/04/KMeans")
OUTPUT_FILE = f"{OUTPUT_DIR}/kmeans_clustered.csv"

class ClusterObservations(BaseModel):
    inertia: float
    silhouette_score_sample: float
    cluster_distribution: dict

class ClusteringResponse(BaseModel):
    status: str
    input_source_read: str
    output_destination_saved: str
    configurations: dict
    observations: ClusterObservations
    visualizations: dict

@router.post("/run-pipeline-clustering/", response_model=ClusteringResponse)
async def run_pipeline_clustering(
    k_selected: int = Query(
        default=3, 
        ge=2, 
        le=15, 
        description="The desired cluster size parameter (K) passed down into the K-Means matrix."
    ),
    init_strategy: str = Query(
        default="k-means++",
        description="The initialization strategy for K-Means clustering."
    ),
    n_init: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of initial runs for K-Means clustering."
    ),
    max_iterations: int = Query(
        default=300,
        ge=1,
        le=1000,
        description="Maximum number of iterations for K-Means clustering."
    )
):
    try:
        # 1. Load and Train the Med CSV From 2020 to 2025
        pd_source = load_local_pca_data(SOURCE_PATH)
        kmeans = train_kmeans(
            pd_source, 
            k=k_selected, 
            init_strategy=init_strategy, 
            n_init=n_init, 
            max_iterations=max_iterations)

        # 1. Access local raw data layer directly from Step 03 folder output which is the pca input
        df = load_local_pca_data(str(INPUT_PATH))

        # 2. Fit model rules and harvest operational stats
        df_clustered = predict_clusters(df, kmeans)
        
        # 3. Secure output destination path rules and save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_clustered.to_csv(OUTPUT_FILE, index=False)
        
        # 4 Extract the cluster metrics from df clustered
        metrics = extract_cluster_metrics(df_clustered, kmeans)
        
        # 5. Generate visual plot component represented safely in base64 string
        base64_img = generate_3d_cluster_base64(df_clustered)
        generate_3d_cluster_image(df_clustered) # for saving it to the backend
        
        # 5. Populate and serialize standard JSON body blueprint
        return ClusteringResponse(
            status="Success",
            input_source_read=str(INPUT_PATH),
            output_destination_saved=str(OUTPUT_FILE),
            configurations={
                "algorithm_fitted": "K-Means++",
                "k_clusters_assigned": k_selected,
                "total_records_evaluated": len(df_clustered)
            },
            observations=ClusterObservations(
                inertia=metrics["inertia"],
                silhouette_score_sample=metrics["silhouette_score_sample"],
                cluster_distribution=metrics["cluster_distribution"]
            ),
            visualizations={
                "cluster_3d_scatter_png_base64": base64_img
            }
        )

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline runtime processing error: {str(e)}")
    

@router.get("/get-kMeans-cluster-plot")
def get_kmeans_plot():
    return {
        "image_url": f"/static/plots/cluster.png"
    }