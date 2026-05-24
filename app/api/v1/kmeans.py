from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# Ingest internal modular library tools
from services.lib_kmeans import (
    load_local_pca_data,
    fit_kmeans_and_extract_metrics,
    generate_3d_cluster_base64
)

router = APIRouter(
    prefix="/analytics",
    tags=["Internal Pipeline Segmentation Engine"]
)

# Enforce strict paths relative to workspace execution
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
    )
):
    try:
        # 1. Access local raw data layer directly from Step 03 folder output
        df = load_local_pca_data(str(INPUT_PATH))
        
        # 2. Fit model rules and harvest operational stats
        df_clustered, metrics = fit_kmeans_and_extract_metrics(df, k=k_selected)
        
        # 3. Secure output destination path rules and save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_clustered.to_csv(OUTPUT_FILE, index=False)
        
        # 4. Generate visual plot component represented safely in base64 string
        base64_img = generate_3d_cluster_base64(df_clustered)
        
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