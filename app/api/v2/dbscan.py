import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import FileResponse, StreamingResponse

from services.lib_SSE import run_with_heartbeats, sse, sse_progress
from services.lib_dbscan import (
    extract_cluster_metrics,
    generate_3d_cluster_image,
    generate_cluster_summary,
    fit_dbscan,
    load_local_pca_data,
    save_predicted_csv,
)

router = APIRouter(
    prefix="/dbscan/v2",
    tags=["DBSCAN Clustering"]
)

INPUT_PATH = "output_source/03/pca_processed.csv"
OUTPUT_DIR = Path("output_source/05/DBSCAN")
OUTPUT_FILE = str(OUTPUT_DIR / "dbscan_clustered.csv")


# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _dbscan_pipeline_stream(
    eps: float,
    min_samples: int,
    metric: str,
    algorithm: str,
    scale_features: bool,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every DBSCAN pipeline step.

    Note: DBSCAN has no separate train/predict — fit_dbscan() does both
    in one shot. The fitted DataFrame is then passed to metrics, summary,
    visualisation, and save steps.

    Event types emitted:
    - progress       — { step, total, message }
    - metrics        — cluster quality diagnostics
    - cluster_summary — per-cluster numeric + categorical breakdown
    - visualization  — { cluster_3d_scatter_png_base64 }
    - result         — full final payload
    - error          — { detail, status_code }
    """

    TOTAL_STEPS = 6
    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain_heartbeats():
        while not hb_queue.empty():
            yield await hb_queue.get()

    # ── Step 1 · Load inference data ─────────────────────────────────────────
    yield sse_progress(f"Loading PCA data from '{INPUT_PATH}'…", step=1, total_steps=TOTAL_STEPS)

    try:
        df = await run_with_heartbeats(hb_queue, load_local_pca_data, INPUT_PATH)
    except FileNotFoundError as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": str(e), "status_code": 404})
        return
    except ValueError as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": str(e), "status_code": 422})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress(
        f"Data loaded — {len(df):,} rows × {len(df.columns)} columns.",
        step=1,
        total_steps=TOTAL_STEPS,
    )

    # ── Step 2 · Fit DBSCAN (train + label in one shot) ──────────────────────
    yield sse_progress(
        f"Fitting DBSCAN — eps={eps}, min_samples={min_samples}, "
        f"metric='{metric}', algorithm='{algorithm}', scale={scale_features}…",
        step=2,
        total_steps=TOTAL_STEPS,
    )

    try:
        df_fitted = await run_with_heartbeats(
            hb_queue,
            fit_dbscan,
            df, eps, min_samples, metric, algorithm, scale_features,
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"DBSCAN fitting failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb

    n_clusters = int((df_fitted["cluster_id"] != -1).nunique() if -1 in df_fitted["cluster_id"].values
                     else df_fitted["cluster_id"].nunique())
    noise_count = int((df_fitted["cluster_id"] == -1).sum())

    yield sse_progress(
        f"DBSCAN complete — {n_clusters} clusters found, {noise_count:,} noise points.",
        step=2,
        total_steps=TOTAL_STEPS,
    )

    # ── Step 3 · Extract metrics ──────────────────────────────────────────────
    yield sse_progress("Computing cluster quality metrics…", step=3, total_steps=TOTAL_STEPS)

    try:
        cluster_metrics = await run_with_heartbeats(
            hb_queue, extract_cluster_metrics, df_fitted
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Metrics extraction failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse("metrics", cluster_metrics)

    sil = cluster_metrics.get("silhouette_score_sample")
    sil_str = f"{sil:.4f}" if sil is not None else "N/A (< 2 clusters)"
    yield sse_progress(
        f"Metrics ready — {cluster_metrics['n_clusters_found']} clusters, "
        f"noise ratio: {cluster_metrics['noise_ratio']:.2%}, silhouette: {sil_str}.",
        step=3,
        total_steps=TOTAL_STEPS,
    )

    # ── Step 4 · Save CSV + generate visualisations ───────────────────────────
    yield sse_progress(f"Saving clustered dataset to '{OUTPUT_FILE}'…", step=4, total_steps=TOTAL_STEPS)

    try:
        saved_path = await run_with_heartbeats(
            hb_queue, save_predicted_csv, df_fitted, OUTPUT_FILE
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"CSV save failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress(f"Dataset saved → {saved_path}", step=4, total_steps=TOTAL_STEPS)

    yield sse_progress("Generating 3D cluster scatter visualisation…", step=4, total_steps=TOTAL_STEPS)

    try:
        await run_with_heartbeats(hb_queue, generate_3d_cluster_image, df_fitted)
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Visualisation generation failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    
    yield sse_progress("3D scatter plot generated and saved.", step=4, total_steps=TOTAL_STEPS)

    # ── Step 5 · Generate cluster summary ─────────────────────────────────────
    yield sse_progress("Generating per-cluster summary…", step=5, total_steps=TOTAL_STEPS)

    try:
        cluster_summary = await run_with_heartbeats(
            hb_queue, generate_cluster_summary, df_fitted
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Cluster summary failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse("cluster_summary", cluster_summary)
    yield sse_progress("Cluster summary ready.", step=5, total_steps=TOTAL_STEPS)

    # ── Step 6 · Final result ─────────────────────────────────────────────────
    yield sse_progress("DBSCAN clustering pipeline finished successfully.", step=6, total_steps=TOTAL_STEPS)
    yield sse("result", {
        "status": "success",
        "input_source_read": INPUT_PATH,
        "output_destination_saved": saved_path,
        "configurations": {
            "algorithm_fitted": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "algorithm": algorithm,
            "scale_features": scale_features,
            "total_records_evaluated": len(df_fitted),
        },
        "observations": cluster_metrics,
        "visualizations": {
            "cluster_3d_scatter_png_base64": "none",
            "image_url": "/static/plots/dbscan_cluster.png",
        },
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/cluster")
async def run_dbscan_clustering(
    eps: float = Form(default=0.5, description="Max distance between two samples to be neighbours."),
    min_samples: int = Form(default=5, ge=1, description="Min samples to form a core point."),
    metric: str = Form(default="euclidean", description="Distance metric ('euclidean', 'manhattan', etc.)."),
    algorithm: str = Form(default="auto", description="Nearest-neighbour algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')."),
    scale_features: bool = Form(default=True, description="StandardScale PC coords before fitting. Strongly recommended."),
):
    """
    Fits DBSCAN on the PCA output CSV, extracts quality metrics, generates
    a cluster summary, saves the labelled dataset, and produces a 3D scatter plot.

    Unlike K-Means, DBSCAN fits and labels in a single pass — there is no
    separate training source or prediction step.

    Returns a **text/event-stream** (SSE) response.

    Event types emitted:
    - `progress`        — human-readable step message
    - `metrics`         — n_clusters, noise_ratio, silhouette score, distribution (Step 3)
    - `cluster_summary` — per-cluster numeric + categorical breakdown (Step 5)
    - `visualization`   — base64 PNG scatter plot (Step 4)
    - `result`          — full final payload (Step 6)
    - `error`           — error detail + HTTP status code
    """
    return StreamingResponse(
        _dbscan_pipeline_stream(eps, min_samples, metric, algorithm, scale_features),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/download/3d-plot")
def download_dbscan_plot():
    file_path = "static/plots/dbscan_cluster.png"

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename="dbscan_cluster_plot.png",
        headers={
            "Content-Disposition": "attachment; filename=dbscan_cluster_plot.png"
        }
    )