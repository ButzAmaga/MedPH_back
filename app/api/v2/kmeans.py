import json
import math
import asyncio
from datetime import date, datetime
import os
from pathlib import Path
from typing import AsyncGenerator, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, Query
from fastapi.responses import FileResponse, StreamingResponse

from services.lib_SSE import run_with_heartbeats, sse, sse_progress
from services.lib_kmeans import (
    extract_cluster_metrics,
    generate_3d_cluster_image,
    generate_3d_cluster_base64,
    generate_cluster_summary,
    load_local_pca_data,
    predict_clusters,
    save_predicted_csv,
    train_kmeans,
)

router = APIRouter(
    prefix="/kmeans/v2",
    tags=["k means clustering"]
)

SOURCE_PATH = "output_source/cluster_source/cluster_src.csv"
INPUT_PATH = "output_source/03/pca_processed.csv"
OUTPUT_DIR = Path("output_source/04/KMeans")
OUTPUT_FILE = str(OUTPUT_DIR / "kmeans_clustered.csv")


# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _kmeans_pipeline_stream(
    k_selected: int,
    init_strategy: str,
    n_init: int,
    max_iterations: int,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every K-Means pipeline step.

    Event types emitted:
    - progress      — { step, total, message }
    - metrics       — cluster quality diagnostics
    - visualization — { cluster_3d_scatter_png_base64 }
    - result        — full final payload
    - error         — { detail, status_code }
    """

    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain_heartbeats():
        while not hb_queue.empty():
            yield await hb_queue.get()

    # ── Step 1 · Load source training data ──────────────────────────────────
    yield sse_progress(f"Loading cluster source data from '{SOURCE_PATH}'…", step=1, total_steps=8)

    try:
        pd_source = await run_with_heartbeats(hb_queue, load_local_pca_data, SOURCE_PATH)
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
        f"Source data loaded — {len(pd_source):,} rows for model training.",
        step=1,
        total_steps=8
    )

    # ── Step 2 · Train K-Means ───────────────────────────────────────────────
    yield sse_progress(
        f"Training K-Means model — k={k_selected}, init='{init_strategy}', "
        f"n_init={n_init}, max_iter={max_iterations}…",
        step=2,
        total_steps=8
    )

    try:
        kmeans = await run_with_heartbeats(
            hb_queue,
            train_kmeans,
            pd_source, k_selected, 42, init_strategy, n_init, max_iterations,
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Model training failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress(
        f"K-Means model trained — inertia: {kmeans.inertia_:.2f}, "
        f"iterations: {kmeans.n_iter_}.",
        step=2,
        total_steps=8
    )

    # ── Step 3 · Load inference data ─────────────────────────────────────────
    yield sse_progress(f"Loading PCA inference data from '{INPUT_PATH}'…", step=3, total_steps=8)

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
        f"Inference data loaded — {len(df):,} rows to classify.",
        step=3,
        total_steps=8
    )

    # ── Step 4 · Predict clusters ────────────────────────────────────────────
    yield sse_progress("Assigning cluster labels to inference dataset…", step=4, total_steps=8)

    try:
        df_clustered = await run_with_heartbeats(hb_queue, predict_clusters, df, kmeans)
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Cluster prediction failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress(
        f"Cluster labels assigned — {df_clustered['cluster_id'].nunique()} distinct clusters found.",
        step=4,
        total_steps=8
    )

    # ── Step 5 · Extract metrics ─────────────────────────────────────────────
    yield sse_progress("Computing cluster quality metrics (inertia, silhouette score)…", step=5, total_steps=8)

    try:
        cluster_metrics = await run_with_heartbeats(
            hb_queue, extract_cluster_metrics, df_clustered, kmeans
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Metrics extraction failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse("metrics", cluster_metrics)
    yield sse_progress(
        f"Metrics ready — inertia: {cluster_metrics['inertia']:.2f}, "
        f"silhouette: {cluster_metrics['silhouette_score_sample']:.4f}.",
        step=5,
        total_steps=7
    )

    # ── Step 6 · Save CSV + generate visualizations ──────────────────────────
    yield sse_progress(f"Saving clustered dataset to '{OUTPUT_FILE}'…", step=6, total_steps=8)

    try:
        saved_path = await run_with_heartbeats(
            hb_queue, save_predicted_csv, df_clustered, OUTPUT_FILE
        )
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"CSV save failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress(f"Dataset saved → {saved_path}", step=6, total_steps=8)

    yield sse_progress("Generating 3D cluster scatter visualisation…", step=6, total_steps=8)

    try:
        base64_img = await run_with_heartbeats(
            hb_queue, generate_3d_cluster_base64, df_clustered
        )
        await run_with_heartbeats(hb_queue, generate_3d_cluster_image, df_clustered)
    except Exception as e:
        async for hb in drain_heartbeats(): yield hb
        yield sse("error", {"detail": f"Visualisation generation failed: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats(): yield hb
    yield sse_progress("3D scatter plot generated and saved.", step=6, total_steps=8)

    # Step 7 . Get Cluster Summary

    yield sse_progress("Generating cluster summary.", step=7, total_steps=8)
    cluster_summary = await run_with_heartbeats(hb_queue, generate_cluster_summary, df_clustered)

    async for hb in drain_heartbeats(): yield hb
    yield sse("cluster_summary", cluster_summary)

    # ── Step 7 · Final result ────────────────────────────────────────────────
    yield sse_progress("K-Means clustering pipeline finished successfully.", step=8, total_steps=8)
    yield sse("result", {
        "status": "success",
        "input_source_read": INPUT_PATH,
        "output_destination_saved": saved_path,
        "configurations": {
            "algorithm_fitted": init_strategy,
            "k_clusters_assigned": k_selected,
            "n_init": n_init,
            "max_iterations": max_iterations,
            "total_records_evaluated": len(df_clustered),
        },
        "observations": cluster_metrics,
        "visualizations": {
            "cluster_3d_scatter_png_base64": "none",
            "image_url": "/static/plots/cluster.png",
        },
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/cluster")
async def run_pipeline_clustering(
    k_selected: int = Form(default=3, ge=2, le=15, description="Desired cluster size parameter (K)."),
    init_strategy: str = Form(default="k-means++", description="K-Means initialisation strategy."),
    n_init: int = Form(default=10, ge=1, le=100, description="Number of initialisation runs."),
    max_iterations: int = Form(default=300, ge=1, le=1000, description="Maximum K-Means iterations."),
):
    """
    Trains K-Means on the cluster source CSV, predicts labels on the PCA output,
    extracts quality metrics, saves results, and generates a 3D scatter plot.

    Returns a **text/event-stream** (SSE) response.

    Event types emitted:
    - `progress`      — human-readable step message
    - `metrics`       — inertia, silhouette score, cluster distribution (after Step 5)
    - `visualization` — base64 PNG scatter plot (after Step 6)
    - `result`        — full final payload (Step 7)
    - `error`         — error detail + HTTP status code
    """
    return StreamingResponse(
        _kmeans_pipeline_stream(k_selected, init_strategy, n_init, max_iterations),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/download/3d-plot")
def download_kmeans_plot():
    file_path = "static/plots/cluster.png"

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename="kmeans_cluster_plot.png",  # downloaded filename
        headers={
            "Content-Disposition": "attachment; filename=kmeans_cluster_plot.png"
        }
    )