# pca_routes.py
import json
import math
import asyncio
from datetime import date, datetime
import os
from typing import AsyncGenerator, Any, List

import cupy as np
import cudf as pd
from fastapi import APIRouter, Form, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from services.lib_SSE import run_with_heartbeats, sse, sse_progress
from services.lib_pca import run_pca_pipeline

router = APIRouter(
    prefix="/pca/v2",
    tags=["Dimensionality Reduction"]
)

PCA_OUTPUT_DIR = "output_source/03"
PCA_FILE_NAME = "pca_processed.csv"

# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _pca_pipeline_stream(
    input_csv_path: str,
    matrix_mode: str,
    max_components: int,
    selected_c: List[str],
) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every PCA pipeline step.

    Event types emitted:
    - progress    — { step, total, message }
    - diagnostics — PCA diagnostic payload (variance ratios, components, etc.)
    - result      — { output_data_path, input_source, diagnostics }
    - error       — { detail, status_code }
    """

    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain_heartbeats():
        while not hb_queue.empty():
            yield await hb_queue.get()

    # ── Step 1 · Validate input & read CSV ──────────────────────────────────
    yield sse_progress(f"Locating preprocessed dataset at '{input_csv_path}'…", step=1, total_steps=4)

    import os
    if not os.path.exists(input_csv_path):
        yield sse("error", {
            "detail": f"Target dataset file not found at: '{input_csv_path}'. Run preprocessing first.",
            "status_code": 404,
        })
        return

    yield sse_progress("Reading dataset into memory…", step=1, total_steps=4)

    try:
        df = await run_with_heartbeats(
            hb_queue,
            lambda: pd.read_csv(input_csv_path),
        )
    except Exception as read_err:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Failed to read dataset: {str(read_err)}", "status_code": 422})
        return

    async for hb in drain_heartbeats():
        yield hb

    if df.empty:
        yield sse("error", {"detail": "Dataset is empty. Aborting PCA pipeline.", "status_code": 400})
        return

    yield sse_progress(
        f"Dataset loaded — {len(df):,} rows × {len(df.columns)} columns.",
        step=1,
        total_steps=4
    )

    # ── Step 2 · Run PCA pipeline ────────────────────────────────────────────
    yield sse_progress(
        f"Running PCA pipeline — mode: '{matrix_mode}', max components: {max_components}…",
        step=2,
        total_steps=4
    )

    try:
        output_path, diagnostics = await run_with_heartbeats(
            hb_queue,
            lambda: run_pca_pipeline(
                input_csv_path=input_csv_path,
                matrix_mode=matrix_mode,
                max_pc=max_components,
                selected_columns=selected_c,
            ),
        )
    except FileNotFoundError as e:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": str(e), "status_code": 404})
        return
    except ValueError as e:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Mathematical or column configuration error: {str(e)}", "status_code": 422})
        return
    except Exception as e:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Unexpected error during PCA calculation: {str(e)}", "status_code": 500})
        return

    async for hb in drain_heartbeats():
        yield hb

    n_components = len(diagnostics.get("explained_variance_ratio", []))
    cumulative = diagnostics.get("cumulative_explained_variance", [])
    total_variance = round(cumulative[-1] * 100, 2) if cumulative else 0

    yield sse_progress(
        f"PCA complete — {n_components} components extracted, "
        f"{total_variance}% cumulative variance explained.",
        step=2,
        total_steps=4
    )

    # ── Step 3 · Stream diagnostics ──────────────────────────────────────────
    yield sse_progress("Emitting PCA diagnostics…", step=3, total_steps=4)
    yield sse("diagnostics", diagnostics)
    yield sse_progress(
        f"Diagnostics ready — features used: {diagnostics.get('features_used', [])}.",
        step=3,
        total_steps=4
    )

    # ── Step 4 · Final result ────────────────────────────────────────────────
    yield sse_progress(f"Output saved → {output_path}", step=4, total_steps=4)
    yield sse_progress("PCA pipeline finished successfully.", step=4, total_steps=4)
    yield sse("result", {
        "status": "success",
        "input_source": input_csv_path,
        "output_data_path": output_path,
        "diagnostics": diagnostics,
    })


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/compute_pca", status_code=status.HTTP_200_OK)
async def calculate_pca(
    matrix_mode: str = Form("base", description="Feature matrix subset selection: 'base' or 'scaled'"),
    max_components: int = Form(3, ge=1, le=10, description="Maximum number of Principal Components to extract"),
    custom_input_path: str = Form(
        "output_source/02/cleaned_preprocessed.csv",
        description="Local relative server path to the preprocessed dataset"
    ),
    # Keep Form here
    selected_cols: List[str] = Form(default=[], description="Selected Columns.")
):
    """
    Triggers the PCA reduction pipeline using a preprocessed file stored on disk.
    Merges original indices with principal component scores and saves to output_source/03/pca_processed.csv.

    Returns a **text/event-stream** (SSE) response.

    Event types emitted:
    - `progress`    — human-readable step message
    - `diagnostics` — PCA variance ratios, components matrix, scaler params (streamed after Step 2)
    - `result`      — final payload with output path + full diagnostics (Step 4)
    - `error`       — error detail + HTTP status code
    """

    print(selected_cols)
    return StreamingResponse(
        _pca_pipeline_stream(
            input_csv_path=custom_input_path,
            matrix_mode=matrix_mode,
            max_components=max_components,
            selected_c=selected_cols
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )



@router.get("/download", response_class=FileResponse)
async def download_cleaned_file():
    """
    Retrieve and download a previously processed and cleaned CSV file.
    """
    file_path = os.path.join(PCA_OUTPUT_DIR, PCA_FILE_NAME)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist or has expired.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=PCA_FILE_NAME,
    )