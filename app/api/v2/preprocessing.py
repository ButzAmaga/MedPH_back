import os
import asyncio
from typing import AsyncGenerator
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from services.lib_SSE import run_with_heartbeats, sse, sse_progress
from services import lib_processing as preprocessor

router = APIRouter(
    prefix="/preprocessing/v2",
    tags=["Data Preprocessing & Features"])

# Define paths for local staging structures
CLEANED_INPUT_DIR = os.path.join("output_source", "01")
PREPROCESSED_OUTPUT_DIR = os.path.join("output_source", "02")
CLEANED_FILE_NAME = "cleaned.csv"
PREPROCESSED_FILE_NAME = "cleaned_preprocessed.csv"

# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _preprocess_pipeline_stream() -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every major preprocessing step.

    Event types emitted:
    - progress   — { step, total, message }
    - metrics    — preprocessing snapshot dict
    - result     — final payload with metrics + output path
    - error      — { detail, status_code }
    """

    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain_heartbeats():
        while not hb_queue.empty():
            yield await hb_queue.get()

    # ── Step 1 · Verify & load cleaned CSV ──────────────────────────────────
    yield sse_progress("Locating cleaned dataset on disk…", step=1, total_steps=5)

    input_path = os.path.join(CLEANED_INPUT_DIR, CLEANED_FILE_NAME)
    if not os.path.exists(input_path):
        yield sse("error", {
            "detail": f"Source file not found at '{input_path}'. Run the cleaning step first.",
            "status_code": 404,
        })
        return

    yield sse_progress("Reading cleaned CSV into memory…", step=1, total_steps=5)

    try:
        df_cleaned = await run_with_heartbeats(
            hb_queue,
            lambda: pd.read_csv(input_path, low_memory=False, encoding="utf-8"),
        )
    except Exception as read_err:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Failed to read data matrix: {str(read_err)}", "status_code": 422})
        return

    async for hb in drain_heartbeats():
        yield hb

    if df_cleaned.empty:
        yield sse("error", {"detail": "Cleaned dataset is empty. Aborting pipeline.", "status_code": 400})
        return

    yield sse_progress(
        f"Dataset loaded — {len(df_cleaned):,} rows × {len(df_cleaned.columns)} columns.",
        step=1,
        total_steps=5
    )

    # ── Step 2 · Execute preprocessing pipeline ──────────────────────────────
    yield sse_progress(
        "Running preprocessing pipeline (temporal parsing → Winsorization → categorical collapsing)…",
        step=2,
        total_steps=5
    )

    try:
        df_preprocessed = await run_with_heartbeats(
            hb_queue,
            preprocessor.execute_preprocessing_pipeline,
            df_cleaned,
        )
    except Exception as pipe_err:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Preprocessing pipeline failed: {str(pipe_err)}", "status_code": 500})
        return

    async for hb in drain_heartbeats():
        yield hb

    yield sse_progress(
        f"Pipeline complete — {len(df_preprocessed):,} rows · "
        f"{len(df_preprocessed.columns)} columns after feature engineering.",
        step=2,
        total_steps=5
    )

    # ── Step 3 · Generate metrics snapshot ───────────────────────────────────
    yield sse_progress("Generating preprocessing metrics snapshot…", step=3, total_steps=5)

    try:
        metrics_snapshot = await run_with_heartbeats(
            hb_queue,
            preprocessor.generate_preprocessing_snapshot,
            df_preprocessed,
        )
    except Exception as snap_err:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {"detail": f"Snapshot generation failed: {str(snap_err)}", "status_code": 500})
        return

    async for hb in drain_heartbeats():
        yield hb

    yield sse("metrics", metrics_snapshot)
    yield sse_progress(
        f"Metrics ready — {metrics_snapshot.get('total_records', 0):,} records, "
        f"{len(metrics_snapshot.get('unique_years_computed', []))} unique years, "
        f"{len(metrics_snapshot.get('unique_regions_collapsed', []))} regions.",
        step=3,
        total_steps=5
    )

    # ── Step 4 · Save preprocessed CSV ───────────────────────────────────────
    yield sse_progress("Saving preprocessed dataset to disk…", step=4)


    output_save_path = os.path.join(PREPROCESSED_OUTPUT_DIR, PREPROCESSED_FILE_NAME)

    try:
        os.makedirs(PREPROCESSED_OUTPUT_DIR, exist_ok=True)
        await run_with_heartbeats(
            hb_queue,
            lambda: df_preprocessed.to_csv(f"{output_save_path}", index=False),
        )
    except Exception as save_err:
        async for hb in drain_heartbeats():
            yield hb
        yield sse("error", {
            "detail": f"Preprocessing succeeded but file write failed: {str(save_err)}",
            "status_code": 500,
        })
        return

    async for hb in drain_heartbeats():
        yield hb

    yield sse_progress(f"Preprocessed file saved → {output_save_path}", step=4, total_steps=5)

    # ── Step 5 · Final result ────────────────────────────────────────────────
    yield sse_progress("Preprocessing pipeline finished successfully.", step=5, total_steps=5)
    yield sse("result", {
        "output_path": output_save_path,
        "metrics": metrics_snapshot,
    })


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/process")
async def process_file():
    """
    Locates the existing cleaned dataset, performs Winsorization,
    categorical dimension collapsing, and temporal feature extraction.

    Returns a **text/event-stream** (SSE) response.

    Event types emitted:
    - `progress` — human-readable step message
    - `metrics`  — preprocessing snapshot (streamed after Step 3)
    - `result`   — final payload with metrics + output path (Step 5)
    - `error`    — error detail + HTTP status code
    """
    return StreamingResponse(
        _preprocess_pipeline_stream(),
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
    file_path = os.path.join(PREPROCESSED_OUTPUT_DIR, PREPROCESSED_FILE_NAME)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist or has expired.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=PREPROCESSED_FILE_NAME,
    )