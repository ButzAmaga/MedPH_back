import os
import json
import asyncio
from typing import AsyncGenerator, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from services.custom_toJson import SafeJsonEncoder
from services import lib_cleaning as cleaner

router = APIRouter(
    prefix="/cleaning/v2",
    tags=["Procurement Processing"])

# Define the base directory where you want to cache or save the cleaned datasets
CLEANED_DIR = os.path.join("output_source", "01")
FILE_NAME = "cleaned.csv"

# Interval (seconds) between keep-alive heartbeats during long blocking operations
HEARTBEAT_INTERVAL = 10

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Formats a single Server-Sent Events frame, safely serialising all pandas/numpy types."""
    return f"event: {event}\ndata: {json.dumps(data, cls=SafeJsonEncoder)}\n\n"
 

def _sse_progress(message: str, step: int, total_steps: int = 6) -> str:
    return _sse("progress", {"step": step, "total": total_steps, "message": message})


def _sse_heartbeat() -> str:
    """SSE comment line — invisible to EventSource listeners but resets proxy timeout."""
    return ": heartbeat\n\n"


# ---------------------------------------------------------------------------
# Concurrent heartbeat runner
# ---------------------------------------------------------------------------

async def _run_with_heartbeats(queue: asyncio.Queue, fn, *args) -> Any:
    """
    Runs a blocking function in a thread pool executor.
    While it is running, pushes a heartbeat into `queue` every
    HEARTBEAT_INTERVAL seconds so the stream generator can forward
    them to the client without blocking on the heavy work.
    """
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, fn, *args)

    while not task.done():
        try:
            # Wait up to HEARTBEAT_INTERVAL seconds; if task finishes sooner,
            # asyncio.wait raises TimeoutError which we just swallow.
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            # Task still running — push a heartbeat for the generator to yield
            await queue.put(_sse_heartbeat())

    return await task  # re-raises any exception from the thread


# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _clean_pipeline_stream(
    contents: bytes,
    filename: str,
    is_2022_format: bool,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every major pipeline step.
    For each blocking operation a concurrent heartbeat loop keeps the
    connection alive by draining an asyncio.Queue every HEARTBEAT_INTERVAL
    seconds while pandas does its work.
    """

    # Shared queue: heartbeat coroutine pushes frames here;
    # the generator drains and yields them between awaits.
    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain_heartbeats():
        """Yield every heartbeat that was queued during the last blocking call."""
        while not hb_queue.empty():
            yield await hb_queue.get()

    # ── Step 1 · Load raw DataFrame ─────────────────────────────────────────
    yield _sse_progress("Loading raw DataFrame from uploaded file…", step=1)

    try:
        df_before = await _run_with_heartbeats(
            hb_queue,
            cleaner.load_dataframe_from_stream,
            contents, filename, is_2022_format,
        )
    except ValueError as val_err:
        async for hb in drain_heartbeats():
            yield hb
        yield _sse("error", {"detail": str(val_err), "status_code": 400})
        return
    except Exception as err:
        async for hb in drain_heartbeats():
            yield hb
        yield _sse("error", {"detail": f"Parsing error: {str(err)}", "status_code": 422})
        return

    async for hb in drain_heartbeats():
        yield hb

    yield _sse_progress(
        f"Raw DataFrame loaded — {len(df_before):,} rows × {len(df_before.columns)} columns.",
        step=1,
    )

    # ── Step 2 · Snapshot before cleaning ───────────────────────────────────
    yield _sse_progress("Generating pre-cleaning metadata snapshot…", step=2)

    before_snapshot = await _run_with_heartbeats(
        hb_queue, cleaner.generate_metadata_snapshot, df_before
    )

    async for hb in drain_heartbeats():
        yield hb

    yield _sse("snapshot_before", before_snapshot)
    yield _sse_progress(
        f"Pre-cleaning snapshot ready — {before_snapshot['total_rows']:,} rows, "
        f"{before_snapshot['total_columns']} columns.",
        step=2,
    )

    # ── Step 3 · Execute cleaning pipeline ──────────────────────────────────
    yield _sse_progress(
        "Running cleaning pipeline (normalize → filter → deduplicate → impute)…",
        step=3,
    )

    df_after = await _run_with_heartbeats(
        hb_queue, cleaner.execute_cleaning_pipeline, df_before
    )

    async for hb in drain_heartbeats():
        yield hb

    yield _sse_progress(
        f"Cleaning complete — {len(df_after):,} rows retained after filtering & deduplication.",
        step=3,
    )

    # ── Step 4 · Snapshot after cleaning ────────────────────────────────────
    yield _sse_progress("Generating post-cleaning metadata snapshot…", step=4)

    after_snapshot = await _run_with_heartbeats(
        hb_queue, cleaner.generate_metadata_snapshot, df_after
    )

    async for hb in drain_heartbeats():
        yield hb

    yield _sse("snapshot_after", after_snapshot)
    yield _sse_progress(
        f"Post-cleaning snapshot ready — {after_snapshot['total_rows']:,} rows, "
        f"{after_snapshot['total_columns']} columns.",
        step=4,
    )

    # ── Step 5 · Persist cleaned CSV ─────────────────────────────────────────
    yield _sse_progress("Saving cleaned CSV to disk…", step=5)

    save_path = os.path.join(CLEANED_DIR, FILE_NAME)
    try:
        os.makedirs(CLEANED_DIR, exist_ok=True)
        await _run_with_heartbeats(
            hb_queue, lambda: df_after.to_csv(save_path, index=False)
        )
    except Exception as save_err:
        async for hb in drain_heartbeats():
            yield hb
        yield _sse(
            "error",
            {
                "detail": f"Data processed but failed to write to storage: {str(save_err)}",
                "status_code": 500,
            },
        )
        return

    async for hb in drain_heartbeats():
        yield hb

    yield _sse_progress(f"Cleaned file saved → {save_path}", step=5)

    # ── Step 6 · Final result ────────────────────────────────────────────────
    yield _sse_progress("Pipeline finished successfully.", step=6)
    yield _sse(
        "result",
        {
            "filename": filename,
            "is_2022_override_applied": is_2022_format,
            "before_processing": before_snapshot,
            "after_processing": after_snapshot,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/clean")
async def clean_summary(
    file: UploadFile = File(...),
    is_2022_format: bool = False,
):
    """
    Upload a raw procurement file.
    Returns a **text/event-stream** (SSE) response so the caller can observe
    each pipeline step in real-time without risking a connection timeout on
    large DataFrames.

    Event types emitted:
    - `progress`        — human-readable step message (streamed throughout)
    - `snapshot_before` — metadata dict before cleaning (streamed after Step 2)
    - `snapshot_after`  — metadata dict after cleaning  (streamed after Step 4)
    - `result`          — full final payload             (streamed after Step 6)
    - `error`           — error detail + HTTP status code
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty.",
        )

    return StreamingResponse(
        _clean_pipeline_stream(contents, file.filename, is_2022_format),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx response buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/download", response_class=FileResponse)
async def download_cleaned_file():
    """
    Retrieve and download a previously processed and cleaned CSV file.
    """
    file_path = os.path.join(CLEANED_DIR, FILE_NAME)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist or has expired.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=FILE_NAME,
    )