import os
import json
import asyncio
from typing import AsyncGenerator, List, Optional
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from services.lib_SSE import run_with_heartbeats, sse, sse_progress
from services.v3 import lib_cleaning as cleaner
from services.v3.lib_cleaning import FilePayload

router = APIRouter(
    prefix="/cleaning/v3",
    tags=["Procurement Processing"],
)

CLEANED_DIR = os.path.join("output_source", "01")
FILE_NAME   = "cleaned.csv"


# ---------------------------------------------------------------------------
# Streaming pipeline generator
# ---------------------------------------------------------------------------

async def _clean_pipeline_stream(
    payloads: List[FilePayload],
) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames for every fragmented pipeline step.

    Event types emitted
    -------------------
    progress           — human-readable step message (throughout)
    format_detection   — per-file format detection results (Step 1)
    file_report        — per-file parse summary after Step 2
    snapshot_before    — merged raw DataFrame metadata (Step 3)
    snapshot_after     — cleaned DataFrame metadata (Step 7)
    result             — final consolidated payload (Step 8)
    error              — {detail, status_code}
    """

    hb_queue: asyncio.Queue = asyncio.Queue()

    async def drain():
        while not hb_queue.empty():
            yield await hb_queue.get()

    total_files = len(payloads)

    # ── Step 1 · Detect format for every file ───────────────────────────────
    yield sse_progress(
        f"Detecting format for {total_files} file(s) "
        "(checking for 2022-style headerless exports)…",
        step=1,
    )

    detection_reports = await run_with_heartbeats(
        hb_queue,
        cleaner.detect_all_formats,
        payloads,
    )

    async for hb in drain():
        yield hb

    yield sse("format_detection", {"files": detection_reports})

    fmt_2022  = [r for r in detection_reports if r["is_2022"]]
    fmt_other = [r for r in detection_reports if not r["is_2022"]]
    yield sse_progress(
        f"Format detection complete — "
        f"{len(fmt_2022)} file(s) identified as 2022-style (headerless), "
        f"{len(fmt_other)} file(s) as standard headered format.",
        step=1,
    )

    # ── Step 2 · Parse & merge all files ────────────────────────────────────
    yield sse_progress(
        f"Parsing and merging {total_files} file(s) using resolved formats…",
        step=2,
    )

    try:
        merged_df, file_reports = await run_with_heartbeats(
            hb_queue,
            cleaner.load_and_merge_files,
            payloads,
            detection_reports,
        )
    except ValueError as exc:
        async for hb in drain():
            yield hb
        yield sse("error", {"detail": str(exc), "status_code": 400})
        return
    except Exception as exc:
        async for hb in drain():
            yield hb
        yield sse("error", {"detail": f"Merge error: {exc}", "status_code": 422})
        return

    async for hb in drain():
        yield hb

    yield sse("file_report", {"files": file_reports})

    failed = [r for r in file_reports if r["parse_error"]]
    ok     = [r for r in file_reports if not r["parse_error"]]
    yield sse_progress(
        f"Parsed {len(ok)}/{total_files} file(s) successfully"
        + (f"; {len(failed)} failed — see file_report event." if failed else ".")
        + f"  Merged shape: {len(merged_df):,} rows × {len(merged_df.columns)} columns.",
        step=2,
    )

    # ── Step 3 · Pre-cleaning snapshot ──────────────────────────────────────
    yield sse_progress("Generating pre-cleaning metadata snapshot…", step=3)

    before_snapshot = await run_with_heartbeats(
        hb_queue, cleaner.generate_metadata_snapshot, merged_df
    )

    async for hb in drain():
        yield hb

    yield sse("snapshot_before", before_snapshot)
    yield sse_progress(
        f"Pre-cleaning snapshot ready — "
        f"{before_snapshot['total_rows']:,} rows, "
        f"{before_snapshot['total_columns']} columns.",
        step=3,
    )

    # ── Step 4 · Normalize float-encoded text columns ───────────────────────
    yield sse_progress("Normalizing float-encoded text columns…", step=4)

    df = await run_with_heartbeats(
        hb_queue, cleaner.normalize_float_text_columns, merged_df
    )

    async for hb in drain():
        yield hb

    yield sse_progress("Float-text normalization complete.", step=4)

    # ── Step 5 · Filter to medical/health rows ───────────────────────────────
    yield sse_progress("Filtering rows by medical/health keywords…", step=5)

    rows_before_filter = len(df)
    df = await run_with_heartbeats(
        hb_queue, cleaner.filter_medical_rows, df
    )

    async for hb in drain():
        yield hb

    yield sse_progress(
        f"Keyword filter complete — "
        f"kept {len(df):,} / {rows_before_filter:,} rows.",
        step=5,
    )

    # ── Step 6 · Deduplicate ─────────────────────────────────────────────────
    yield sse_progress("Removing duplicate rows…", step=6)

    rows_before_dedup = len(df)
    df = await run_with_heartbeats(
        hb_queue, cleaner.deduplicate, df
    )

    async for hb in drain():
        yield hb

    dropped = rows_before_dedup - len(df)
    yield sse_progress(
        f"Deduplication complete — "
        f"dropped {dropped:,} duplicate(s), {len(df):,} rows remain.",
        step=6,
    )

    # ── Step 7 · Impute missing values + post-cleaning snapshot ─────────────
    yield sse_progress("Imputing missing values (median / mode)…", step=7)

    df = await run_with_heartbeats(
        hb_queue, cleaner.impute_missing_values, df
    )

    async for hb in drain():
        yield hb

    yield sse_progress("Missing-value imputation complete.", step=7)

    after_snapshot = await run_with_heartbeats(
        hb_queue, cleaner.generate_metadata_snapshot, df
    )

    async for hb in drain():
        yield hb

    yield sse("snapshot_after", after_snapshot)
    yield sse_progress(
        f"Post-cleaning snapshot ready — "
        f"{after_snapshot['total_rows']:,} rows, "
        f"{after_snapshot['total_columns']} columns.",
        step=7,
    )

    # ── Step 8 · Persist cleaned CSV ─────────────────────────────────────────
    yield sse_progress("Saving cleaned CSV to disk…", step=8)

    save_path = os.path.join(CLEANED_DIR, FILE_NAME)
    try:
        os.makedirs(CLEANED_DIR, exist_ok=True)
        await run_with_heartbeats(
            hb_queue, lambda: df.to_csv(save_path, index=False)
        )
    except Exception as save_err:
        async for hb in drain():
            yield hb
        yield sse(
            "error",
            {
                "detail": f"Data processed but failed to write: {save_err}",
                "status_code": 500,
            },
        )
        return

    async for hb in drain():
        yield hb

    yield sse_progress(f"Cleaned file saved → {save_path}", step=8)

    # ── Step 9 · Final result ────────────────────────────────────────────────
    yield sse_progress("Pipeline finished successfully.", step=9)
    yield sse(
        "result",
        {
            "files_submitted":  total_files,
            "files_parsed_ok":  len(ok),
            "files_failed":     len(failed),
            "format_detection": detection_reports,
            "file_reports":     file_reports,
            "before_processing": before_snapshot,
            "after_processing":  after_snapshot,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/clean")
async def clean_summary(
    files: List[UploadFile] = File(...),
    files_meta: Optional[str] = Form(default=None),
):
    """
    Upload one or more raw PhilGEPS procurement files (CSV / XLSX / XLS).

    Returns a **text/event-stream** (SSE) response so the caller can observe
    each pipeline step in real-time.

    Form fields
    -----------
    files      : one or more file uploads (multipart/form-data)
    files_meta : optional JSON string — a list of per-file metadata objects.

                 Schema (each object):
                 {
                   "filename":  "<must match the uploaded file's filename>",
                   "is_2022":   true | false | null
                 }

                 • true  → force 2022-style (headerless) parsing for that file.
                 • false → force standard headered parsing.
                 • null  → auto-detect from file content (default when omitted).

                 Files not listed in files_meta default to auto-detection.

    Example files_meta value:
        [
          {"filename": "philgeps_2022_q1.csv", "is_2022": true},
          {"filename": "philgeps_2024_q3.csv", "is_2022": false},
          {"filename": "philgeps_unknown.csv",  "is_2022": null}
        ]

    SSE event types emitted
    -----------------------
    progress          — human-readable step message
    format_detection  — per-file format detection: is_2022, method, confidence, detail
    file_report       — per-file parse summary (rows, columns, errors)
    snapshot_before   — merged raw DataFrame metadata
    snapshot_after    — cleaned DataFrame metadata
    result            — full final payload
    error             — {detail, status_code}
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were uploaded.",
        )

    # ── Parse optional per-file metadata ────────────────────────────────────
    meta_map: dict[str, Optional[bool]] = {}   # filename → override or None
    if files_meta:
        try:
            meta_list = json.loads(files_meta)
            if not isinstance(meta_list, list):
                raise ValueError("files_meta must be a JSON array.")
            for entry in meta_list:
                fname    = entry.get("filename")
                override = entry.get("is_2022")   # true / false / null / absent
                if fname:
                    # Normalise: only accept bool or None
                    if isinstance(override, bool):
                        meta_map[fname] = override
                    else:
                        meta_map[fname] = None     # treat anything else as auto
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid files_meta JSON: {exc}",
            )

    # ── Build FilePayload list ───────────────────────────────────────────────
    payloads: list[FilePayload] = []
    for upload in files:
        contents = await upload.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{upload.filename}' is empty.",
            )
        override = meta_map.get(upload.filename, None)   # None → auto-detect
        payloads.append(FilePayload(contents, upload.filename, override))

    return StreamingResponse(
        _clean_pipeline_stream(payloads),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@router.get("/download", response_class=FileResponse)
async def download_cleaned_file():
    """
    Retrieve and download the most recently processed and cleaned CSV file.
    """
    file_path = os.path.join(CLEANED_DIR, FILE_NAME)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist or has not been generated yet.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=FILE_NAME,
    )