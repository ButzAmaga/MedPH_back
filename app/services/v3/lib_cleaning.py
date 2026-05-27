# lib_cleaning.py
from __future__ import annotations

import io
import os
import csv
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration Schemas
# ---------------------------------------------------------------------------

FLOAT_LIKELY_TEXT_COLS: list[str] = [
    "Procuring Entity (PE)", "Region", "Province", "City/Municipality",
    "Government Branch", "PE Organization Type", "PE Organization Type (Grouped)",
    "Bid Notice Status", "Award Reference No.", "Award Notice Status",
    "Country of Awardee", "Region of Awardee", "Province of Awardee",
    "City/Municipality of Awardee", "Awardee Size", "Awardee Joint Venture",
    "Contract Effectivity Date",
]

PHILGEPS_46_COLS: list[str] = [
    "Procuring Entity (PE)", "Region", "Province", "City/Municipality",
    "Government Branch", "PE Organization Type", "PE Organization Type (Grouped)",
    "Bid Reference No.", "Notice Title", "Classification", "Procurement Mode",
    "Business Category", "Funding Source", "Funding Instrument", "Trade Agreement",
    "Approved Budget of the Contract", "Published Date", "Closing Date",
    "Area of Delivery", "Contract Duration", "Calendar Type", "Line Item No",
    "Item Name", "Item Description", "Quantity", "UOM", "Item Budget",
    "Bid Notice Status", "Award Reference No.", "Award Title", "UNSPSC Code",
    "UNSPSC Description", "Published Date(Award)", "Award Date", "Contract Amount",
    "Award Notice Status", "Notice to Proceed Date", "Contract Effectivity Date",
    "Contract End Date", "Awardee Organization Name", "Country of Awardee",
    "Region of Awardee", "Province of Awardee", "City/Municipality of Awardee",
    "Awardee Size", "Awardee Joint Venture",
]

# Canonical first-column value present in headered (non-2022) PhilGEPS exports
_HEADERED_FIRST_COL = "procuring entity (pe)"

MEDICAL_KEYWORDS: list[str] = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical",
]
PATTERN = "|".join(MEDICAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Format detection result type
# ---------------------------------------------------------------------------

class FormatDetectionResult:
    """
    Carries the outcome of per-file format detection.

    Attributes
    ----------
    is_2022 : bool
        True  → file is headerless 2022-style; inject PHILGEPS_46_COLS.
        False → file has its own header row; read normally.
    method : str
        How the decision was reached:
        'override_true'  — caller explicitly set is_2022=True
        'override_false' — caller explicitly set is_2022=False
        'auto_header'    — first cell matched the known header label
        'auto_no_header' — first cell did NOT match; treated as headerless
        'auto_col_count' — CSV column count matched 46 exactly (fallback)
        'excel_sheet'    — Excel sheet: header presence checked per-sheet
    confidence : str  — 'high' | 'medium' | 'low'
    detail : str      — human-readable explanation
    """
    __slots__ = ("is_2022", "method", "confidence", "detail")

    def __init__(
        self,
        is_2022: bool,
        method: str,
        confidence: str,
        detail: str,
    ) -> None:
        self.is_2022     = is_2022
        self.method      = method
        self.confidence  = confidence
        self.detail      = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_2022":    self.is_2022,
            "method":     self.method,
            "confidence": self.confidence,
            "detail":     self.detail,
        }


# ---------------------------------------------------------------------------
# Step A — Per-file format detection
# ---------------------------------------------------------------------------

def detect_file_format(
    contents: bytes,
    filename: str,
    override: Optional[bool] = None,
) -> FormatDetectionResult:
    """
    Determines whether *filename* should be loaded with 2022-style logic
    (headerless CSV that needs PHILGEPS_46_COLS injected).

    Parameters
    ----------
    contents : bytes
        Raw file bytes.
    filename : str
        Original filename — used to determine the file extension.
    override : bool | None
        • True  → always treat as 2022 format.
        • False → always treat as headered format.
        • None  → auto-detect from file content.

    Returns
    -------
    FormatDetectionResult
        Detection outcome with is_2022 flag, method, confidence, and
        a human-readable detail string.

    Detection logic (CSV)
    ---------------------
    1. Read the first non-empty row with the csv module (fast, no pandas).
    2. If the first cell, lowercased, matches _HEADERED_FIRST_COL the file
       has a header → is_2022 = False (high confidence).
    3. Otherwise check the column count of that row:
       * == 46 → very likely headerless 2022 export → is_2022 = True  (medium)
       * != 46 → unrecognised format; default to headered → is_2022 = False (low)

    Detection logic (Excel)
    -----------------------
    Read sheet names and inspect the first cell of the first sheet exactly
    like the CSV path above.  Per-sheet override happens inside
    parse_single_file; the detection result here is advisory.
    """
    if override is not None:
        label = "True" if override else "False"
        return FormatDetectionResult(
            is_2022=override,
            method=f"override_{str(override).lower()}",
            confidence="high",
            detail=f"Caller explicitly set is_2022={label}; auto-detection skipped.",
        )

    ext = os.path.splitext(filename)[1].lower()

    # ── CSV detection ────────────────────────────────────────────────────────
    if ext == ".csv":
        try:
            # Decode just enough to read the first row
            sample = contents[:4096].decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(sample))
            first_row: list[str] = []
            for row in reader:
                stripped = [c.strip() for c in row]
                if any(stripped):          # skip blank leading rows
                    first_row = stripped
                    break

            if not first_row:
                return FormatDetectionResult(
                    is_2022=False,
                    method="auto_header",
                    confidence="low",
                    detail="CSV appears empty; defaulting to headered format.",
                )

            first_cell = first_row[0].lower()

            if first_cell == _HEADERED_FIRST_COL:
                return FormatDetectionResult(
                    is_2022=False,
                    method="auto_header",
                    confidence="high",
                    detail=(
                        f"First cell '{first_row[0]}' matches the known PhilGEPS "
                        "header label → file has its own header row."
                    ),
                )

            col_count = len(first_row)
            if col_count == len(PHILGEPS_46_COLS):
                return FormatDetectionResult(
                    is_2022=True,
                    method="auto_col_count",
                    confidence="medium",
                    detail=(
                        f"First cell '{first_row[0]}' is not a header label but "
                        f"column count ({col_count}) matches the 46-column 2022 "
                        "schema → treating as headerless 2022 export."
                    ),
                )

            # First cell doesn't look like a header AND column count ≠ 46.
            # Most likely a headered file from a non-2022 year with a
            # differently-shaped first column.
            return FormatDetectionResult(
                is_2022=False,
                method="auto_no_header",
                confidence="low",
                detail=(
                    f"First cell '{first_row[0]}' is not the expected header label "
                    f"and column count ({col_count}) ≠ 46.  Defaulting to headered "
                    "format; verify manually if results look wrong."
                ),
            )

        except Exception as exc:
            return FormatDetectionResult(
                is_2022=False,
                method="auto_header",
                confidence="low",
                detail=f"CSV header inspection failed ({exc}); defaulting to headered format.",
            )

    # ── Excel detection ──────────────────────────────────────────────────────
    elif ext in {".xlsx", ".xls"}:
        try:
            xl = pd.read_excel(
                io.BytesIO(contents),
                sheet_name=0,
                nrows=1,
                header=None,
                dtype=object,
            )
            first_cell = str(xl.iloc[0, 0]).strip().lower()

            if first_cell == _HEADERED_FIRST_COL:
                return FormatDetectionResult(
                    is_2022=False,
                    method="excel_sheet",
                    confidence="high",
                    detail=(
                        f"First cell '{xl.iloc[0,0]}' matches the known PhilGEPS "
                        "header label → Excel file has its own header row."
                    ),
                )

            col_count = xl.shape[1]
            if col_count == len(PHILGEPS_46_COLS):
                return FormatDetectionResult(
                    is_2022=True,
                    method="excel_sheet",
                    confidence="medium",
                    detail=(
                        f"First cell '{xl.iloc[0,0]}' is not a header label but "
                        f"column count ({col_count}) matches the 46-column 2022 "
                        "schema → treating as headerless 2022 Excel export."
                    ),
                )

            return FormatDetectionResult(
                is_2022=False,
                method="excel_sheet",
                confidence="low",
                detail=(
                    f"First cell '{xl.iloc[0,0]}' is not the expected header label "
                    f"and column count ({col_count}) ≠ 46.  Defaulting to headered "
                    "format."
                ),
            )

        except Exception as exc:
            return FormatDetectionResult(
                is_2022=False,
                method="excel_sheet",
                confidence="low",
                detail=f"Excel header inspection failed ({exc}); defaulting to headered format.",
            )

    # ── Unsupported extension ────────────────────────────────────────────────
    return FormatDetectionResult(
        is_2022=False,
        method="auto_header",
        confidence="low",
        detail=f"Extension '{ext}' not recognised; detection skipped.",
    )


# ---------------------------------------------------------------------------
# Step B — Parse a single file into a DataFrame
# ---------------------------------------------------------------------------

def parse_single_file(
    contents: bytes,
    filename: str,
    is_2022: bool,
) -> pd.DataFrame:
    """
    Parses raw bytes of a single uploaded file into a DataFrame.
    Handles .csv, .xlsx, and .xls.

    For 2022-format files the 46-column header is injected automatically.
    Raises ValueError for unsupported formats or empty workbooks.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        if is_2022:
            return pd.read_csv(
                io.BytesIO(contents),
                header=None,
                names=PHILGEPS_46_COLS,
                low_memory=False,
                encoding="utf-8",
            )
        return pd.read_csv(
            io.BytesIO(contents),
            header=0,
            low_memory=False,
            encoding="utf-8",
        )

    elif ext in {".xlsx", ".xls"}:
        excel_dict = pd.read_excel(io.BytesIO(contents), sheet_name=None, dtype=object)
        sheets: list[pd.DataFrame] = []
        for _, sheet_df in excel_dict.items():
            if is_2022 and sheet_df.shape[1] == len(PHILGEPS_46_COLS):
                first_val = str(sheet_df.iloc[0, 0]).strip().lower()
                if not (first_val.startswith("procuring") or "procuring entity" in first_val):
                    sheet_df.columns = PHILGEPS_46_COLS
            sheets.append(sheet_df)
        if not sheets:
            raise ValueError("The uploaded Excel workbook contains no data sheets.")
        return pd.concat(sheets, ignore_index=True)

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Accepted: .csv, .xlsx, .xls"
        )


# ---------------------------------------------------------------------------
# Step C — Detect formats then load and merge multiple files
# ---------------------------------------------------------------------------

class FilePayload:
    """
    Carries everything the pipeline needs to know about one uploaded file.

    Parameters
    ----------
    contents : bytes
        Raw file bytes.
    filename : str
        Original filename (used for extension detection and provenance tagging).
    is_2022_override : bool | None
        • True / False → caller forces the format.
        • None         → auto-detect from file content.
    """
    __slots__ = ("contents", "filename", "is_2022_override")

    def __init__(
        self,
        contents: bytes,
        filename: str,
        is_2022_override: Optional[bool] = None,
    ) -> None:
        self.contents         = contents
        self.filename         = filename
        self.is_2022_override = is_2022_override


def detect_all_formats(
    payloads: List[FilePayload],
) -> List[Dict[str, Any]]:
    """
    Runs format detection on every FilePayload and returns a list of
    detection report dicts, one per file, suitable for direct SSE emission.

    Each dict contains:
        filename    : str
        is_2022     : bool   — resolved value (override OR auto-detected)
        method      : str    — how the decision was made
        confidence  : str    — 'high' | 'medium' | 'low'
        detail      : str    — human-readable explanation
    """
    reports: list[Dict[str, Any]] = []
    for payload in payloads:
        result = detect_file_format(
            payload.contents,
            payload.filename,
            payload.is_2022_override,
        )
        reports.append({"filename": payload.filename, **result.to_dict()})
    return reports


def load_and_merge_files(
    payloads: List[FilePayload],
    detection_reports: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Parses every FilePayload (using the already-resolved is_2022 flag from
    *detection_reports*), aligns columns across all DataFrames, and
    concatenates them into one merged DataFrame.

    Parameters
    ----------
    payloads : list[FilePayload]
    detection_reports : list[dict]
        Output of detect_all_formats().  Must be in the same order as
        *payloads*; the resolved ``is_2022`` value is read from here.

    Returns
    -------
    merged_df : pd.DataFrame
        Combined DataFrame with a ``_source_file`` column for provenance.
    file_reports : list[dict]
        Per-file summary extending detection_reports with:
            row_count    : int
            column_count : int
            parse_error  : str | None
    """
    parsed_frames: list[pd.DataFrame] = []
    file_reports:  list[Dict[str, Any]] = []

    for payload, detection in zip(payloads, detection_reports):
        report = dict(detection)   # copy so we can add parse keys
        try:
            df = parse_single_file(
                payload.contents,
                payload.filename,
                detection["is_2022"],
            )
            df["_source_file"] = payload.filename
            report["row_count"]    = len(df)
            report["column_count"] = len(df.columns) - 1  # exclude _source_file
            report["parse_error"]  = None
            parsed_frames.append(df)
        except Exception as exc:
            report["row_count"]    = 0
            report["column_count"] = 0
            report["parse_error"]  = str(exc)

        file_reports.append(report)

    if not parsed_frames:
        raise ValueError(
            "No files could be parsed successfully. "
            "Check per-file errors in the file_report SSE event."
        )

    # ── Column alignment: union of all columns, fill gaps with pd.NA ─────────
    all_cols: list[str] = []
    seen: set[str] = set()
    for frame in parsed_frames:
        for col in frame.columns:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    aligned: list[pd.DataFrame] = []
    for frame in parsed_frames:
        missing = [c for c in all_cols if c not in frame.columns]
        if missing:
            frame = frame.assign(**{c: pd.NA for c in missing})
        aligned.append(frame[all_cols])

    merged_df = pd.concat(aligned, ignore_index=True)
    return merged_df, file_reports


# ---------------------------------------------------------------------------
# Step D — Normalize float-encoded text columns
# ---------------------------------------------------------------------------

def normalize_float_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces columns that arrive as float64 in raw PhilGEPS exports to
    nullable string, stripping trailing '.0' artefacts and collapsing
    empty strings to pd.NA.
    """
    for col in FLOAT_LIKELY_TEXT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda x: pd.NA if pd.isna(x) else str(x).strip())
        df[col] = df[col].replace("", pd.NA)
    return df


# ---------------------------------------------------------------------------
# Step E — Filter to medical/health rows
# ---------------------------------------------------------------------------

def filter_medical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retains only rows whose UNSPSC Description, Item Name, or Item
    Description matches at least one medical/health keyword.
    """
    col_unspsc = "UNSPSC Description"
    col_iname  = "Item Name"
    col_idesc  = "Item Description"

    unspsc_match = (
        df[col_unspsc].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_unspsc in df.columns else pd.Series(False, index=df.index)
    )
    iname_match = (
        df[col_iname].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_iname in df.columns else pd.Series(False, index=df.index)
    )
    idesc_match = (
        df[col_idesc].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_idesc in df.columns else pd.Series(False, index=df.index)
    )

    return df.loc[unspsc_match | iname_match | idesc_match].copy()


# ---------------------------------------------------------------------------
# Step F — Deduplicate
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops exact duplicate rows, excluding the ``_source_file`` provenance
    column from the comparison key.
    """
    key_cols = [c for c in df.columns if c != "_source_file"]
    return df.drop_duplicates(subset=key_cols).copy()


# ---------------------------------------------------------------------------
# Step G — Impute missing values
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values:
    * Numeric columns  → median (or 0 when the entire column is NaN).
    * Categorical cols → mode  (or 'N/A' when no mode is available).
    Datetime columns are left untouched.
    """
    numeric_targets = [
        "Contract Amount", "Item Budget", "Quantity",
        "Approved Budget of the Contract", "Line Item No",
    ]
    numeric_cols = [c for c in numeric_targets if c in df.columns]

    for col in numeric_cols:
        series     = pd.to_numeric(df[col], errors="coerce")
        median_val = series.median()
        df[col]    = series.fillna(median_val if pd.notna(median_val) else 0)

    cat_cols = [
        c for c in df.columns
        if c not in numeric_cols
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for col in cat_cols:
        mode_series = df[col].mode()
        fill_val = (
            mode_series.iloc[0]
            if len(mode_series) > 0 and pd.notna(mode_series.iloc[0])
            else "N/A"
        )
        df[col] = (
            df[col]
            .fillna(fill_val)
            .replace("", fill_val)
            .replace("nan", "N/A")
            .replace("None", "N/A")
        )

    return df


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def generate_metadata_snapshot(
    df: pd.DataFrame,
    preview_rows: int = 3,
) -> Dict[str, Any]:
    """
    Returns a JSON-safe dict with row/column counts and a short row preview.
    NaN / NA values are replaced with None to avoid invalid float JSON tokens.
    """
    clean_preview = df.head(preview_rows).replace({pd.NA: None, np.nan: None})
    return {
        "total_rows":     len(df),
        "total_columns":  len(df.columns),
        "sample_preview": clean_preview.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def execute_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full cleaning chain in one call.
    Prefer calling individual step functions from the SSE route so that
    granular progress events can be emitted between each stage.
    """
    df = normalize_float_text_columns(df)
    df = filter_medical_rows(df)
    df = deduplicate(df)
    df = impute_missing_values(df)
    return df