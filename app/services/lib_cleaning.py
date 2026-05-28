# cleaner.py
from __future__ import annotations

import io
import os
import cudf as pd
import cupy as np
from typing import Any, Tuple, Dict

# --- Configuration Schemas ---
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

MEDICAL_KEYWORDS = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical",
]
PATTERN = "|".join(MEDICAL_KEYWORDS)


def load_dataframe_from_stream(contents: bytes, filename: str, is_2022: bool) -> pd.DataFrame:
    """Parses raw uploaded file bytes into a cudf DataFrame."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        if is_2022:
            return pd.read_csv(io.BytesIO(contents), header=None, names=PHILGEPS_46_COLS, low_memory=False, encoding="utf-8")
        return pd.read_csv(io.BytesIO(contents), header=0, low_memory=False, encoding="utf-8")
    elif ext in {".xlsx", ".xls"}:
        excel_dict = pd.read_excel(io.BytesIO(contents), sheet_name=None, dtype=object)
        sheets = []
        for _, s in excel_dict.items():
            if is_2022 and s.shape[1] == len(PHILGEPS_46_COLS):
                first_val = str(s.iloc[0, 0]).lower()
                if not (first_val.startswith("procuring") or "procuring entity" in first_val):
                    s.columns = PHILGEPS_46_COLS
            sheets.append(s)
        if not sheets:
            raise ValueError("The uploaded Excel workbook contains no data sheets.")
        return pd.concat(sheets, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def normalize_float_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces columns that arrive as float64 in raw exports to nullable string."""
    for c in FLOAT_LIKELY_TEXT_COLS:
        if c not in df.columns:
            continue
        df[c] = df[c].map(lambda x: pd.NA if pd.isna(x) else str(x).strip())
        df[c] = df[c].replace("", pd.NA)
    return df


def filter_medical_rows_step00_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Filters data frame to keep rows matching medical keywords."""
    col_u, col_in, col_id = "UNSPSC Description", "Item Name", "Item Description"
    unspsc_match = df[col_u].astype(str).str.contains(PATTERN, case=False, na=False) if col_u in df.columns else pd.Series(False, index=df.index)
    in_match = df[col_in].astype(str).str.contains(PATTERN, case=False, na=False) if col_in in df.columns else pd.Series(False, index=df.index)
    id_match = df[col_id].astype(str).str.contains(PATTERN, case=False, na=False) if col_id in df.columns else pd.Series(False, index=df.index)
    return df.loc[unspsc_match | in_match | id_match].copy()


def handle_missing_values_philgeps(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values (numeric median / categorical mode)."""
    numeric_cols = ["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract", "Line Item No"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        median_val = series.median()
        df[col] = series.fillna(median_val if pd.notna(median_val) else 0)

    cat_cols = [c for c in df.columns if c not in numeric_cols and not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "N/A"
        df[col] = (df[col].fillna(fill_val).replace("", fill_val).replace("nan", "N/A").replace("None", "N/A"))
    return df


def generate_metadata_snapshot(df: pd.DataFrame, preview_rows: int = 3) -> Dict[str, Any]:
    """Generates structural metadata metrics and a clean JSON-safe preview dict."""
    # Replace NaN/NA elements with None to avoid invalid float JSON representations
    clean_df = df.head(preview_rows).replace({pd.NA: None, np.nan: None})
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "sample_preview": clean_df.to_dict(orient="records")
    }


def execute_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executes the full chain of data cleaning rules sequentially on a DataFrame."""
    processed_df = normalize_float_text_columns(df)
    processed_df = filter_medical_rows_step00_rule(processed_df)
    processed_df = processed_df.drop_duplicates()
    processed_df = handle_missing_values_philgeps(processed_df)
    return processed_df