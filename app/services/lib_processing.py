import pandas as pd
import numpy as np

def execute_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data preprocessing, feature engineering, and analytics 
    normalizations to a single cleaned DataFrame instance.
    """
    if df.empty:
        return df

    processed_df = df.copy()

    # 1. Temporal Component Parsing & Sorting
    if "Published Date" in processed_df.columns:
        processed_df["Published Date"] = pd.to_datetime(processed_df["Published Date"], errors='coerce')
        default_date = pd.Timestamp('2026-01-01')
        processed_df["Published Date"] = processed_df["Published Date"].fillna(default_date)
        
        processed_df = processed_df.sort_values(by="Published Date", ascending=True).reset_index(drop=True)
        
        processed_df["Year"] = processed_df["Published Date"].dt.year
        processed_df["Month"] = processed_df["Published Date"].dt.month
        processed_df["Quarter"] = processed_df["Published Date"].dt.quarter
        
        processed_df["Published Date"] = processed_df["Published Date"].dt.strftime('%Y-%m-%d')

    # 2. Outlier Boundary Capping (Winsorization limits)
    financial_targets = ["Approved Budget of the Contract", "Item Budget", "Contract Amount"]
    for col in financial_targets:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)
            q99 = processed_df[col].quantile(0.99)
            if q99 > 0:
                processed_df[col] = np.where(processed_df[col] > q99, q99, processed_df[col])

    # 3. High-Cardinality Categorical Collapsing
    categorical_targets = ["Region", "Business Category", "PE Organization Type"]
    for col in categorical_targets:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna("None").astype(str).str.strip()
            counts = processed_df[col].value_counts()
            total_rows = len(processed_df)
            
            rare_categories = counts[counts / total_rows < 0.015].index
            if not rare_categories.empty:
                processed_df[col] = processed_df[col].replace(rare_categories, "Others")

    return processed_df

def generate_preprocessing_snapshot(df: pd.DataFrame) -> dict:
    """
    Generates high-level feature dimensions metadata for evaluation validation telemetry.
    Converts native NumPy types back into native Python types to satisfy serialization requirements.
    """
    if df.empty:
        return {"total_records": 0, "column_count": 0, "columns_present": [], "unique_years_computed": [], "unique_regions_collapsed": []}
    
    # convert the number to object to be not included in PCA ready 
    df["UNSPSC Code"] = df["UNSPSC Code"].astype("object")
        
    return {
        "total_records": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns_present": [str(c) for c in df.columns],
        "pca_ready_columns": [
            str(c) for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
        ],
        #  .tolist() converts NumPy arrays (e.g. numpy.int32) into native Python types
        "unique_years_computed": [int(y) for y in df["Year"].unique()] if "Year" in df.columns else [],
        "unique_regions_collapsed": [str(r) for r in df["Region"].unique()] if "Region" in df.columns else []
    }