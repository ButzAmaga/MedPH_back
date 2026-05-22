import pandas as pd
import numpy as np

def execute_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data preprocessing, feature engineering, and analytics 
    normalizations to a single cleaned DataFrame instance.
    """
    if df.empty:
        return df

    # Work on an isolated copy to prevent setting-with-copy slicing warnings
    processed_df = df.copy()

    # 1. Temporal Component Parsing & Sorting
    if "Published Date" in processed_df.columns:
        processed_df["Published Date"] = pd.to_datetime(processed_df["Published Date"], errors='coerce')
        # Fill missing dates to avoid breaking temporal feature extractions
        default_date = pd.Timestamp('2026-01-01')
        processed_df["Published Date"] = processed_df["Published Date"].fillna(default_date)
        
        # Chronological sort order matching historical baseline scripts
        processed_df = processed_df.sort_values(by="Published Date", ascending=True).reset_index(drop=True)
        
        # Feature Engineering: Extract time dimensions
        processed_df["Year"] = processed_df["Published Date"].dt.year
        processed_df["Month"] = processed_df["Published Date"].dt.month
        processed_df["Quarter"] = processed_df["Published Date"].dt.quarter
        
        # Revert date field back to standardized string representation for CSV writing
        processed_df["Published Date"] = processed_df["Published Date"].dt.strftime('%Y-%m-%d')

    # 2. Outlier Boundary Capping (Winsorization Winsor limits)
    # Caps extreme financial metrics to safeguard downstream statistical metrics
    financial_targets = ["Approved Budget of the Contract", "Item Budget", "Contract Amount"]
    for col in financial_targets:
        if col in processed_df.columns:
            # Enforce clean numeric baseline
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)
            
            q99 = processed_df[col].quantile(0.99)
            if q99 > 0:
                # Cap values exceeding the 99th percentile boundary
                processed_df[col] = np.where(processed_df[col] > q99, q99, processed_df[col])

    # 3. High-Cardinality Categorical Collapsing
    # Groups thin tail nodes under an 'Others' label bucket to limit dimensionality explosion
    categorical_targets = ["Region", "Business Category", "PE Organization Type"]
    for col in categorical_targets:
        if col in processed_df.columns:
            # Enforce consistent textual representation
            processed_df[col] = processed_df[col].fillna("None").astype(str).str.strip()
            
            # Identify frequencies
            counts = processed_df[col].value_counts()
            total_rows = len(processed_df)
            
            # Find categories with less than 1.5% representation in the dataset
            rare_categories = counts[counts / total_rows < 0.015].index
            if not rare_categories.empty:
                processed_df[col] = processed_df[col].replace(rare_categories, "Others")

    return processed_df

def generate_preprocessing_snapshot(df: pd.DataFrame) -> dict:
    """
    Generates high-level feature dimensions metadata for evaluation validation telemetry.
    """
    if df.empty:
        return {"records": 0, "columns": []}
        
    return {
        "total_records": len(df),
        "column_count": len(df.columns),
        "columns_present": list(df.columns),
        "unique_years_computed": list(df["Year"].unique()) if "Year" in df.columns else [],
        "unique_regions_collapsed": list(df["Region"].unique()) if "Region" in df.columns else []
    }