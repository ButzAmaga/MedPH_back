# pca_library.py
import os
import json
from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Column config fallback from original script
POLICY_PCA_BASE_COLUMNS = [
    "Approved Budget of the Contract",
    "Quantity",
    "Item Budget",
    "Contract Amount"
]

def run_pca_pipeline(
    input_csv_path: str = "output_source/02/cleaned_preprocessed.csv",
    matrix_mode: str = "base", 
    max_pc: int = 3, 
    output_dir: str = "output_source/03",
    selected_columns: List = []
    # results_dir: str = "results/03/Clustering"
) -> str:
    """
    Reads a preprocessed CSV from disk, runs standard PCA, 
    appends PC scores, and outputs a combined CSV and diagnostic JSON.
    """

    

    if not isinstance(selected_columns, list):
        raise TypeError(f"selected_columns must be list, got {type(selected_columns)}")
    
    
    # 1. Validation check on the target input file
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Target dataset file not found at: '{input_csv_path}'")
        
    df = pd.read_csv(input_csv_path)
    print("cols:", df.columns.tolist())
    print("selected cols:", selected_columns)
    if df.empty:
        raise ValueError(f"The source dataset file at '{input_csv_path}' is empty.")

    # 2. Setup internal folder trees
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(results_dir, exist_ok=True)

    # 3. Select appropriate columns based on the requested matrix mode
    if matrix_mode == "base":
        available_cols = [
            c for c in selected_columns 
            if c in df.columns and not pd.api.types.is_string_dtype(df[c])
        ]
        
        if not available_cols: 
            raise ValueError("None of the base PCA columns were discovered in the dataset matrix.")
        working_df = df[available_cols].copy()

    elif matrix_mode == "scaled":
        working_df = df.select_dtypes(include=[np.number]).copy()
    else:
        raise ValueError(f"Unsupported matrix mode: '{matrix_mode}'. Choose 'base' or 'scaled'.")

    print("Step 4 of PCA")
    print(working_df)
    # 4. Fill missing entries and drop zero/near-zero variance constants
    working_df = working_df.fillna(working_df.median())
    low_variance_cols = [col for col in working_df.columns if working_df[col].var() < 1e-18]
    if low_variance_cols:
        working_df = working_df.drop(columns=low_variance_cols)

    if working_df.empty:
        raise ValueError("No valid numeric features left to process PCA after filtering variance thresholds.")

    # Save exact features slice for tracking integrity
    features_csv_path = os.path.join(output_dir, "philgeps_clustering_features.csv")
    working_df.to_csv(features_csv_path, index=False)

    # 5. Scale features to standard space (Zero mean, unit variance)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(working_df)

    # 6. Apply Principal Component Analysis
    n_components = min(max_pc, working_df.shape[1])
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(scaled_data)

    # 7. Merge PC column fields directly back into the original dataset structure
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(pc_scores, columns=pc_cols, index=df.index)
    processed_df = pd.concat([df, scores_df], axis=1)
    
    # Save precisely to the path requested: output/03/pca_processed.csv
    processed_csv_path = os.path.join(output_dir, "pca_processed.csv")
    processed_df.to_csv(processed_csv_path, index=False)

    # 8. Export Diagnostics Metadata
    diagnostics = {
        "matrix_mode": matrix_mode,
        "features_used": working_df.columns.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "components_matrix": pca.components_.tolist(),
        "scaler_means": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
        "scaler_vars": scaler.var_.tolist() if hasattr(scaler, 'var_') else []
    }
    '''
    json_diagnostics_path = os.path.join(results_dir, "pca_theme_clustering.json")
    with open(json_diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=4)
    '''

    return processed_csv_path, diagnostics