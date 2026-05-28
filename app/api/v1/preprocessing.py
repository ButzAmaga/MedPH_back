import os
import pathlib
from fastapi import APIRouter, HTTPException, status
import cudf as pd
from pydantic import BaseModel
from services import lib_processing as preprocessor
from typing import List

router = APIRouter(tags=["Data Preprocessing & Features"])

# Define paths for local staging structures
CLEANED_INPUT_DIR = os.path.join("output_source", "01")
PREPROCESSED_OUTPUT_DIR = os.path.join("output_source", "02")
CLEANED_FILE_NAME = "cleaned.csv"

# Explicit inner validation metric structure
class PreprocessingMetrics(BaseModel):
    total_records: int
    column_count: int
    columns_present: List[str]
    unique_years_computed: List[int]
    unique_regions_collapsed: List[str]

class PreprocessResultResponse(BaseModel):
    metrics: PreprocessingMetrics

@router.post("/preprocessing", response_model=PreprocessResultResponse)
async def process_file():
    """
    Locates an existing cleaned dataset on disk, performs Winsorization, 
    categorical dimension collapsing, and temporal feature extraction.
    """
    # 1. Clean input path verification safety mapping
    input_path = os.path.join(CLEANED_INPUT_DIR, CLEANED_FILE_NAME)
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Target source file not found under context cache path: '{input_path}'. Please clean it first."
        )

    # 2. Extract Data Matrix
    try:
        # Use simple standard cudf reading configuration since data was cleaned by Step 1
        df_cleaned = pd.read_csv(input_path, low_memory=False, encoding="utf-8")
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read data matrix: {str(read_err)}"
        )

    if df_cleaned.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target cleaned database file is empty. Aborting pipeline process."
        )

    # 3. Execute Preprocessing pipeline transformations
    df_preprocessed = preprocessor.execute_preprocessing_pipeline(df_cleaned)
    metrics_snapshot = preprocessor.generate_preprocessing_snapshot(df_preprocessed)

    # 4. Save Resulting File Asset
    try:
        # Guarantee directory node tree existence
        os.makedirs(PREPROCESSED_OUTPUT_DIR, exist_ok=True)
        
        # Transform filename structure from '_cleaned.csv' to '_preprocessed.csv' cleanly
        path_obj = pathlib.Path(CLEANED_FILE_NAME)
        base_name = path_obj.stem
        
        # Replace or append the file descriptor modifier suffix pattern
        if base_name.endswith("_cleaned"):
            output_name = f"{base_name.replace('_cleaned', '_preprocessed')}{path_obj.suffix}"
        else:
            output_name = f"{base_name}_preprocessed{path_obj.suffix}"
            
        output_save_path = os.path.join(PREPROCESSED_OUTPUT_DIR, output_name)
        
        # Export processed array dropping system range indices
        df_preprocessed.to_csv(output_save_path, index=False)
        
    except Exception as save_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data preprocessing successful but file writing failed: {str(save_err)}"
        )

    return PreprocessResultResponse(
        metrics=metrics_snapshot
    )