import os
import pathlib
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from schemas.pydan_cleaning import CleanResultResponse
from services import lib_cleaning as cleaner
from fastapi.responses import FileResponse 

router = APIRouter(
    prefix="/cleaning",
    tags=["Procurement Processing"])

# Define the base directory where you want to cache or save the cleaned datasets
CLEANED_DIR = os.path.join("output_source", "01") 
FILE_NAME = "cleaned.csv"

@router.post("/clean", response_model=CleanResultResponse)
async def clean_summary(
    file: UploadFile = File(...),
    is_2022_format: bool = False
):
    """
    Upload a raw procurement file to audit metadata state alterations 
    and save the output to the server disk storage.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    try:
        df_before = cleaner.load_dataframe_from_stream(contents, file.filename, is_2022_format)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Parsing error: {str(err)}")

    before_snapshot = cleaner.generate_metadata_snapshot(df_before)
    df_after = cleaner.execute_cleaning_pipeline(df_before)
    after_snapshot = cleaner.generate_metadata_snapshot(df_after)

    # --- Save the Cleaned File to Storage ---
    try:
        # 1. Ensure the destination directory path tree exists
        os.makedirs(CLEANED_DIR, exist_ok=True)
        
        # 2. Extract base name and extension (e.g., 'MedFlowSampleData' and '.csv')
        
        cleaned_filename = FILE_NAME
        
        # 3. Create absolute save path location
        save_path = os.path.join(CLEANED_DIR, cleaned_filename)
        
        # 4. Save DataFrame to storage disk
        # (index=False avoids adding an unmapped numeric ID column row key)
        df_after.to_csv(save_path, index=False)
        
    except Exception as save_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data processed successfully but failed to write to storage: {str(save_err)}"
        )
    # ----------------------------------------

    return CleanResultResponse(
        filename=file.filename,
        is_2022_override_applied=is_2022_format, # no header
        before_processing=before_snapshot,
        after_processing=after_snapshot
    )


@router.get("/download", response_class=FileResponse)
async def download_cleaned_file():
    """
    Retrieve and download a previously processed and cleaned CSV file.
    """
    # 1. Construct the absolute path to the requested file
    file_path = os.path.join(CLEANED_DIR, FILE_NAME)

    # 2. Check if the file actually exists on the server disk
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="The requested file does not exist or has expired."
        )

    # 3. Return the file as a stream with the correct headers for browser downloading
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=FILE_NAME  # This enforces browser download instead of rendering text
    )