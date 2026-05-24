# pca_routes.py
from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import JSONResponse
from services.lib_pca import run_pca_pipeline

router = APIRouter(
    prefix="/analytics",
    tags=["Dimensionality Reduction"]
)

@router.post("/pca", status_code=status.HTTP_200_OK)
async def calculate_pca(
    matrix_mode: str = Query("base", description="Feature matrix subset selection: 'base' or 'scaled'"),
    max_components: int = Query(3, ge=1, le=10, description="Maximum number of Principal Components to extract"),
    custom_input_path: str = Query("output_source/02/cleaned_preprocessed.csv", description="The local relative server path to the cleaned dataset")
):
    """
    Triggers the PCA reduction pipeline using a cleaned file stored on the local disk.
    Merges original indices with principal configurations and drops the output to output/03/pca_processed.csv.
    """
    try:
        # Unpack both values from the updated library pipeline call
        saved_file_destination, pca_diagnostics = run_pca_pipeline(
            input_csv_path=custom_input_path,
            matrix_mode=matrix_mode,
            max_pc=max_components
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "Success",
                "message": "PCA processed file generated successfully from local storage data.",
                "input_source": custom_input_path,
                "output_data_path": saved_file_destination,
                "diagnostics": pca_diagnostics  # Included raw payload directly in the response
            }
        )   

    except FileNotFoundError as file_missing_error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(file_missing_error)
        )
    except ValueError as format_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mathematical or column configuration error: {str(format_error)}"
        )
    except Exception as runtime_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error halted the localized PCA calculation: {str(runtime_error)}"
        )