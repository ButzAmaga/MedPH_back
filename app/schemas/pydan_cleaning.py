from pydantic import BaseModel
from typing import List, Dict, Any

class DataSnapshot(BaseModel):
    total_rows: int
    total_columns: int
    sample_preview: List[Dict[str, Any]]

class CleanResultResponse(BaseModel):
    filename: str
    is_2022_override_applied: bool
    before_processing: DataSnapshot
    after_processing: DataSnapshot