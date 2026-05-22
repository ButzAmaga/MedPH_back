from fastapi import FastAPI
from api.v1.cleaning import router as cleaning_router
from api.v1.preprocessing import router as preprocessing_router


app = FastAPI(
    title="MedFlowPH — Clean Architecture API",
    description="Highly organized production API structure separating routers, schemas, and processing units.",
    version="2.0.0"
)
    
# Include separated structural endpoints
app.include_router(cleaning_router)
app.include_router(preprocessing_router)