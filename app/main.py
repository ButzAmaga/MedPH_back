from fastapi import FastAPI
from api.v1.cleaning import router as cleaning_router
from api.v2.cleaning import router as cleaning_router_stream
from api.v1.preprocessing import router as preprocessing_router
from api.v2.preprocessing import router as preprocessing_router_stream
from api.v2.pca import router as pca_router_stream
from api.v1.kmeans import router as kmeans_router

app = FastAPI(
    title="MedFlowPH — Clean Architecture API",
    description="Highly organized production API structure separating routers, schemas, and processing units.",
    version="2.0.0"
)
    
# Include separated structural endpoints
app.include_router(cleaning_router_stream)
app.include_router(preprocessing_router_stream)
app.include_router(pca_router_stream)
app.include_router(kmeans_router)