from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.v1.cleaning import router as cleaning_router
from api.v2.cleaning import router as cleaning_router_stream
from api.v1.preprocessing import router as preprocessing_router
from api.v2.preprocessing import router as preprocessing_router_stream
from api.v2.pca import router as pca_router_stream
from api.v2.kmeans import router as kmeans_router_stream
from api.v2.dbscan import router as dbscan_router_stream

app = FastAPI(
    title="MedFlowPH — Clean Architecture API",
    description="Highly organized production API structure separating routers, schemas, and processing units.",
    version="2.0.0"
)

# Serve static files (images)
app.mount("/static", StaticFiles(directory="static"), name="static")
    
# Include separated structural endpoints
app.include_router(cleaning_router_stream)
app.include_router(preprocessing_router_stream)
app.include_router(pca_router_stream)
app.include_router(dbscan_router_stream)