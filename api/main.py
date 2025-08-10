from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.score import router as score_router
from api.routers.search import router as search_router 

app = FastAPI(
    title="Health Inspection Compliance Coach",
    version="0.1.0",
    description="Predict next inspection risk and likely violation categories for NYC restaurants."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "model_version": "0.1.0",
        "data_window_days": "1095",
        "source": "NYC Open Data (inspections), nightly ETL"
    }

app.include_router(score_router, prefix="")

app.include_router(search_router, prefix="")
