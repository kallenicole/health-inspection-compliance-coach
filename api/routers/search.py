import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
PARQUET = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
RAW_FILE = os.path.join(PARQUET, "inspections_raw.parquet")

@router.get("/search")
def search(name: str = Query(..., min_length=2)):
    # Ensure we have data from your ETL step
    if not os.path.exists(RAW_FILE):
        raise HTTPException(status_code=500, detail="No data file. Run ETL first (make etl).")
    # Load only columns we need (faster)
    df = pd.read_parquet(RAW_FILE, columns=["camis","dba","boro","building","street","zipcode","inspection_date"])
    # Keep the latest record per CAMIS
    df = df.dropna(subset=["camis","dba"]).copy()
    df["dba_u"] = df["dba"].astype(str).str.upper()
    q = name.upper()
    hits = (df[df["dba_u"].str.contains(q, na=False)]
            .sort_values(["camis","inspection_date"])
            .groupby("camis", as_index=False).tail(1)  # last row per CAMIS
            .sort_values("dba_u")
            .head(25))
    return [
        {
            "camis": str(r.camis),
            "name": r.dba,
            "address": " ".join([str(r.building or ""), str(r.street or ""), str(r.zipcode or "")]).strip(),
            "boro": r.boro
        }
        for r in hits.itertuples(index=False)
    ]
