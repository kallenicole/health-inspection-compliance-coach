import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

# Prefer runtime parquet in /tmp (Cloud Run writable), fallback to baked parquet
RUNTIME_PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "/tmp/data/parquet")
BAKED_PARQUET_DIR = os.getenv("BAKED_FEATURE_DIR", "./data/parquet")
RAW_FILE_RUNTIME = os.path.join(RUNTIME_PARQUET_DIR, "inspections_raw.parquet")
RAW_FILE_BAKED = os.path.join(BAKED_PARQUET_DIR, "inspections_raw.parquet")

@router.get("/search")
def search(name: str = Query(..., min_length=2)):
    # Use runtime-refreshed parquet if available; else baked parquet from the image
    raw_file = RAW_FILE_RUNTIME if os.path.exists(RAW_FILE_RUNTIME) else RAW_FILE_BAKED
    if not os.path.exists(raw_file):
        raise HTTPException(status_code=500, detail="No data parquet found. Run ETL locally then rebuild, or call /admin/refresh in prod.")
    df = pd.read_parquet(raw_file, columns=["camis","dba","boro","building","street","zipcode","inspection_date"])
    df = df.dropna(subset=["camis","dba"]).copy()
    df["dba_u"] = df["dba"].astype(str).str.upper()
    q = name.upper()
    hits = (df[df["dba_u"].str.contains(q, na=False)]
            .sort_values(["camis","inspection_date"])
            .groupby("camis", as_index=False).tail(1)
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
