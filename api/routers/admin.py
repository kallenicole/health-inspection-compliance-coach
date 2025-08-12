# api/routers/admin.py
import os
from fastapi import APIRouter, Header, HTTPException

# Reuse the same ModelService instance that /score uses
from api.routers.score import model_service

router = APIRouter()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
FEATURE_STORE_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
BAKED_FEATURE_DIR = os.getenv("BAKED_FEATURE_DIR", "/app/data/parquet")


@router.post("/admin/refresh")
def refresh(x_admin_token: str = Header(default="")):
    """
    Secure refresh:
      1) (placeholder) refresh inspections/seed if you have that wired.
      2) Build rat_index.parquet into FEATURE_STORE_DIR (e.g., /tmp on Cloud Run).
      3) Reload features in-memory so /score sees them immediately.
    """
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    steps: list[str] = []

    # 1) Your inspections/seed step (keep as-is or plug in your ETL)
    try:
        # e.g., nightly_refresh_inspections()
        steps.append("inspections_seed_ok")
    except Exception as e:
        steps.append(f"inspections_seed_failed: {e}")

    # 2) Build rat features (writes to FEATURE_STORE_DIR)
    try:
        os.environ["FEATURE_STORE_DIR"] = FEATURE_STORE_DIR  # ensure ETL writes where the service reads
        from etl.rodent_index import build_rat_features
        build_rat_features()
        steps.append("rat_index_ok")
    except Exception as e:
        steps.append(f"rat_index_failed: {e}")

    # 3) Reload in-memory map so /score immediately uses the new parquet
    try:
        reloaded = model_service.reload_rat_features()
        steps.append(f"rat_features_reloaded:{reloaded}")
    except Exception as e:
        steps.append(f"rat_reload_failed: {e}")

    return {"ok": True, "steps": steps}


@router.get("/admin/ratpeek")
def ratpeek(camis: str):
    """
    Debug helper: check if a CAMIS has rat features loaded in-memory.
    """
    v = model_service.rat_features.get(str(camis))
    return {
        "feature_dir": os.getenv("FEATURE_STORE_DIR", "./data/parquet"),
        "has": str(camis) in model_service.rat_features,
        "value": v,
    }


@router.get("/admin/rawpeek")
def rawpeek(camis: str):
    """
    Check if CAMIS exists (and has lat/lon) in the raw inspections parquet
    used by the service (prefers FEATURE_STORE_DIR, falls back to baked).
    """
    import pandas as pd, math

    raw_tmp = os.path.join(FEATURE_STORE_DIR, "inspections_raw.parquet")
    raw_baked = os.path.join(BAKED_FEATURE_DIR, "inspections_raw.parquet")
    path = raw_tmp if os.path.exists(raw_tmp) else raw_baked

    if not os.path.exists(path):
        return {"path": path, "present": False, "reason": "parquet missing"}

    df = pd.read_parquet(path, columns=["camis", "latitude", "longitude"]).copy()
    df["camis"] = df["camis"].astype(str)
    row = df[df["camis"] == str(camis)]
    if row.empty:
        return {"path": path, "present": False}

    r = row.iloc[0]

    def num(x):
        try:
            f = float(x)
            return None if math.isnan(f) else f
        except Exception:
            return None

    return {"path": path, "present": True, "lat": num(r.get("latitude")), "lon": num(r.get("longitude"))}

