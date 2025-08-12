# api/routers/admin.py
import os
from fastapi import APIRouter, Header, HTTPException

router = APIRouter()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
FEATURE_STORE_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")

@router.post("/admin/refresh")
def refresh(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = {"ok": True, "steps": []}

    # 1) whatever you already do to refresh inspections/seed...
    try:
        # Example: from etl.nyc_inspections import nightly_refresh
        # nightly_refresh()
        result["steps"].append("inspections_seed_ok")
    except Exception as e:
        result["steps"].append(f"inspections_seed_failed: {e}")

    # 2) build rat_index.parquet (uses FEATURE_STORE_DIR and NYC_APP_TOKEN)
    try:
        # ensure the ETL writes to the same parquet dir used by the service
        os.environ["FEATURE_STORE_DIR"] = FEATURE_STORE_DIR
        from etl.rodent_index import build_rat_features
        build_rat_features()
        result["steps"].append("rat_index_ok")
    except Exception as e:
        # don't fail the whole refresh; just record the error
        result["steps"].append(f"rat_index_failed: {e}")

    return result
