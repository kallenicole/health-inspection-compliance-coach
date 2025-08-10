import os, subprocess
from fastapi import APIRouter, Header, HTTPException

router = APIRouter()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Cloud Run is read-only except /tmp; set runtime defaults there
os.environ.setdefault("FEATURE_STORE_DIR", "/tmp/data/parquet")
os.environ.setdefault("DEMO_SEED_FILE", "/tmp/data/demo_seed.json")

@router.post("/admin/refresh")
def admin_refresh(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Regenerate parquet and demo seed at runtime
        subprocess.check_call(["python", "etl/nyc_inspections_etl.py"])
        subprocess.check_call(["python", "etl/feature_engineering.py"])
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ETL/seed failed: {e}")

    return {"ok": True, "message": "Refreshed inspections & demo seed in /tmp"}
