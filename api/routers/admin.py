# api/routers/admin.py
import os
import threading
from fastapi import APIRouter, Header, HTTPException, Query

from api.routers.score import model_service

router = APIRouter()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
FEATURE_STORE_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")

_refresh_status: dict = {"running": False, "last": None}


def _run_refresh():
    steps: list[str] = []
    try:
        from etl.nyc_inspections_etl import fetch_to_parquet
        total = fetch_to_parquet()
        if total == 0:
            steps.append("inspections_empty_response")
        else:
            steps.append(f"inspections_ok:{total}_rows")
            from api.routers.score import _latest_visit_summary
            _latest_visit_summary.cache_clear()
    except Exception as e:
        steps.append(f"inspections_failed:{e}")

    try:
        os.environ["FEATURE_STORE_DIR"] = FEATURE_STORE_DIR
        from etl.rodent_index import build_rat_features
        build_rat_features()
        steps.append("rat_index_ok")
    except Exception as e:
        steps.append(f"rat_index_failed:{e}")

    try:
        count = model_service.reload_rat_features()
        steps.append(f"rat_features_reloaded:{count}")
    except Exception as e:
        steps.append(f"rat_reload_failed:{e}")

    try:
        from api.routers.insights import _compute_insights
        _compute_insights.cache_clear()
        steps.append("insights_cache_cleared")
    except Exception as e:
        steps.append(f"insights_cache_clear_failed:{e}")

    try:
        from api.routers.score import _address_index, _latest_visit_summary
        _address_index.cache_clear()
        _latest_visit_summary.cache_clear()
        steps.append("score_caches_cleared")
    except Exception as e:
        steps.append(f"score_cache_clear_failed:{e}")

    try:
        from api.routers.search import _load_search_index
        _load_search_index.cache_clear()
        steps.append("search_cache_cleared")
    except Exception as e:
        steps.append(f"search_cache_clear_failed:{e}")

    try:
        from api.routers.neighborhood import _load_neighborhood_index
        _load_neighborhood_index.cache_clear()
        steps.append("neighborhood_cache_cleared")
    except Exception as e:
        steps.append(f"neighborhood_cache_clear_failed:{e}")

    _refresh_status["running"] = False
    _refresh_status["last"] = steps


@router.post("/admin/refresh", summary="Trigger a full data refresh (admin only)")
def refresh(x_admin_token: str = Header(default="", description="Admin token — must match the ADMIN_TOKEN environment variable")):
    """
    Triggers a background refresh of the NYC inspection dataset and rat pressure index.

    Fetches all ~296k inspection records from NYC Open Data, rebuilds the rat pressure index
    from 311 rodent complaints and DOHMH rat inspection data, then reloads the in-memory
    feature store so `/score` immediately uses the latest data.

    Requires the `X-Admin-Token` header. Returns immediately — the refresh runs in a background
    thread. Poll `/admin/refresh/status` to check progress. Called nightly at 03:00 UTC by
    Cloud Scheduler.
    """
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _refresh_status["running"]:
        return {"ok": False, "message": "Refresh already in progress"}

    _refresh_status["running"] = True
    threading.Thread(target=_run_refresh, daemon=True).start()
    return {"ok": True, "message": "Refresh started in background — call /admin/refresh/status to check progress"}


@router.get("/admin/refresh/status", summary="Check background refresh status (admin only)")
def refresh_status(x_admin_token: str = Header(default="", description="Admin token")):
    """Returns whether a refresh is currently running and the result of the last completed refresh."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "running": _refresh_status["running"],
        "last_result": _refresh_status["last"],
    }


@router.get("/admin/ratpeek", summary="Debug: rat features for a CAMIS")
def ratpeek(camis: str = Query(..., description="NYC CAMIS identifier")):
    """Debug helper: check whether a given CAMIS has rat pressure features loaded in-memory."""
    v = model_service.rat_features.get(str(camis))
    return {
        "feature_dir": os.getenv("FEATURE_STORE_DIR", "./data/parquet"),
        "has": str(camis) in model_service.rat_features,
        "value": v,
    }
