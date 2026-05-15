import os
from datetime import date
from functools import lru_cache
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

RUNTIME_PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "/tmp/data/parquet")
BAKED_PARQUET_DIR = os.getenv("BAKED_FEATURE_DIR", "./data/parquet")
RAW_FILE_RUNTIME = os.path.join(RUNTIME_PARQUET_DIR, "inspections_raw.parquet")
RAW_FILE_BAKED = os.path.join(BAKED_PARQUET_DIR, "inspections_raw.parquet")

COLS = ["camis", "dba", "boro", "building", "street", "zipcode",
        "cuisine_description", "inspection_date", "score", "grade",
        "latitude", "longitude"]


@lru_cache(maxsize=1)
def _load_neighborhood_index() -> pd.DataFrame:
    """
    Load inspections parquet, find each restaurant's latest inspection, and
    pre-aggregate to ONE row per restaurant with proper grade detection.
    Cached until manually cleared on refresh. ~26k rows vs. ~150k raw rows.
    """
    p = RAW_FILE_RUNTIME if os.path.exists(RAW_FILE_RUNTIME) else RAW_FILE_BAKED
    if not os.path.exists(p):
        raise HTTPException(status_code=500, detail="No data parquet found.")
    df = pd.read_parquet(p, columns=COLS)
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["zipcode"] = df["zipcode"].astype(str).str.strip()

    # Latest inspection date per restaurant
    latest_dates = df.dropna(subset=["inspection_date"]).groupby("camis")["inspection_date"].max()
    df = df.join(latest_dates.rename("latest_date"), on="camis")
    latest = df[df["inspection_date"] == df["latest_date"]]

    # Aggregate to ONE row per restaurant; scan all violation rows for a valid grade
    def _first_valid_grade(s: pd.Series) -> "str | None":
        for v in s.dropna():
            g = str(v).strip().upper()
            if g in ("A", "B", "C"):
                return g
        return None

    agg = latest.groupby("camis", as_index=False).agg(
        dba=("dba", "first"),
        boro=("boro", "first"),
        building=("building", "first"),
        street=("street", "first"),
        zipcode=("zipcode", "first"),
        cuisine_description=("cuisine_description", "first"),
        inspection_date=("inspection_date", "first"),
        score=("score", "first"),
        grade=("grade", _first_valid_grade),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )

    def _infer_grade(row) -> "str | None":
        g = row["grade"]
        if g in ("A", "B", "C"):
            return g
        s = row["score"]
        if s is None or pd.isna(s):
            return None
        if s <= 13: return "A"
        if s <= 27: return "B"
        return "C"

    agg["grade_display"] = agg.apply(_infer_grade, axis=1)
    return agg


@router.get("/neighborhood", summary="List restaurants in a zip code by inspection risk")
def neighborhood(
    zip: str = Query(..., min_length=5, max_length=5, description="5-digit NYC zip code"),
    limit: int = Query(default=30, description="Maximum number of restaurants to return (default 30)"),
):
    """
    Returns restaurants in the given NYC zip code, ranked by most recent inspection score
    (highest / most risky first). Each result includes the restaurant's last grade, score,
    inspection date, and days since last inspection.

    Grades are inferred from the point score when not explicitly recorded (0–13 = A, 14–27 = B, 28+ = C).
    Only the most recent inspection per restaurant is returned.
    """
    idx = _load_neighborhood_index()
    df = idx[idx["zipcode"] == zip.strip()]
    if df.empty:
        return []

    df = df.sort_values("score", ascending=False, na_position="last").head(limit)

    today = date.today()
    results = []
    for r in df.itertuples(index=False):
        last_date = str(r.inspection_date)[:10] if pd.notna(r.inspection_date) else None
        days_since = None
        if last_date:
            try:
                days_since = (today - date.fromisoformat(last_date)).days
            except Exception:
                pass
        score_val = int(r.score) if pd.notna(r.score) else None
        try:
            lat = float(r.latitude) if pd.notna(r.latitude) else None
            lon = float(r.longitude) if pd.notna(r.longitude) else None
        except (TypeError, ValueError):
            lat = lon = None
        results.append({
            "camis": str(r.camis),
            "name": str(r.dba),
            "address": " ".join(filter(None, [str(r.building or "").strip(), str(r.street or "").strip(), zip])),
            "boro": str(r.boro or ""),
            "cuisine": str(r.cuisine_description or ""),
            "last_grade": r.grade_display,
            "last_score": score_val,
            "last_date": last_date,
            "days_since": days_since,
            "latitude": lat,
            "longitude": lon,
        })
    return results
