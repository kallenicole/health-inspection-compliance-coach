# api/routers/score.py
import os
from functools import lru_cache
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.models import ScoreRequest, ScoreResponse, ViolationProb
from api.services.model_service import ModelService

router = APIRouter()

# Single shared service instance (admin router will reload this one)
DEMO_SEED_FILE = os.getenv("DEMO_SEED_FILE", "./data/demo_seed.json")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/dummy.joblib")
model_service = ModelService(model_path=MODEL_PATH, demo_seed=DEMO_SEED_FILE)

# Parquet locations: prefer runtime (/tmp on Cloud Run), fallback to baked in image
FEATURE_STORE_DIR = os.getenv("FEATURE_STORE_DIR", "/tmp")
BAKED_FEATURE_DIR = os.getenv("BAKED_FEATURE_DIR", "/app/data/parquet")
RAW_FILE_RUNTIME = os.path.join(FEATURE_STORE_DIR, "inspections_raw.parquet")
RAW_FILE_BAKED = os.path.join(BAKED_FEATURE_DIR, "inspections_raw.parquet")

CODE_LABELS: Dict[str, str] = {
    "04M": "Food not held at proper temp",
    "04L": "Evidence of mice",
    "10F": "Personal cleanliness",
    "06C": "Food not protected from contamination",
    "20-06": "Current letter grade or Grade Pending card not posted",
}

def _parquet_path() -> str:
    p = RAW_FILE_RUNTIME if os.path.exists(RAW_FILE_RUNTIME) else RAW_FILE_BAKED
    if not os.path.exists(p):
        raise HTTPException(status_code=500, detail="No data parquet found. Run /admin/refresh or rebuild with data.")
    return p

@lru_cache(maxsize=1024)
def _latest_visit_summary(camis: str) -> Optional[Dict[str, Any]]:
    p = _parquet_path()
    camis_s = str(camis)

    # Try predicate pushdown first; if engine can't, fallback to filter in pandas
    try:
        df = pd.read_parquet(p, filters=[("camis", "=", camis_s)])
    except Exception:
        df = pd.read_parquet(p)
        df["camis"] = df["camis"].astype(str)
        df = df[df["camis"] == camis_s]

    if df.empty:
        return None

    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
        df = df.sort_values("inspection_date")
    last = df.tail(1).iloc[0]

    last_date = str(last["inspection_date"])[:10] if "inspection_date" in df.columns and pd.notna(last["inspection_date"]) else None

    try:
        last_score = int(last["score"]) if "score" in df.columns and pd.notna(last["score"]) else None
    except Exception:
        last_score = None

    last_grade = str(last["grade"]).strip().upper() if "grade" in df.columns and pd.notna(last["grade"]) else None

    same_visit = df[df["inspection_date"] == last["inspection_date"]] if "inspection_date" in df.columns else df.tail(1)

    # Prefer violation_description from dataset for labels
    labels_by_code: Dict[str, str] = {}
    if not same_visit.empty and {"violation_code", "violation_description"} <= set(same_visit.columns):
        tmp = same_visit.dropna(subset=["violation_code", "violation_description"]).copy()
        if not tmp.empty:
            labels_by_code = (
                tmp.groupby("violation_code")["violation_description"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .astype(str)
                .to_dict()
            )

    vio_counts: List[tuple] = []
    if not same_visit.empty and "violation_code" in same_visit.columns:
        counts = same_visit["violation_code"].dropna().astype(str).value_counts().head(3)
        for code, cnt in counts.items():
            label = labels_by_code.get(code) or CODE_LABELS.get(code) or f"Violation {code}"
            vio_counts.append((code, int(cnt), label))

    return {
        "last_date": last_date,
        "last_score": last_score,
        "last_grade": last_grade,
        "vio_counts": vio_counts,  # list of (code, count, label)
    }

def _heuristic_from_summary(s: Dict[str, Any]):
    last_score = s["last_score"]
    last_grade = s["last_grade"]

    if last_score is not None:
        prob_bc = 0.75 if last_score >= 21 else 0.55 if last_score >= 14 else 0.35 if last_score >= 8 else 0.15
        predicted_points = last_score
        reasons = [f"Last points: {last_score}"]
    elif last_grade in {"B", "C"}:
        prob_bc, predicted_points, reasons = 0.55, 18.0, [f"Last grade: {last_grade}"]
    else:
        prob_bc, predicted_points, reasons = 0.20, 10.0, ["Limited history"]
    if last_grade and f"Last grade: {last_grade}" not in reasons:
        reasons.append(f"Last grade: {last_grade}")

    # Convert codes to probability distribution for display
    top_vios: List[ViolationProb] = []
    total = sum(cnt for _, cnt, _ in s["vio_counts"]) or 1
    for code, cnt, label in s["vio_counts"]:
        top_vios.append(ViolationProb(code=code, probability=float(cnt) / float(total), label=label))

    return prob_bc, float(predicted_points), reasons, top_vios

@router.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    camis = str(req.camis)

    # Seeded fast-path
    try:
        payload: Dict[str, Any] = model_service.score_camis(camis)  # includes rat heuristic if available
        s = _latest_visit_summary(camis)
        if s:
            payload.update({
                "last_inspection_date": s["last_date"],
                "last_points": s["last_score"],
                "last_grade": s["last_grade"],
            })
        # ensure rat keys exist even if missing
        payload.setdefault("rat_index", None)
        payload.setdefault("rat311_cnt_180d_k1", None)
        payload.setdefault("ratinsp_fail_365d_k1", None)
        return ScoreResponse(**payload)

    except KeyError:
        # Fallback heuristic when not in demo seed
        s = _latest_visit_summary(camis)
        if not s:
            raise HTTPException(status_code=404, detail="CAMIS not found")

        prob_bc, predicted_points, reasons, top_vios = _heuristic_from_summary(s)
        payload: Dict[str, Any] = {
            "camis": camis,
            "prob_bc": float(prob_bc),
            "predicted_points": float(predicted_points),
            "top_reasons": reasons,
            "top_violation_probs": top_vios,
            "model_version": "heuristic-fallback-0.1",
            "data_version": "runtime",
            "last_inspection_date": s["last_date"],
            "last_points": s["last_score"],
            "last_grade": s["last_grade"],
        }

        # Attach rat features + tiny heuristic bump to mice if we have them
        payload = model_service._apply_rat_heuristics(camis, payload)  # uses in-memory map

        # always include rat keys so frontend never sees missing properties
        payload.setdefault("rat_index", None)
        payload.setdefault("rat311_cnt_180d_k1", None)
        payload.setdefault("ratinsp_fail_365d_k1", None)

        return ScoreResponse(**payload)
