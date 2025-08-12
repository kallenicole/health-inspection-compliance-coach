import os
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from typing import List

from api.models import ScoreRequest, ScoreResponse, ViolationProb
from api.services.model_service import ModelService


router = APIRouter()
svc = ModelService(model_path="", demo_seed="./data/demo_seed.json")

router = APIRouter()

DEMO_SEED_FILE = os.getenv("DEMO_SEED_FILE", "./data/demo_seed.json")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/dummy.joblib")
model_service = ModelService(model_path=MODEL_PATH, demo_seed=DEMO_SEED_FILE)

RUNTIME_PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "/tmp/data/parquet")
BAKED_PARQUET_DIR = os.getenv("BAKED_FEATURE_DIR", "./data/parquet")
RAW_FILE_RUNTIME = os.path.join(RUNTIME_PARQUET_DIR, "inspections_raw.parquet")
RAW_FILE_BAKED = os.path.join(BAKED_PARQUET_DIR, "inspections_raw.parquet")

CODE_LABELS = {
    "04M": "Food not held at proper temp",
    "04L": "Evidence of mice",
    "10F": "Personal cleanliness",
}

def _parquet_path() -> str:
    p = RAW_FILE_RUNTIME if os.path.exists(RAW_FILE_RUNTIME) else RAW_FILE_BAKED
    if not os.path.exists(p):
        raise HTTPException(status_code=500, detail="No data parquet found. Run /admin/refresh or rebuild with data.")
    return p

@lru_cache(maxsize=1024)
def _latest_visit_summary(camis: str):
    p = _parquet_path()
    try:
        df = pd.read_parquet(p, filters=[("camis", "=", camis)])
    except Exception:
        df = pd.read_parquet(p)
        df = df[df["camis"].astype(str) == str(camis)]
    if df.empty:
        return None

    if "inspection_date" in df.columns:
        df = df.sort_values("inspection_date")
    last = df.tail(1).iloc[0]

    # last date
    last_date = str(last["inspection_date"])[:10] if "inspection_date" in df.columns and pd.notna(last["inspection_date"]) else None

    # last score/grade
    try:
        last_score = int(last["score"]) if "score" in df.columns and pd.notna(last["score"]) else None
    except Exception:
        last_score = None
    last_grade = str(last["grade"]).strip().upper() if "grade" in df.columns and pd.notna(last["grade"]) else None

    # same visit rows
    same_visit = df[df["inspection_date"] == last["inspection_date"]] if "inspection_date" in df.columns else df.tail(1)

    # build labels per code using violation_description when present
    labels_by_code = {}
    if not same_visit.empty and "violation_code" in same_visit.columns and "violation_description" in same_visit.columns:
        tmp = same_visit.dropna(subset=["violation_code", "violation_description"]).copy()
        if not tmp.empty:
            # pick the most common description per code
            labels_by_code = (
                tmp.groupby("violation_code")["violation_description"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .astype(str)
                .to_dict()
            )

    # top codes from that visit
    vio_counts = []
    if not same_visit.empty and "violation_code" in same_visit.columns:
        counts = same_visit["violation_code"].dropna().astype(str).value_counts().head(3)
        for code, cnt in counts.items():
            # prefer dataset description, then our small dict, else generic
            label = labels_by_code.get(code) or CODE_LABELS.get(code) or f"Violation {code}"
            vio_counts.append((code, int(cnt), label))

    return {
        "last_date": last_date,
        "last_score": last_score,
        "last_grade": last_grade,
        "vio_counts": vio_counts,  # list of (code, count, label)
    }

def _heuristic_from_summary(s):
    last_score = s["last_score"]
    last_grade = s["last_grade"]

    if last_score is not None:
        prob_bc = 0.75 if last_score >= 21 else 0.55 if last_score >= 14 else 0.35 if last_score >= 8 else 0.15
        predicted_points = last_score
        reasons = [f"Last points: {last_score}"]
    elif last_grade in {"B", "C"}:
        prob_bc, predicted_points, reasons = 0.55, 18, [f"Last grade: {last_grade}"]
    else:
        prob_bc, predicted_points, reasons = 0.20, 10, ["Limited history"]
    if last_grade and f"Last grade: {last_grade}" not in reasons:
        reasons.append(f"Last grade: {last_grade}")

    # convert counts to probabilities with labels
    top_vios: List[ViolationProb] = []
    total = sum(cnt for _, cnt, _ in s["vio_counts"]) or 1
    for code, cnt, label in s["vio_counts"]:
        top_vios.append(ViolationProb(code=code, probability=float(cnt)/float(total), label=label))

    return prob_bc, predicted_points, reasons, top_vios


@router.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    camis = str(req.camis)

    def attach_rat(payload: dict) -> dict:
        # pull preloaded features from the ModelService
        rf = getattr(model_service, "rat_features", {}).get(camis)
        if rf:
            payload.update({
                "rat_index": rf.get("rat_index"),
                "rat311_cnt_180d_k1": rf.get("rat311_cnt_180d_k1"),
                "ratinsp_fail_365d_k1": rf.get("ratinsp_fail_365d_k1"),
            })
        return payload

    # Try seeded (demo) path
    try:
        payload = model_service.score_camis(camis)  # already may include rat features + heuristic bumps
        s = _latest_visit_summary(camis)
        if s:
            payload.update({
                "last_inspection_date": s["last_date"],
                "last_points": s["last_score"],
                "last_grade": s["last_grade"],
            })
        payload = attach_rat(payload)  # safe even if already present
        return ScoreResponse(**payload)

    except KeyError:
        # Heuristic fallback
        s = _latest_visit_summary(camis)
        if not s:
            raise HTTPException(status_code=404, detail="CAMIS not found")

        prob_bc, predicted_points, reasons, top_vios = _heuristic_from_summary(s)
        payload = {
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
        payload = attach_rat(payload)  # add rat features in fallback too
        return ScoreResponse(**payload)

