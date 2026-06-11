import os
import pandas as pd
from datetime import date
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from typing import List

from api.models import ScoreRequest, ScoreResponse, ViolationProb, LastViolation, CamisSuggestion
from api.services.model_service import ModelService


router = APIRouter()

DEMO_SEED_FILE = os.getenv("DEMO_SEED_FILE", "./data/demo_seed.json")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/score_model.joblib")
model_service = ModelService(model_path=MODEL_PATH, demo_seed=DEMO_SEED_FILE)

RUNTIME_PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "/tmp/data/parquet")
BAKED_PARQUET_DIR = os.getenv("BAKED_FEATURE_DIR", "./data/parquet")
RAW_FILE_RUNTIME = os.path.join(RUNTIME_PARQUET_DIR, "inspections_raw.parquet")
RAW_FILE_BAKED = os.path.join(BAKED_PARQUET_DIR, "inspections_raw.parquet")

CODE_LABELS = {
    "04M": "Food not held at proper temp",
    "04L": "Evidence of mice or live mice",
    "04N": "Filth flies or food/refuse/sewage-associated flies",
    "06D": "Food contact surface not properly washed/rinsed/sanitized",
    "06C": "Food not protected from contamination",
    "08A": "Facility not vermin-proof",
    "10B": "Plumbing/sewage not properly drained",
    "10F": "Non-food contact surface not properly maintained",
    "10H": "Proper sanitization not provided for utensil washing",
    "02G": "Cold food held above 41°F (smoked fish, reduced oxygen)",
    "04A": "Food protection — improper handling",
}

def _parquet_path() -> str:
    p = RAW_FILE_RUNTIME if os.path.exists(RAW_FILE_RUNTIME) else RAW_FILE_BAKED
    if not os.path.exists(p):
        raise HTTPException(status_code=500, detail="No data parquet found. Run /admin/refresh or rebuild with data.")
    return p


@lru_cache(maxsize=256)
def _latest_visit_summary(camis: str):
    def _read_filtered(path: str, camis: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, filters=[("camis", "=", camis)])
        except Exception:
            df = pd.read_parquet(path)
            return df[df["camis"].astype(str) == str(camis)]

    def _extract_latlon(row: pd.Series) -> tuple:
        lat = lon = None
        for c in ["latitude", "Latitude", "lat", "LATITUDE"]:
            if c in row.index and pd.notna(row[c]):
                try: lat = float(row[c]); break
                except Exception: pass
        for c in ["longitude", "Longitude", "lon", "LONGITUDE"]:
            if c in row.index and pd.notna(row[c]):
                try: lon = float(row[c]); break
                except Exception: pass
        return lat, lon

    p = _parquet_path()
    df = _read_filtered(p, camis)
    if df.empty:
        return None

    if "inspection_date" in df.columns:
        df = df.sort_values("inspection_date")

    last = df.tail(1).iloc[0]

    last_date = (
        str(last["inspection_date"])[:10]
        if "inspection_date" in df.columns and pd.notna(last["inspection_date"])
        else None
    )

    try:
        last_score = int(last["score"]) if "score" in df.columns and pd.notna(last["score"]) else None
    except Exception:
        last_score = None

    lat, lon = _extract_latlon(last)

    # fill coords from alternate parquet if missing
    alt = RAW_FILE_BAKED if p == RAW_FILE_RUNTIME else RAW_FILE_RUNTIME
    if (lat is None or lon is None) and os.path.exists(alt):
        df2 = _read_filtered(alt, camis)
        if not df2.empty:
            if "inspection_date" in df2.columns:
                df2 = df2.sort_values("inspection_date")
            last2 = df2.tail(1).iloc[0]
            lat2, lon2 = _extract_latlon(last2)
            lat = lat if lat is not None else lat2
            lon = lon if lon is not None else lon2

    cuisine = (
        str(last["cuisine_description"]).strip()
        if "cuisine_description" in df.columns and pd.notna(last.get("cuisine_description"))
        else None
    )
    boro = (
        str(last["boro"]).strip()
        if "boro" in df.columns and pd.notna(last.get("boro"))
        else None
    )

    same_visit = df[df["inspection_date"] == last["inspection_date"]] if "inspection_date" in df.columns else df.tail(1)

    # Find grade by scanning all rows of the latest inspection — the grade column is
    # only populated on some rows and can differ within an inspection. Prefer any
    # explicit A/B/C present; fall back to inference from score.
    last_grade = None
    if "grade" in same_visit.columns:
        for v in same_visit["grade"].dropna():
            g = str(v).strip().upper()
            if g in ("A", "B", "C"):
                last_grade = g
                break
    if last_grade is None and last_score is not None:
        if last_score <= 13:
            last_grade = "A"
        elif last_score <= 27:
            last_grade = "B"
        else:
            last_grade = "C"

    labels_by_code: dict = {}
    if not same_visit.empty and "violation_code" in same_visit.columns and "violation_description" in same_visit.columns:
        tmp = same_visit.dropna(subset=["violation_code", "violation_description"]).copy()
        if not tmp.empty:
            labels_by_code = (
                tmp.groupby("violation_code")["violation_description"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .astype(str)
                .to_dict()
            )

    vio_counts: list = []
    if not same_visit.empty and "violation_code" in same_visit.columns:
        counts = same_visit["violation_code"].dropna().astype(str).value_counts().head(3)
        for code, cnt in counts.items():
            label = labels_by_code.get(code) or CODE_LABELS.get(code) or f"Violation {code}"
            vio_counts.append((code, int(cnt), label))

    # All violations from the last visit, deduplicated by code
    # critical first, then non-critical, each group sorted by code
    last_violations: list = []
    if not same_visit.empty and "violation_code" in same_visit.columns:
        tmp = same_visit.dropna(subset=["violation_code"]).copy()
        tmp["violation_code"] = tmp["violation_code"].astype(str)
        if "critical_flag" not in tmp.columns:
            tmp["critical_flag"] = None
        seen = set()
        rows = []
        for _, row in tmp.iterrows():
            code = row["violation_code"]
            if code in seen:
                continue
            seen.add(code)
            desc = labels_by_code.get(code) or CODE_LABELS.get(code) or f"Violation {code}"
            flag = str(row.get("critical_flag") or "").strip()
            is_critical = flag == "Critical"
            rows.append((code, desc, is_critical))
        rows.sort(key=lambda r: (0 if r[2] else 1, r[0]))
        last_violations = [{"code": c, "description": d, "critical": ic} for c, d, ic in rows]

    # Critical flag fraction for last visit
    critical_fraction: float = 0.0
    if not same_visit.empty and "critical_flag" in same_visit.columns:
        flags = same_visit["critical_flag"].dropna()
        if len(flags):
            critical_fraction = float((flags == "Critical").sum()) / len(flags)

    # Score history and grade history across distinct inspection dates
    score_history: list = []
    grade_history: list = []
    inspection_count = 1
    recurrence: dict = {}

    if "inspection_date" in df.columns:
        dated = df.dropna(subset=["inspection_date"]).sort_values("inspection_date")

        if "score" in df.columns:
            by_date = (
                dated.groupby("inspection_date")["score"]
                .apply(lambda s: _safe_int(s.dropna().iloc[0]) if not s.dropna().empty else None)
                .reset_index()
            )
            score_history = [
                (str(r["inspection_date"])[:10], r["score"])
                for _, r in by_date.iterrows()
                if r["score"] is not None
            ]
            inspection_count = len(by_date)

        if "grade" in df.columns:
            by_date_g = (
                dated[dated["grade"].isin(["A", "B", "C"])]
                .groupby("inspection_date")["grade"]
                .first()
                .reset_index()
            )
            grade_history = [
                (str(r["inspection_date"])[:10], str(r["grade"]))
                for _, r in by_date_g.iterrows()
            ]

        if "violation_code" in df.columns:
            recurrence = (
                df.dropna(subset=["violation_code", "inspection_date"])
                .groupby("violation_code")["inspection_date"]
                .nunique()
                .to_dict()
            )

    # Consecutive A grades at end of grade history
    consec_a = 0
    for _, g in reversed(grade_history):
        if g == "A":
            consec_a += 1
        else:
            break

    # Days since last inspection
    days_since_last = None
    if last_date:
        try:
            days_since_last = (date.today() - date.fromisoformat(last_date)).days
        except Exception:
            pass

    return {
        "last_date": last_date,
        "last_score": last_score,
        "last_grade": last_grade,
        "cuisine": cuisine,
        "boro": boro,
        "critical_fraction": critical_fraction,
        "consec_a": consec_a,
        "vio_counts": vio_counts,
        "last_violations": last_violations,
        "latitude": lat,
        "longitude": lon,
        "score_history": score_history,
        "grade_history": grade_history,
        "inspection_count": inspection_count,
        "days_since_last": days_since_last,
        "recurrence": recurrence,
    }


def _safe_int(val):
    try:
        return int(val)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _address_index():
    """Build a (building, street, zipcode) → [(camis, dba, max_date)] lookup. Cached after first call."""
    try:
        p = _parquet_path()
        df = pd.read_parquet(p, columns=["camis", "dba", "building", "street", "zipcode", "inspection_date"])
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
        df = df.dropna(subset=["building", "street", "zipcode", "inspection_date"])
        df["building"] = df["building"].astype(str).str.strip().str.upper()
        df["street"]   = df["street"].astype(str).str.strip().str.upper()
        df["zipcode"]  = df["zipcode"].astype(str).str.strip()
        agg = (
            df.groupby(["camis", "dba", "building", "street", "zipcode"])["inspection_date"]
            .max()
            .reset_index()
        )
        idx: dict = {}
        for r in agg.itertuples(index=False):
            key = (r.building, r.street, r.zipcode)
            idx.setdefault(key, []).append((str(r.camis), str(r.dba), str(r.inspection_date)[:10]))
        return idx
    except Exception:
        return {}


def _find_newer_camis(camis: str, s: dict) -> list:
    """Return other CAMISes at the same address that have more recent inspections."""
    STALE_DAYS = 730
    days_since = s.get("days_since_last")
    if days_since is None or days_since < STALE_DAYS:
        return []

    last_date = s.get("last_date")
    # Normalise address fields from the parquet row
    p = _parquet_path()
    try:
        df = pd.read_parquet(p, columns=["camis", "building", "street", "zipcode"],
                             filters=[("camis", "=", camis)])
        if df.empty:
            return []
        row = df.iloc[0]
        building = str(row.get("building", "") or "").strip().upper()
        street   = str(row.get("street",   "") or "").strip().upper()
        zipcode  = str(row.get("zipcode",  "") or "").strip()
        if not building or not street or not zipcode:
            return []
    except Exception:
        return []

    idx = _address_index()
    candidates = idx.get((building, street, zipcode), [])
    results = []
    for c_camis, c_name, c_date in candidates:
        if c_camis == camis:
            continue
        if last_date and c_date <= last_date:
            continue
        results.append(CamisSuggestion(camis=c_camis, name=c_name, last_inspection_date=c_date))
    results.sort(key=lambda x: x.last_inspection_date or "", reverse=True)
    return results[:3]




# Borough B/C risk adjustment derived from NYC inspection data (deviation from 7.8% mean)
BORO_RISK_DELTA = {
    "Queens": +0.03,
    "Bronx": +0.01,
    "Brooklyn": -0.01,
    "Staten Island": -0.01,
    "Manhattan": -0.02,
}


def _heuristic_from_summary(s) -> tuple:
    last_score = s["last_score"]
    last_grade = s["last_grade"]
    score_history = s.get("score_history", [])
    days_since = s.get("days_since_last")
    inspection_count = s.get("inspection_count", 1)
    recurrence = s.get("recurrence", {})
    boro = s.get("boro")
    critical_fraction = s.get("critical_fraction", 0.0)
    consec_a = s.get("consec_a", 0)

    # --- base prob from last score ---
    if last_score is not None:
        if last_score >= 28:
            prob_bc = 0.85
        elif last_score >= 21:
            prob_bc = 0.70
        elif last_score >= 14:
            prob_bc = 0.50
        elif last_score >= 8:
            prob_bc = 0.30
        else:
            prob_bc = 0.12
        predicted_points = last_score
        reasons = [f"Last points: {last_score}"]
    elif last_grade in {"B", "C"}:
        prob_bc, predicted_points, reasons = 0.55, 18, [f"Last grade: {last_grade}"]
    else:
        prob_bc, predicted_points, reasons = 0.20, 10, ["Limited history"]

    if last_grade and f"Last grade: {last_grade}" not in reasons:
        reasons.append(f"Last grade: {last_grade}")

    # --- score trend ---
    scores = [sc for _, sc in score_history if sc is not None]
    if len(scores) >= 2:
        recent, prior = scores[-1], scores[-2]
        delta = recent - prior
        if delta >= 10:
            prob_bc = min(0.95, prob_bc + 0.12)
            reasons.append(f"Score worsening ({prior}→{recent})")
        elif delta >= 5:
            prob_bc = min(0.95, prob_bc + 0.06)
            reasons.append(f"Score trending up ({prior}→{recent})")
        elif delta <= -10:
            prob_bc = max(0.05, prob_bc - 0.12)
            reasons.append(f"Score improving ({prior}→{recent})")
        elif delta <= -5:
            prob_bc = max(0.05, prob_bc - 0.06)
            reasons.append(f"Score trending down ({prior}→{recent})")

    # --- consecutive clean A grades ---
    if consec_a >= 3:
        prob_bc = max(0.04, prob_bc - 0.08)
        reasons.append(f"Consistent A history ({consec_a} consecutive)")
    elif consec_a == 2:
        prob_bc = max(0.04, prob_bc - 0.04)
        reasons.append("Consistent A history (2 consecutive)")

    # --- borough risk ---
    if boro and boro in BORO_RISK_DELTA:
        delta = BORO_RISK_DELTA[boro]
        prob_bc = min(0.95, max(0.04, prob_bc + delta))
        if abs(delta) >= 0.02:
            direction = "higher" if delta > 0 else "lower"
            reasons.append(f"{boro} restaurants trend {direction}-risk")

    # --- critical violations ---
    if critical_fraction == 0:
        prob_bc = max(0.04, prob_bc - 0.04)
        reasons.append("No critical violations in last inspection")
    elif critical_fraction > 0.67:
        prob_bc = min(0.95, prob_bc + 0.04)
        reasons.append("Majority of violations were critical")
    elif critical_fraction > 0.33:
        prob_bc = min(0.95, prob_bc + 0.01)

    # --- days since last inspection ---
    if days_since is not None:
        if days_since > 540:
            prob_bc = 0.4 * prob_bc + 0.6 * 0.35
            reasons.append(f"Not inspected in {days_since} days")
        elif days_since > 365:
            prob_bc = 0.7 * prob_bc + 0.3 * 0.35
            reasons.append(f"Not inspected in {days_since} days")

    # --- inspection count: flag low confidence ---
    if inspection_count == 1:
        reasons.append("Limited history (1 inspection on record)")
    elif inspection_count == 2:
        reasons.append("Early history (2 inspections on record)")

    # --- violation probs with recurrence boost ---
    top_vios: List[ViolationProb] = []
    total = sum(cnt for _, cnt, _ in s["vio_counts"]) or 1
    for code, cnt, label in s["vio_counts"]:
        base_prob = float(cnt) / float(total)
        recur_count = recurrence.get(code, 1)
        if recur_count >= 4:
            base_prob = min(0.99, base_prob * 1.6)
        elif recur_count >= 3:
            base_prob = min(0.99, base_prob * 1.4)
        elif recur_count >= 2:
            base_prob = min(0.99, base_prob * 1.2)
        top_vios.append(ViolationProb(code=code, probability=base_prob, label=label))

    return prob_bc, predicted_points, reasons, top_vios


@router.post("/score", response_model=ScoreResponse, summary="Get risk score for a restaurant")
def score(req: ScoreRequest):
    """
    Returns a risk prediction for the given NYC restaurant (identified by CAMIS).

    **Scoring approach:** heuristic-based (not ML-trained). The `prob_bc` field is the
    estimated probability of receiving a B or C grade on the next inspection, derived from:
    - Last inspection score and grade
    - Score trend across inspection history
    - Consecutive clean A-grade streak
    - Borough-level risk adjustment
    - Critical violation fraction at last visit
    - Local rodent pressure index (rat_index)

    For the 25 pre-seeded demo restaurants the response is returned from an in-memory cache
    (~1 ms). All other restaurants are scored live from the inspection parquet (~50–200 ms).
    """
    camis = str(req.camis)

    def attach_rat(payload: dict) -> dict:
        rf = getattr(model_service, "rat_features", {}).get(camis)
        if rf:
            payload.update({
                "rat_index":            rf.get("rat_index"),
                "pest_index":           rf.get("pest_index"),
                "rat311_cnt_180d_k1":   rf.get("rat311_cnt_180d_k1"),
                "mouse311_cnt_180d_k1": rf.get("mouse311_cnt_180d_k1"),
                "pest311_cnt_180d_k1":  rf.get("pest311_cnt_180d_k1"),
                "ratinsp_fail_365d_k1": rf.get("ratinsp_fail_365d_k1"),
            })
        return payload

    # Load inspection history — needed for all paths
    s = _latest_visit_summary(camis)
    if not s:
        raise HTTPException(status_code=404, detail="CAMIS not found")

    # Rat features for this restaurant
    rf = getattr(model_service, "rat_features", {}).get(camis, {})

    # Heuristic gives reasons, predicted_points, violation probs
    h_prob_bc, predicted_points, reasons, top_vios = _heuristic_from_summary(s)

    # ML scores are pre-computed at ETL time — simple dict lookup, no sklearn at runtime
    ml_prob = model_service.ml_scores.get(camis)
    if ml_prob is not None:
        prob_bc       = float(ml_prob)
        model_version = "ml-v1"
    else:
        prob_bc       = h_prob_bc
        model_version = "heuristic-v3"

    payload = {
        "camis": camis,
        "prob_bc": float(prob_bc),
        "predicted_points": float(predicted_points),
        "top_reasons": reasons,
        "top_violation_probs": top_vios,
        "model_version": model_version,
        "data_version": "runtime",
        "last_inspection_date": s["last_date"],
        "last_points": s["last_score"],
        "last_grade": s["last_grade"],
        "latitude": s.get("latitude"),
        "longitude": s.get("longitude"),
        "score_history": s.get("score_history", []),
        "last_violations": s.get("last_violations", []),
        "suggested_camis": _find_newer_camis(camis, s),
    }
    payload = attach_rat(payload)
    return ScoreResponse(**payload)
