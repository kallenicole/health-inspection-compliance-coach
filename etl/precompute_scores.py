"""
Pre-compute ML prob_bc scores for all restaurants and write to ml_scores.parquet.

Run after ETL so the API can serve predictions from a fast parquet lookup
instead of loading scikit-learn at runtime (saves ~80 MB of startup RSS).

Usage:
    python -m etl.precompute_scores
"""
import gc
import os

import joblib
import numpy as np
import pandas as pd

from etl.train_model import (
    ALL_FEATURES, CYCLE_TYPES,
    _cuisine_group, _boro_clean, _infer_grade,
)

PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
MODEL_PATH  = os.getenv("MODEL_PATH",        "./models/score_model.joblib")
RAW_FILE    = os.path.join(PARQUET_DIR, "inspections_raw.parquet")
RAT_FILE    = os.path.join(PARQUET_DIR, "rat_index.parquet")
OUT_FILE    = os.path.join(PARQUET_DIR, "ml_scores.parquet")


def compute_inference_features(raw: pd.DataFrame, rat) -> pd.DataFrame:
    """
    Build one feature row per restaurant using their LATEST inspection,
    to predict the NEXT inspection outcome.  Mirrors the training feature
    construction in train_model.build_features(), but uses the most recent
    inspection rather than consecutive pairs.
    """
    raw = raw.copy()
    raw["inspection_date"] = pd.to_datetime(raw["inspection_date"], errors="coerce")
    raw["score"]           = pd.to_numeric(raw["score"], errors="coerce")
    raw["camis"]           = raw["camis"].astype(str)
    raw["violation_code"]  = raw.get("violation_code", pd.Series(dtype=str)).fillna("").astype(str)
    raw["critical_flag"]   = raw.get("critical_flag",  pd.Series(dtype=str)).astype(str)

    if "inspection_type" in raw.columns:
        raw = raw[raw["inspection_type"].isin(CYCLE_TYPES)].copy()

    raw["is_critical"]    = raw["critical_flag"].str.strip() == "Critical"
    raw["is_mice_vio"]    = raw["violation_code"] == "04L"
    raw["is_temp_vio"]    = raw["violation_code"] == "04M"
    raw["is_vermin_vio"]  = raw["violation_code"] == "08A"

    def first_valid_grade(s):
        for v in s.dropna():
            g = str(v).strip().upper()
            if g in ("A", "B", "C"):
                return g
        return None

    inspections = (
        raw.groupby(["camis", "inspection_date"], sort=False)
        .agg(
            score          = ("score",                "first"),
            grade_raw      = ("grade",                first_valid_grade),
            n_violations   = ("violation_code",       lambda x: (x != "").sum()),
            n_critical     = ("is_critical",          "sum"),
            has_mice_vio   = ("is_mice_vio",          "any"),
            has_temp_vio   = ("is_temp_vio",          "any"),
            has_vermin_vio = ("is_vermin_vio",        "any"),
            cuisine        = ("cuisine_description",  "first"),
            boro           = ("boro",                 "first"),
        )
        .reset_index()
    )
    inspections["critical_fraction"] = (
        inspections["n_critical"] / inspections["n_violations"].replace(0, 1)
    ).clip(0, 1)
    inspections["grade"] = inspections.apply(
        lambda r: _infer_grade(r["grade_raw"], r["score"]), axis=1
    )
    inspections = inspections.sort_values(["camis", "inspection_date"])

    # Violation recurrence (max # inspections any single code appeared in)
    recur_df = (
        raw[raw["violation_code"] != ""]
        .dropna(subset=["violation_code", "inspection_date"])
        .groupby(["camis", "violation_code"])["inspection_date"]
        .nunique()
        .reset_index(name="n_insp")
    )
    max_recur = recur_df.groupby("camis")["n_insp"].max().reset_index(name="max_recurrence")

    del raw, recur_df
    gc.collect()

    # Join rat features
    if rat is not None and not rat.empty:
        r = rat.copy()
        r["camis"] = r["camis"].astype(str)
        rat_cols = ["camis"] + [c for c in ("rat_index", "pest_index") if c in r.columns]
        inspections = inspections.merge(r[rat_cols], on="camis", how="left")
    if "rat_index"  not in inspections.columns: inspections["rat_index"]  = np.nan
    if "pest_index" not in inspections.columns: inspections["pest_index"] = np.nan

    inspections = inspections.merge(max_recur, on="camis", how="left")
    inspections["max_recurrence"] = inspections["max_recurrence"].fillna(0).clip(upper=10)

    today = pd.Timestamp.now(tz="UTC").tz_localize(None)

    rows = []
    for camis_id, grp in inspections.groupby("camis", sort=False):
        grp    = grp.sort_values("inspection_date").reset_index(drop=True)
        latest = grp.iloc[-1]
        n      = len(grp)

        scores = grp["score"].tolist()
        grades = [g for g in grp["grade"].tolist()]

        consec_a = 0
        for g in reversed(grades):
            if g == "A":
                consec_a += 1
            else:
                break

        valid_scores = [s for s in scores if pd.notna(s)]
        if len(valid_scores) >= 2:
            score_delta = float(valid_scores[-1] - valid_scores[-2])
            win         = valid_scores[-4:]
            score_trend = (
                float(np.polyfit(np.arange(len(win), dtype=float), win, 1)[0])
                if len(win) >= 3 else score_delta
            )
        else:
            score_delta = score_trend = 0.0

        last_ts    = latest["inspection_date"]
        days_since = int((today - last_ts).days) if pd.notna(last_ts) else 0
        days_since = max(0, days_since)

        rows.append({
            "camis":             camis_id,
            "last_score":        float(latest["score"])            if pd.notna(latest["score"])                  else np.nan,
            "score_delta":       score_delta,
            "score_trend":       score_trend,
            "days_since_last":   float(min(days_since, 730)),
            "days_overdue":      float(max(0, days_since - 365)),
            "inspection_count":  float(min(n, 10)),
            "critical_fraction": float(latest["critical_fraction"]) if pd.notna(latest["critical_fraction"]) else 0.0,
            "n_violations":      float(latest["n_violations"])      if pd.notna(latest["n_violations"])      else 0.0,
            "has_mice_vio":      int(bool(latest["has_mice_vio"])),
            "has_temp_vio":      int(bool(latest["has_temp_vio"])),
            "has_vermin_vio":    int(bool(latest["has_vermin_vio"])),
            "consec_a":          float(consec_a),
            "max_recurrence":    float(latest["max_recurrence"])    if pd.notna(latest.get("max_recurrence")) else 0.0,
            "rat_index":         float(latest["rat_index"])         if pd.notna(latest.get("rat_index"))      else np.nan,
            "pest_index":        float(latest["pest_index"])        if pd.notna(latest.get("pest_index"))     else np.nan,
            "cuisine_group":     _cuisine_group(str(latest["cuisine"] or "")),
            "boro":              _boro_clean(str(latest["boro"]     or "")),
        })

    return pd.DataFrame(rows)


def precompute(model_path: str = MODEL_PATH) -> int:
    print(f"[scores] Loading model from {model_path} …")
    model = joblib.load(model_path)

    print(f"[scores] Loading raw inspections from {RAW_FILE} …")
    raw = pd.read_parquet(RAW_FILE)

    rat = None
    if os.path.exists(RAT_FILE):
        rat = pd.read_parquet(RAT_FILE)
        print(f"[scores] Rat index: {len(rat):,} rows")

    print("[scores] Building inference features …")
    feats = compute_inference_features(raw, rat)
    del raw, rat
    gc.collect()

    print(f"[scores] Running model.predict_proba on {len(feats):,} restaurants …")
    X     = feats[ALL_FEATURES]
    probs = model.predict_proba(X)[:, 1]

    # Free sklearn model immediately after inference
    del model
    gc.collect()

    out = pd.DataFrame({
        "camis":   feats["camis"].astype(str),
        "prob_bc": probs.astype("float32"),
    })

    os.makedirs(PARQUET_DIR, exist_ok=True)
    out.to_parquet(OUT_FILE, index=False)
    print(f"[scores] Wrote {len(out):,} scores → {OUT_FILE}")
    return len(out)


if __name__ == "__main__":
    precompute()
