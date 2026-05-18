"""
Train a GradientBoosting classifier on NYC restaurant inspection history.

For each restaurant, we use features from inspection N to predict whether
inspection N+1 results in a B or C grade (score > 13).  Only "Cycle
Inspection" records (Initial + Re-inspection) are used — these are the
graded inspections that result in public A/B/C postings.

Usage:
  # from project root, with venv active:
  python -m etl.train_model             # train and save
  python -m etl.train_model --eval      # cross-val AUC first, then save
"""

import argparse
import gc
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
RAT_FILE = os.path.join(PARQUET_DIR, "rat_index.parquet")
RAW_FILE = os.path.join(PARQUET_DIR, "inspections_raw.parquet")
MODEL_OUT = os.getenv("MODEL_PATH", "./models/score_model.joblib")

# ---------------------------------------------------------------------------
# Cuisine grouping — kept in sync with score.py
# ---------------------------------------------------------------------------
CUISINE_GROUPS = {
    "Chinese": "asian", "Japanese": "asian", "Korean": "asian",
    "Thai": "asian", "Vietnamese": "asian", "Asian": "asian",
    "Taiwanese": "asian", "Filipino": "asian", "Chinese/Japanese": "asian",
    "Sushi": "asian",
    "Mexican": "latin", "Caribbean": "latin", "Latin (Cuban, Dominican, Puerto Rican, South & Central American)": "latin",
    "Dominican": "latin", "Peruvian": "latin",
    "American": "american", "Hamburgers": "american",
    "Sandwiches": "american", "Sandwiches/Salads/Mixed Buffet": "american",
    "Steakhouse": "american",
    "Italian": "italian", "Pizza": "italian", "Pizza/Italian": "italian",
    "Chicken": "fast_food", "Hotdogs": "fast_food",
    "Mediterranean": "mediterranean", "Middle Eastern": "mediterranean",
    "Turkish": "mediterranean", "Greek": "mediterranean", "Moroccan": "mediterranean",
    "Indian": "indian", "Pakistani": "indian", "Bangladeshi": "indian",
    "Afghan": "indian",
    "Café/Coffee/Tea": "cafe", "Bakery": "cafe", "Donuts": "cafe",
    "Juice, Smoothies, Fruit Salads": "cafe",
}

# Inspection types that generate public grades
CYCLE_TYPES = {
    "Cycle Inspection / Initial Inspection",
    "Cycle Inspection / Re-inspection",
}

NUMERIC_FEATURES = [
    "last_score",
    "score_delta",
    "score_trend",
    "days_since_last",
    "days_overdue",
    "inspection_count",
    "critical_fraction",
    "n_violations",
    "has_mice_vio",
    "has_temp_vio",
    "has_vermin_vio",
    "consec_a",
    "max_recurrence",
    "rat_index",
    "pest_index",
]
CATEGORICAL_FEATURES = ["cuisine_group", "boro"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Grade helpers
# ---------------------------------------------------------------------------
def _infer_grade(grade_raw, score):
    if str(grade_raw).strip().upper() in ("A", "B", "C"):
        return str(grade_raw).strip().upper()
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s <= 13:
        return "A"
    if s <= 27:
        return "B"
    return "C"


def _cuisine_group(raw: str) -> str:
    return CUISINE_GROUPS.get(str(raw).strip(), "other")


def _boro_clean(raw: str) -> str:
    b = str(raw).strip()
    return b if b in ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island") else "other"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_features(raw: pd.DataFrame, rat) -> pd.DataFrame:
    """
    Returns a DataFrame with ALL_FEATURES columns plus 'grade_bc' target,
    'camis', and 'inspection_date' (of the PREDICTED inspection).
    """
    raw = raw.copy()
    raw["inspection_date"] = pd.to_datetime(raw["inspection_date"], errors="coerce")
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
    raw["camis"] = raw["camis"].astype(str)
    raw["violation_code"] = raw.get("violation_code", pd.Series(dtype=str)).fillna("").astype(str)
    raw["critical_flag"] = raw.get("critical_flag", pd.Series(dtype=str)).astype(str)

    # Keep only cycle inspections for both aggregation and sequencing
    if "inspection_type" in raw.columns:
        raw = raw[raw["inspection_type"].isin(CYCLE_TYPES)].copy()

    # Per-row flags
    raw["is_critical"] = raw["critical_flag"].str.strip() == "Critical"
    raw["is_mice_vio"] = raw["violation_code"] == "04L"
    raw["is_temp_vio"] = raw["violation_code"] == "04M"
    raw["is_vermin_vio"] = raw["violation_code"] == "08A"

    # --- aggregate to one row per (camis, inspection_date) ---
    def first_valid_grade(s):
        for v in s.dropna():
            g = str(v).strip().upper()
            if g in ("A", "B", "C"):
                return g
        return None

    inspections = (
        raw.groupby(["camis", "inspection_date"], sort=False)
        .agg(
            score=("score", "first"),
            grade_raw=("grade", first_valid_grade),
            n_violations=("violation_code", lambda x: (x != "").sum()),
            n_critical=("is_critical", "sum"),
            has_mice_vio=("is_mice_vio", "any"),
            has_temp_vio=("is_temp_vio", "any"),
            has_vermin_vio=("is_vermin_vio", "any"),
            cuisine=("cuisine_description", "first"),
            boro=("boro", "first"),
        )
        .reset_index()
    )
    inspections["critical_fraction"] = (
        inspections["n_critical"] / inspections["n_violations"].replace(0, 1)
    ).clip(0, 1)
    inspections["grade"] = inspections.apply(
        lambda r: _infer_grade(r["grade_raw"], r["score"]), axis=1
    )
    inspections["grade_bc"] = inspections["grade"].isin(["B", "C"]).astype(int)
    inspections = inspections.sort_values(["camis", "inspection_date"])

    # --- join rat features ---
    if rat is not None and not rat.empty:
        rat = rat.copy()
        rat["camis"] = rat["camis"].astype(str)
        rat_cols = ["camis"] + [c for c in ("rat_index", "pest_index") if c in rat.columns]
        inspections = inspections.merge(rat[rat_cols], on="camis", how="left")
    if "rat_index" not in inspections.columns:
        inspections["rat_index"] = np.nan
    if "pest_index" not in inspections.columns:
        inspections["pest_index"] = np.nan

    # --- per-camis violation recurrence (how many inspections each code appeared in) ---
    recur_df = (
        raw[raw["violation_code"] != ""]
        .dropna(subset=["violation_code", "inspection_date"])
        .groupby(["camis", "violation_code"])["inspection_date"]
        .nunique()
        .reset_index(name="n_inspections")
    )
    max_recur = recur_df.groupby("camis")["n_inspections"].max().reset_index(name="max_recurrence")
    inspections = inspections.merge(max_recur, on="camis", how="left")
    inspections["max_recurrence"] = inspections["max_recurrence"].fillna(0).clip(upper=10)

    del raw, recur_df, max_recur
    gc.collect()

    # --- build (prev_inspection → next_outcome) training samples ---
    rows = []
    for camis_id, grp in inspections.groupby("camis", sort=False):
        grp = grp.sort_values("inspection_date").reset_index(drop=True)
        if len(grp) < 2:
            continue

        score_series = grp["score"].tolist()
        grade_series = grp["grade"].tolist()

        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]
            curr = grp.iloc[i]

            # Consecutive A grades ending at prev (exclusive of current)
            consec_a = 0
            for j in range(i - 1, -1, -1):
                if grade_series[j] == "A":
                    consec_a += 1
                else:
                    break

            # Score delta and trend
            prior_scores = [s for s in score_series[:i] if pd.notna(s)]
            if len(prior_scores) >= 2:
                score_delta = float(prior_scores[-1] - prior_scores[-2])
                recent_window = prior_scores[-4:]
                if len(recent_window) >= 3:
                    xs = np.arange(len(recent_window), dtype=float)
                    score_trend = float(np.polyfit(xs, recent_window, 1)[0])
                else:
                    score_trend = score_delta
            else:
                score_delta = 0.0
                score_trend = 0.0

            days_since = (curr["inspection_date"] - prev["inspection_date"]).days
            days_overdue = float(max(0, days_since - 365))

            rows.append({
                "last_score":        float(prev["score"]) if pd.notna(prev["score"]) else np.nan,
                "score_delta":       score_delta,
                "score_trend":       score_trend,
                "days_since_last":   float(min(days_since, 730)),
                "days_overdue":      days_overdue,
                "inspection_count":  float(min(i, 10)),
                "critical_fraction": float(prev["critical_fraction"]) if pd.notna(prev["critical_fraction"]) else 0.0,
                "n_violations":      float(prev["n_violations"]) if pd.notna(prev["n_violations"]) else 0.0,
                "has_mice_vio":      int(bool(prev["has_mice_vio"])),
                "has_temp_vio":      int(bool(prev["has_temp_vio"])),
                "has_vermin_vio":    int(bool(prev["has_vermin_vio"])),
                "consec_a":          float(consec_a),
                "max_recurrence":    float(prev["max_recurrence"]) if pd.notna(prev["max_recurrence"]) else 0.0,
                "rat_index":         float(prev["rat_index"]) if pd.notna(prev.get("rat_index")) else np.nan,
                "pest_index":        float(prev["pest_index"]) if pd.notna(prev.get("pest_index")) else np.nan,
                "cuisine_group":     _cuisine_group(prev["cuisine"] or ""),
                "boro":              _boro_clean(prev["boro"] or ""),
                # target + metadata
                "grade_bc":          int(curr["grade_bc"]),
                "camis":             camis_id,
                "inspection_date":   curr["inspection_date"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    base_clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
    return Pipeline([("prep", preprocessor), ("clf", calibrated)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(eval_mode: bool = False) -> Pipeline:
    print(f"[train] Loading {RAW_FILE} …")
    raw = pd.read_parquet(RAW_FILE)
    print(f"[train] {len(raw):,} raw rows, {raw['camis'].nunique():,} restaurants")

    rat = None
    if os.path.exists(RAT_FILE):
        rat = pd.read_parquet(RAT_FILE)
        print(f"[train] Rat index: {len(rat):,} rows")
    else:
        print("[train] No rat_index.parquet — rat features will be null")

    print("[train] Building features …")
    df = build_features(raw, rat)
    del raw, rat
    gc.collect()

    pos_rate = df["grade_bc"].mean()
    print(f"[train] {len(df):,} training samples | positive rate (B/C): {pos_rate:.1%}")

    X = df[ALL_FEATURES]
    y = df["grade_bc"]

    if eval_mode:
        print("[train] 5-fold stratified cross-validation …")
        pipeline = make_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
            print(f"  fold {fold}: AUC={auc:.4f}")
            aucs.append(auc)
        print(f"[train] Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print("[train] Fitting final model on all data …")
    pipeline = make_pipeline()
    pipeline.fit(X, y)

    # Attach feature metadata so inference code can reconstruct the same vector
    pipeline.numeric_features = NUMERIC_FEATURES
    pipeline.categorical_features = CATEGORICAL_FEATURES
    pipeline.all_features = ALL_FEATURES

    os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT, compress=3)
    size_kb = os.path.getsize(MODEL_OUT) // 1024
    print(f"[train] Saved → {MODEL_OUT} ({size_kb:,} KB)")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NYC inspection risk model")
    parser.add_argument("--eval", action="store_true", help="Run cross-val before final fit")
    args = parser.parse_args()
    train(eval_mode=args.eval)
