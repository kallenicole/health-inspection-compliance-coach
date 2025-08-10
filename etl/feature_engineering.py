import os, json, pandas as pd

FEATURE_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
DEMO_SEED_FILE = os.getenv("DEMO_SEED_FILE", "./data/demo_seed.json")

def load_raw() -> pd.DataFrame:
    path = os.path.join(FEATURE_DIR, "inspections_raw.parquet")
    return pd.read_parquet(path)

def engineer_baseline(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["camis","inspection_date","grade","score","violation_code","violation_description","dba"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df.sort_values(["camis","inspection_date"]).copy()
    last = df.groupby("camis").tail(1)
    last["score"] = pd.to_numeric(last["score"], errors="coerce").fillna(0)
    last["prob_bc"] = (last["score"] / 28.0).clip(0, 0.95)
    last["top_violation_probs"] = last.apply(
        lambda r: [
            {"code":"04M","probability":min(0.90,0.5+r["prob_bc"]/2),"label":"Food not held at proper temp"},
            {"code":"04L","probability":min(0.80,0.4+r["prob_bc"]/2),"label":"Evidence of mice"},
            {"code":"10F","probability":min(0.60,0.3+r["prob_bc"]/2),"label":"Personal cleanliness"}
        ], axis=1)
    last["top_reasons"] = last.apply(
        lambda r: [f"Last points: {int(r['score'])}",
                   "Recent temp/holding issues" if r["prob_bc"]>0.4 else "Consistent A history"], axis=1)
    return last[["camis","prob_bc","score","top_violation_probs","top_reasons","dba"]]

def build_demo_seed(features: pd.DataFrame, n: int = 25):
    os.makedirs(os.path.dirname(DEMO_SEED_FILE), exist_ok=True)
    sample = features.dropna(subset=["camis"]).head(n)
    seed = {}
    for _, r in sample.iterrows():
        seed[str(r["camis"])] = {
            "camis": str(r["camis"]),
            "prob_bc": float(round(r["prob_bc"], 3)),
            "predicted_points": float(round(r["score"] if pd.notnull(r["score"]) else 12.0, 1)),
            "top_reasons": list(r["top_reasons"]),
            "top_violation_probs": r["top_violation_probs"],
            "model_version": "0.1.0",
            "data_version": "demo-seed"
        }
    with open(DEMO_SEED_FILE, "w") as f:
        json.dump(seed, f, indent=2)
    print(f"Wrote demo seed with {len(seed)} examples to {DEMO_SEED_FILE}")

if __name__ == "__main__":
    raw = load_raw()
    feats = engineer_baseline(raw)
    build_demo_seed(feats, n=25)
