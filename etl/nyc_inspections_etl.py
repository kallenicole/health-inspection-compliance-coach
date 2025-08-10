import os, httpx, pandas as pd
from datetime import datetime, timedelta

BASE_URL = os.getenv("SOCRATA_BASE_URL", "https://data.cityofnewyork.us")
DATASET = os.getenv("SOCRATA_DATASET_ID", "43nn-pn8j")
FEATURE_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")

def fetch(limit=50000) -> pd.DataFrame:
    url = f"{BASE_URL}/resource/{DATASET}.json"
    params = {"$limit": limit, "$order": "inspection_date DESC"}
    r = httpx.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if not df.empty and "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    return df

def write_parquet(df: pd.DataFrame, name: str):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    path = os.path.join(FEATURE_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")

if __name__ == "__main__":
    df = fetch()
    if df.empty:
        print("No rows fetched. Check dataset id / query.")
    else:
        write_parquet(df, "inspections_raw")
