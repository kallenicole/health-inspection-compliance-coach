# etl/rodent_index.py
import os, time, datetime as dt, requests
import pandas as pd
import h3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Where to read/write at runtime
PARQUET_DIR = os.getenv("FEATURE_STORE_DIR", "./data/parquet")       # e.g., /tmp on Cloud Run
BAKED_DIR   = os.getenv("BAKED_FEATURE_DIR", "/app/data/parquet")    # read-only inside the image

RAW_TMP   = os.path.join(PARQUET_DIR, "inspections_raw.parquet")     # preferred at runtime
RAW_BAKED = os.path.join(BAKED_DIR,   "inspections_raw.parquet")     # fallback if /tmp is empty
OUT_FILE  = os.path.join(PARQUET_DIR, "rat_index.parquet")

# Public NYC datasets
SODA_311  = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"  # 311 (Rodent)
SODA_RATS = "https://data.cityofnewyork.us/resource/p937-wjvj.json"  # DOHMH rodent inspections

# H3 + time windows
RES        = int(os.getenv("RAT_H3_RES", "9"))       # ~150–200m cells
DAYS_311   = int(os.getenv("RATS_DAYS_311", "180"))  # window for 311
DAYS_INSP  = int(os.getenv("RATS_DAYS_INSP", "365")) # window for rat inspections

# Networking
PAGE_LIMIT   = int(os.getenv("SODA_PAGE_LIMIT", "10000"))
READ_TIMEOUT = int(os.getenv("SODA_READ_TIMEOUT", "120"))
RETRIES      = int(os.getenv("SODA_RETRIES", "5"))
BACKOFF      = float(os.getenv("SODA_BACKOFF", "0.6"))

def _session():
    s = requests.Session()
    retry = Retry(
        total=RETRIES,
        backoff_factor=BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    token = os.getenv("NYC_APP_TOKEN")
    if token:
        s.headers.update({"X-App-Token": token})
    return s

def _paged_get(url, params, limit=PAGE_LIMIT, max_rows=200000):
    sess = _session()
    rows, offset = [], 0
    while True:
        qp = dict(params); qp["$limit"] = limit; qp["$offset"] = offset
        r = sess.get(url, params=qp, timeout=READ_TIMEOUT)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        offset += limit
        if offset >= max_rows:
            break
        time.sleep(0.2)  # be nice to the API
    return rows

def fetch_311_rodents(since: dt.datetime) -> pd.DataFrame:
    where = f"complaint_type='Rodent' AND latitude IS NOT NULL AND created_date >= '{since.isoformat()}'"
    rows = _paged_get(SODA_311, {"$select":"created_date,latitude,longitude", "$where": where})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df["cell"] = df.apply(lambda r: h3.latlng_to_cell(r.lat, r.lon, RES), axis=1)
    return df[["created_date","cell"]]

def fetch_dohmh_rats(since: dt.datetime) -> pd.DataFrame:
    where = f"inspection_date >= '{since.isoformat()}' AND latitude IS NOT NULL"
    rows = _paged_get(SODA_RATS, {"$select":"inspection_date,result,latitude,longitude", "$where": where})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df["fail"] = df["result"].str.contains("Active Rat", case=False, na=False)
    df["cell"] = df.apply(lambda r: h3.latlng_to_cell(r.lat, r.lon, RES), axis=1)
    return df[["inspection_date","cell","fail"]]

def build_rat_features():
    os.makedirs(PARQUET_DIR, exist_ok=True)

    # Choose raw inspections parquet: prefer /tmp, fallback to baked inside image
    raw_path = RAW_TMP if os.path.exists(RAW_TMP) else RAW_BAKED
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No inspections_raw.parquet found at {RAW_TMP} or {RAW_BAKED}")

    since311 = dt.datetime.utcnow() - dt.timedelta(days=DAYS_311)
    sinceRats = dt.datetime.utcnow() - dt.timedelta(days=DAYS_INSP)

    print(f"[rat] Using raw inspections from: {raw_path}")
    print(f"[rat] Fetching 311 rodents since {since311.date()} …")
    df311 = fetch_311_rodents(since311)

    print(f"[rat] Fetching DOHMH rodent inspections since {sinceRats.date()} …")
    dfr = fetch_dohmh_rats(sinceRats)

    cnt311 = df311.groupby("cell").size().rename("cnt311").to_dict() if not df311.empty else {}
    cntR_fail = dfr[dfr["fail"]].groupby("cell").size().rename("cntR").to_dict() if not dfr.empty else {}

    raw = pd.read_parquet(raw_path, columns=["camis","latitude","longitude"]).dropna()
    raw = raw.drop_duplicates("camis", keep="last").copy()
    raw["cell"] = raw.apply(lambda r: h3.latlng_to_cell(float(r.latitude), float(r.longitude), RES), axis=1)

    def ring_sum(cell: str, src: dict, k: int = 1) -> int:
        total = 0
        for c in h3.grid_disk(cell, k):
            total += src.get(c, 0)
        return total

    out = raw[["camis","cell"]].copy()
    out["rat311_cnt_180d_k1"]   = out["cell"].apply(lambda c: ring_sum(c, cnt311, 1))
    out["ratinsp_fail_365d_k1"] = out["cell"].apply(lambda c: ring_sum(c, cntR_fail, 1))

    # robust 0–1 normalization
    def qnorm(s: pd.Series) -> pd.Series:
        if s.empty: return s
        q1, q9 = s.quantile(0.1), s.quantile(0.9)
        return ((s - q1) / (q9 - q1 + 1e-9)).clip(0, 1)

    out["rat_index"] = qnorm(0.7*out["rat311_cnt_180d_k1"] + 0.3*out["ratinsp_fail_365d_k1"])
    out = out.drop(columns=["cell"])
    out.to_parquet(OUT_FILE, index=False)
    print(f"[rat] Wrote {OUT_FILE} with {len(out):,} rows")

if __name__ == "__main__":
    build_rat_features()
