# DineSafe NYC — Health Inspection Compliance Coach

## Live links
**API docs (Swagger):** https://hicc-api-srf7acimsa-uc.a.run.app/docs  
**Health check:** https://hicc-api-srf7acimsa-uc.a.run.app/health

**Frontend demo:** https://hicc-web-kalle-georgievs-projects.vercel.app/

Predicts next-inspection risk (B/C vs. A) and top two likely next violation categories for NYC restaurants.  
FastAPI API, nightly ETL to Parquet, deployable to Cloud Run (scale to zero).

See the Beginner Install Guide in the repo issues or ask the assistant for the latest steps.


> ⚠️ Admin endpoints require a secret header (`X-Admin-Token`). Do **not** share your token.

---

## What this is:
- **Level-3 portfolio project**: live ETL ➜ feature store ➜ API ➜ lightweight UI.
- **Heuristic MVP**: blends last inspection results with a **local rat pressure** feature:
  - 311 “Rodent” complaints (last 180 days) near the restaurant
  - DOHMH rat inspection **failures** (last 365 days) near the restaurant  
  - Combined & normalized into `rat_index ∈ [0,1]` (quantile scaled). This slightly bumps:
    - overall `prob_bc` (probability of B/C next inspection)
    - mice violation (`04L`) probability

---

## Quickstart (local)

```bash
# From repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt

# (optional) set a NYC Open Data app token for higher API limits
export NYC_APP_TOKEN="...your token..."

# Build rat features into ./data/parquet (also run your inspections ETL/seed if you have it)
FEATURE_STORE_DIR="$PWD/data/parquet" python etl/rodent_index.py

# Run API
make dev   # or: uvicorn api.main:app --reload --port 8080
# Open: http://127.0.0.1:8080/docs
```

Deploy (Google Cloud Run)
Build the image on Cloud Build and deploy to Cloud Run (no local Docker needed):

# Build
gcloud builds submit --tag us-central1-docker.pkg.dev/health-inspection-demo/hicc/hicc-api:local .

# Deploy
```gcloud run deploy hicc-api \
  --image us-central1-docker.pkg.dev/health-inspection-demo/hicc/hicc-api:local \
  --region us-central1 --platform managed --allow-unauthenticated \
  --set-env-vars FEATURE_STORE_DIR="/tmp",BAKED_FEATURE_DIR="/app/data/parquet",ADMIN_TOKEN="$SECRET",NYC_APP_TOKEN="$NYC_APP_TOKEN" \
  --min-instances=0 --max-instances=3 --memory=1Gi --cpu=1 --timeout=900```
```
Nightly refresh (recommended)
Create a Cloud Scheduler HTTP job (cron 0 3 * * *, timezone America/Chicago) that calls:

```POST https://hicc-api-srf7acimsa-uc.a.run.app/admin/refresh```
```Header: X-Admin-Token: <your-secret>```

# API
Health
```GET /health
→ {"status":"ok"}

Search restaurants
GET /search?name=pizza
→ [
  { "camis":"50000000","name":"PIZZA PLACE","address":"123 MAIN ST 10001","boro":"MANHATTAN" },
  ...
]
```

Score a Restaurant
```POST /score
{"camis":"50117047"}
→ {
  "camis": "50117047",
  "prob_bc": 0.28,
  "predicted_points": 10.0,
  "top_reasons": ["Limited history"],
  "top_violation_probs": [
    {"code":"20-06","probability":1.0,"label":"Current letter grade or Grade Pending card not posted"}
  ],
  "model_version": "heuristic-fallback-0.1",
  "data_version": "runtime",
  "last_inspection_date": "2025-05-13",
  "last_points": null,
  "last_grade": null,

  "rat_index": 0.68,
  "rat311_cnt_180d_k1": 63,
  "ratinsp_fail_365d_k1": 0
}
```

# Admin

Refresh data & features
```POST /admin/refresh
Header: X-Admin-Token: <your-secret>
→ { "ok": true, "steps": ["inspections_seed_ok","rat_index_ok","rat_features_reloaded:####"] }
```

Check in-memory rat features for a CAMIS
```GET /admin/ratpeek?camis=50117047
→ { "feature_dir":"/tmp","has":true,"value":{ "rat_index":0.68, "rat311_cnt_180d_k1":63, "ratinsp_fail_365d_k1":0 } }
```

Check raw parquet for CAMIS + coordinates
```GET /admin/rawpeek?camis=50117047
→ { "path":"/app/data/parquet/inspections_raw.parquet","present":true,"lat":..., "lon":... }
```

# Configuration
Environment variables:
```FEATURE_STORE_DIR — runtime parquet dir (Cloud Run: /tmp)
BAKED_FEATURE_DIR — image-baked parquet fallback (/app/data/parquet)
ADMIN_TOKEN — required for /admin/refresh
NYC_APP_TOKEN — optional Socrata token for NYC Open Data
RATS_DAYS_311 (default 180), RATS_DAYS_INSP (default 365)
RAT_H3_RES (default 9)
```

Notes & limitations
- This is a heuristic MVP (not a trained model yet).
- Rat features require lat/lon in the inspections dataset.
- Some CAMIS are pre-seeded; others use the heuristic fallback.

License: MIT

