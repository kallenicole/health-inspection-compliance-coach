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
