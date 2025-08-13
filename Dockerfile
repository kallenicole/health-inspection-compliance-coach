# Dockerfile (at repo root)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# OS deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt \
    && pip install --no-cache-dir requests h3

# App code
COPY api /app/api
COPY etl /app/etl
COPY models /app/models
COPY data /app/data

# Bake feature store (so we have a cold-start fallback)
# Ensure you have these locally before building:
#   data/parquet/rat_index.parquet
#   data/parquet/inspections_raw.parquet
RUN mkdir -p /app/data/parquet
COPY data/parquet /app/data/parquet

# Environment defaults
ENV BAKED_FEATURE_DIR="/app/data/parquet" \
    BAKED_DEMO_SEED_FILE="/app/data/demo_seed.json" \
    FEATURE_STORE_DIR="/tmp"

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]


