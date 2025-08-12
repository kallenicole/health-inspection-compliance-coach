# Dockerfile (at repo root)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# App code
COPY api /app/api
COPY etl /app/etl
COPY models /app/models

# Baked data (fallbacks)
RUN mkdir -p /app/data/parquet
COPY data /app/data

# Defaults (Cloud Run will override FEATURE_STORE_DIR to /tmp)
ENV FEATURE_STORE_DIR="/tmp" \
    BAKED_FEATURE_DIR="/app/data/parquet"

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]


