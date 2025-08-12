FROM python:3.11-slim

WORKDIR /app

# optional but nice to have SSL certs & curl
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && python -c "import requests, h3, fastapi, uvicorn; print('deps ok')"

# copy app code
COPY api /app/api
COPY etl /app/etl
COPY models /app/models
COPY data /app/data

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]


