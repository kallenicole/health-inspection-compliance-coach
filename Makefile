.PHONY: dev api etl features seed docker-run

dev:
	uvicorn api.main:app --reload --port 8080

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8080

etl:
	python etl/nyc_inspections_etl.py

features:
	python etl/feature_engineering.py

seed:
	python etl/feature_engineering.py

docker-run:
	docker build -t hicc-api -f docker/Dockerfile .
	docker run -p 8080:8080 --env-file .env hicc-api

.PHONY: rats
rats:
	python etl/rodent_index.py