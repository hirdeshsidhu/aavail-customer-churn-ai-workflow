# AAVAiL Customer Churn Prediction & Deployment

Complete AI workflow: ingestion, missing-value handling, EDA, baseline/model comparison,
FastAPI prediction API, prediction logging, monitoring, isolated unit tests, and Docker.

## Run
pip install -r requirements.txt
python scripts/ingest.py
python scripts/eda.py
python scripts/train.py
uvicorn app.api:app --host 0.0.0.0 --port 8000

## Docker
docker compose up --build

## Tests
bash run_tests.sh

## API
GET /health
GET /predict?country=Singapore
GET /predict
GET /monitor

Put the course AAVAiL JSON files in data/raw/. The project has a deterministic demo fallback
so the Docker image remains runnable before the course files are mounted. For the graded
submission, replace the fallback with the supplied course data and regenerate artifacts.
