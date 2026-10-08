FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
COPY tests tests
COPY data data
COPY reports reports
COPY artifacts artifacts
COPY run_tests.sh .
RUN chmod +x run_tests.sh
RUN python scripts/ingest.py && python scripts/eda.py && python scripts/train.py
EXPOSE 8000
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000"]
