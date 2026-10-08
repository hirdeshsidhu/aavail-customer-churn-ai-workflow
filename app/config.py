from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
RAW_DIR=ROOT/"data"/"raw"
ARTIFACT_DIR=ROOT/"artifacts"
FIGURE_DIR=ROOT/"reports"/"figures"
MODEL_PATH=ARTIFACT_DIR/"model.joblib"
METRICS_PATH=ARTIFACT_DIR/"metrics.json"
LOG_PATH=Path(os.getenv("AAVAIL_LOG_PATH", ARTIFACT_DIR/"predictions.jsonl"))
for p in (RAW_DIR,ARTIFACT_DIR,FIGURE_DIR): p.mkdir(parents=True,exist_ok=True)
