import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import LOG_PATH
from app.logging_utils import read_logs
logs=read_logs(LOG_PATH)
print("Logged predictions:",len(logs))
if logs: print("Average predicted churn probability:",round(sum(x["probability"] for x in logs)/len(logs),4))
