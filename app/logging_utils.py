import json
from datetime import datetime,timezone
from pathlib import Path
def log_prediction(path,payload,prediction,probability):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    rec={"timestamp":datetime.now(timezone.utc).isoformat(),"request":payload,
         "prediction":int(prediction),"probability":float(probability)}
    with open(path,"a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
    return rec
def read_logs(path):
    p=Path(path)
    if not p.exists(): return []
    with open(p,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
