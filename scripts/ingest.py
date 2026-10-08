import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.data import load_json_directory,clean_data
from app.config import ARTIFACT_DIR
df=load_json_directory(); clean,target,audit=clean_data(df)
clean.to_csv(ARTIFACT_DIR/"clean_data.csv",index=False)
print("INGESTION COMPLETE"); print(audit)
