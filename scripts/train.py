import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.data import load_json_directory,clean_data
from app.model import train_models
df,target,_=clean_data(load_json_directory())
results,best=train_models(df,target)
print("MODEL COMPARISON")
for k,v in results.items(): print(k,v)
print("BEST MODEL:",best)
