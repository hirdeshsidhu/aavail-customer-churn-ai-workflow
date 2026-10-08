import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from app.data import load_json_directory,clean_data
from app.config import FIGURE_DIR
raw=load_json_directory(); df,target,audit=clean_data(raw)
if "country" in df:
    r=df.groupby("country")[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(9,5)); r.plot(kind="bar"); plt.ylabel("Churn rate"); plt.title("AAVAiL churn rate by country")
    plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"churn_by_country.png",dpi=160); plt.close()
m=raw.isna().sum().sort_values(ascending=False); m=m[m>0]
plt.figure(figsize=(9,5))
if len(m): m.plot(kind="bar"); plt.ylabel("Missing values")
else: plt.text(.5,.5,"No missing values detected",ha="center",va="center")
plt.title("Missing values at ingestion"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"missing_values.png",dpi=160); plt.close()
print("EDA COMPLETE")
