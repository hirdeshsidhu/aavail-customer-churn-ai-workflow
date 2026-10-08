from pathlib import Path
import json
import pandas as pd
import numpy as np
from .config import RAW_DIR

TARGET_CANDIDATES=["churn","Churn","is_churn","churned"]

def demo_data(n=1200,seed=42):
    rng=np.random.default_rng(seed)
    country=rng.choice(["Singapore","United Kingdom","USA","EIRE","Germany"],n,p=[.18,.27,.25,.12,.18])
    tenure=rng.integers(1,25,n)
    watch=np.maximum(0,rng.normal(18,9,n))
    sessions=np.maximum(1,rng.poisson(12,n))
    support=rng.poisson(1,n)
    plan=rng.choice(["basic","standard","premium"],n,p=[.42,.38,.20])
    risk=1.2-.055*watch-.035*sessions-.025*tenure+.45*(plan=="basic")+.25*support+.12*(country=="Singapore")
    prob=1/(1+np.exp(-risk))
    churn=rng.binomial(1,np.clip(prob,.02,.95))
    return pd.DataFrame({"country":country,"tenure_months":tenure,"watch_hours":watch.round(2),
                         "sessions":sessions,"support_contacts":support,"plan":plan,"churn":churn})

def load_json_directory(directory=RAW_DIR):
    records=[]
    for path in sorted(Path(directory).glob("*.json")):
        with open(path,encoding="utf-8") as f: obj=json.load(f)
        if isinstance(obj,list): records.extend(obj)
        elif isinstance(obj,dict) and isinstance(obj.get("data"),list): records.extend(obj["data"])
        elif isinstance(obj,dict): records.append(obj)
    return pd.json_normalize(records) if records else demo_data()

def identify_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns: return c
    raise ValueError("No churn target found")

def clean_data(df):
    df=df.copy(); target=identify_target(df)
    missing_before={c:int(df[c].isna().sum()) for c in df.columns}
    for c in df.columns:
        if c==target: continue
        if pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(df[c].median())
        else:
            m=df[c].mode(dropna=True); df[c]=df[c].fillna(m.iloc[0] if len(m) else "Unknown")
    before=len(df); df=df.drop_duplicates().reset_index(drop=True)
    audit={"rows_before":before,"rows_after":len(df),"duplicates_removed":before-len(df),
           "missing_before":missing_before,"missing_after":{c:int(df[c].isna().sum()) for c in df.columns},
           "dropped_columns":[],"target":target}
    return df,target,audit
