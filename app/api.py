from fastapi import FastAPI,HTTPException
from .config import LOG_PATH
from .data import load_json_directory,clean_data
from .model import load_model
from .logging_utils import log_prediction,read_logs

app=FastAPI(title="AAVAiL Churn Prediction API",version="1.0.0")

def rows(country=None):
    df,target,_=clean_data(load_json_directory())
    if country:
        if "country" not in df: raise HTTPException(400,"country field unavailable")
        df=df[df.country.astype(str).str.lower()==country.lower()]
    return df,target

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/predict")
def predict(country:str|None=None):
    df,target=rows(country)
    if df.empty: raise HTTPException(404,"No records found")
    X=df.drop(columns=[target]); model=load_model()
    p=model.predict_proba(X)[:,1]; pred=(p>=.5).astype(int)
    log_prediction(LOG_PATH,{"country":country,"rows":len(X)},int(pred.mean()>=.5),float(p.mean()))
    return {"country":country or "all","rows":len(X),"predicted_churn_rate":float(pred.mean()),
            "average_churn_probability":float(p.mean())}

@app.get("/monitor")
def monitor():
    logs=read_logs(LOG_PATH)
    return {"logged_predictions":len(logs),
            "average_probability":None if not logs else sum(x["probability"] for x in logs)/len(logs)}
