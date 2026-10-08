import pandas as pd
from app.model import train_models
def test_multiple_models_are_compared(tmp_path,monkeypatch):
    import app.model as m
    monkeypatch.setattr(m,"MODEL_PATH",tmp_path/"model.joblib")
    monkeypatch.setattr(m,"METRICS_PATH",tmp_path/"metrics.json")
    df=pd.DataFrame({"country":["SG","UK","US","SG","UK","US"]*10,
                     "watch_hours":[2,20,15,4,25,18]*10,
                     "sessions":[1,10,8,2,12,9]*10,
                     "plan":["basic","premium","standard","basic","premium","standard"]*10,
                     "churn":[1,0,0,1,0,0]*10})
    results,best=train_models(df,"churn")
    assert set(["baseline","logistic_regression","random_forest"]).issubset(results)
    assert best in ["logistic_regression","random_forest"]
