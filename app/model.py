from pathlib import Path
import json,joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from .features import build_preprocessor
from .config import MODEL_PATH,METRICS_PATH

def train_models(df,target):
    X=df.drop(columns=[target]); y=df[target].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={"baseline":DummyClassifier(strategy="most_frequent"),
                "logistic_regression":LogisticRegression(max_iter=1000,class_weight="balanced"),
                "random_forest":RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced",n_jobs=-1)}
    results={}; fitted={}
    for name,est in candidates.items():
        pipe=Pipeline([("preprocessor",build_preprocessor(Xtr)),("model",est)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        results[name]={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
                       "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),
                       "roc_auc":roc_auc_score(yte,prob)}
        fitted[name]=pipe
    names=["logistic_regression","random_forest"]
    best=max(names,key=lambda n:results[n]["roc_auc"])
    joblib.dump(fitted[best],MODEL_PATH)
    with open(METRICS_PATH,"w") as f: json.dump({"best_model":best,"models":results},f,indent=2)
    return results,best

def load_model():
    if not Path(MODEL_PATH).exists(): raise FileNotFoundError("Run scripts/train.py first")
    return joblib.load(MODEL_PATH)
