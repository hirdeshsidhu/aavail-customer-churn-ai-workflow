import pandas as pd
from app.data import clean_data
def test_missing_values_are_handled():
    df=pd.DataFrame({"country":["SG",None,"UK"],"watch_hours":[10,None,20],"churn":[1,0,1]})
    clean,target,audit=clean_data(df)
    assert clean.isna().sum().sum()==0
    assert target=="churn"
    assert audit["missing_after"]["country"]==0
