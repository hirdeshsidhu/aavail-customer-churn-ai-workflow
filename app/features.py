import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def build_preprocessor(X):
    num=X.select_dtypes(include="number").columns.tolist()
    cat=X.select_dtypes(exclude="number").columns.tolist()
    np=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cp=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num",np,num),("cat",cp,cat)])
