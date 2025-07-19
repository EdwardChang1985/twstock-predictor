import pandas as pd
import xgboost as xgb
import pickle
from data.fetch_twse import get_historical_data

def train_model(stock_code):
    df = get_historical_data(stock_code)
    X = df.drop(columns=["1D", "3D", "5D"])
    for target in ["1D", "3D", "5D"]:
        y = df[target]
        model = xgb.XGBRegressor()
        model.fit(X, y)
        pickle.dump(model, open(f"models/xgb_model_{target}.pkl", "wb"))
