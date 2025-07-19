import pandas as pd
import xgboost as xgb
import pickle
from data.fetch_twse import get_historical_data
import os
import numpy as np

# 單模型訓練並整合輸出一個模型
class MultiOutputModel:
    def __init__(self):
        self.models = {}

    def fit(self, X, y_dict):
        for label in y_dict:
            model = xgb.XGBRegressor()
            model.fit(X, y_dict[label])
            self.models[label] = model

    def predict(self, X):
        results = []
        for label in ["1D", "3D", "5D"]:
            pred = self.models[label].predict(X)
            results.append(pred[0])
        return np.array(results)

def train_model(stock_code="2330"):
    df = get_historical_data(stock_code)
    X = df.drop(columns=["1D", "3D", "5D"])
    y_dict = {label: df[label] for label in ["1D", "3D", "5D"]}
    model = MultiOutputModel()
    model.fit(X, y_dict)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model("2330")
