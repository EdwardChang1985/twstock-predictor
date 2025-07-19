import pickle
import pandas as pd
from data.fetch_twse import get_stock_features

def predict_stock(stock_code: str):
    features = get_stock_features(stock_code)
    model = pickle.load(open("models/xgb_model.pkl", "rb"))
    preds = model.predict(features)
    return {
        "1D": float(preds[0]),
        "3D": float(preds[1]),
        "5D": float(preds[2])
    }
