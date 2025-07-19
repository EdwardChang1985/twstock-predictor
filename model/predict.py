import pickle
import pandas as pd
from data.fetch_twse import get_stock_features
import os

def predict_stock(stock_code: str):
    features = get_stock_features(stock_code)
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        return {"error": "模型尚未訓練，請稍後再試。"}
    model = pickle.load(open(model_path, "rb"))
    preds = model.predict(features)
    return {
        "1D": float(preds[0]),
        "3D": float(preds[1]),
        "5D": float(preds[2])
    }
