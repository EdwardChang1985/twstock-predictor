from fastapi import FastAPI
from model.predict import predict_stock
from scheduler.cron import start_scheduler

app = FastAPI()

@app.get("/predict/{stock_code}")
def predict(stock_code: str):
    result = predict_stock(stock_code)
    return result

start_scheduler()
