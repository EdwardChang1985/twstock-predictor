from apscheduler.schedulers.background import BackgroundScheduler
from model.train import train_model

scheduler = BackgroundScheduler()

@scheduler.scheduled_job("cron", hour=5)
def scheduled_job():
    for stock in ["2330", "2317", "2603"]:
        train_model(stock)

def start_scheduler():
    scheduler.start()
